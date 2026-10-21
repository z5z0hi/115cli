from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from email.utils import formatdate
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import BinaryIO, Callable
from urllib.parse import urlparse, urlsplit, urlunsplit
import uuid
import xml.etree.ElementTree as ET

import httpcore
import httpx
from p115cipher import ecdh_aes_decrypt, make_upload_payload

from cli115.exceptions import InstantUploadNotAvailableError
from .base import (
    APP_USER_AGENT,
    APP_VERSION,
    APIClient,
    DEFAULT_USER_AGENT,
    Endpoint,
)

logger = logging.getLogger(__name__)

# 16 MB per chunk; also used as the threshold for switching to multipart upload
MULTIPART_UPLOAD_PART_SIZE = 16 * 1024 * 1024

_CHUNK_SIZE = 64 * 1024

# Refresh the STS token this long before its actual expiry time.
TOKEN_REFRESH_MARGIN = timedelta(minutes=5)

# How many times a single part may be retried after the STS token expired.
MAX_TOKEN_RETRIES = 3

# Directory where interrupted multipart upload sessions are recorded so they
# can be resumed later.  Overridable via the CLI115_UPLOAD_STATE_DIR env var.
DEFAULT_UPLOAD_STATE_DIR = Path.home() / ".115cli" / "uploads"


class OSSTokenExpiredError(OSError):
    """Raised when OSS rejects a request because the STS token has expired."""


class OSSSessionGoneError(OSError):
    """Raised when OSS rejects a request because the upload session
    (uploadId) no longer exists on the server."""


def _upload_state_dir() -> Path:
    return Path(os.environ.get("CLI115_UPLOAD_STATE_DIR") or DEFAULT_UPLOAD_STATE_DIR)


def _parse_token_expiration(token: dict) -> datetime | None:
    """Return the token's expiration as an aware datetime, if it carries one."""
    raw = token.get("Expiration")
    if not raw:
        return None
    try:
        expiration = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if expiration.tzinfo is None:
        expiration = expiration.replace(tzinfo=timezone.utc)
    return expiration


def _token_expires_soon(
    token: dict,
    *,
    margin: timedelta = TOKEN_REFRESH_MARGIN,
    now: datetime | None = None,
) -> bool:
    """Whether the STS token is expired or expires within ``margin``."""
    expiration = _parse_token_expiration(token)
    if expiration is None:
        return False
    if now is None:
        now = datetime.now(timezone.utc)
    return now >= expiration - margin


def _is_token_expired_body(body: bytes) -> bool:
    """Whether an OSS error body indicates the security token has expired."""
    return b"SecurityTokenExpired" in body


def _is_no_such_upload_body(body: bytes) -> bool:
    """Whether an OSS error body indicates the upload session is gone."""
    return b"NoSuchUpload" in body


def _file_identity(file: BinaryIO) -> tuple[str, int, int] | None:
    """Return ``(abspath, size, mtime_ns)`` for a real local file, else None.

    Identity is taken from the file descriptor so that other file-like
    objects (BytesIO, RemoteFile, ...) whose ``name`` happens to match a
    local file can never pick up that file's resume state.
    """
    name = getattr(file, "name", None)
    if not isinstance(name, str) or not name:
        return None
    try:
        st = os.fstat(file.fileno())
    except (AttributeError, OSError, ValueError):
        return None
    return os.path.abspath(name), st.st_size, st.st_mtime_ns


class UploadClient:
    """Handles all upload-related operations.

    Provides two public upload methods:
      - ``simple_upload``: single-request upload via the sample-upload endpoint.
      - ``multipart_upload``: OSS multipart upload for large files, with
        automatic STS token refresh and resume support for interrupted
        uploads.
    """

    def __init__(self, api: APIClient):
        self._api = api
        self._client = httpx.Client(
            headers={
                "User-Agent": api.headers.get("User-Agent", DEFAULT_USER_AGENT),
                "Referer": "https://115.com/",
            }
        )

    def instant_upload(
        self,
        *,
        file: BinaryIO,
        filename: str,
        file_size: int,
        sha1: str,
        dir_id: str,
    ) -> None:
        """Attempt an instant upload (zero-copy if the file is already on 115).

        Raises:
            InstantUploadNotAvailableError: If the server does not have the file,
                with ``response_data`` carrying the initupload response so the
                caller can reuse bucket/object/callback for a multipart upload.
        """

        def read_range(range_str: str) -> bytes:
            start, end = [int(x) for x in range_str.split("-")]
            file.seek(start)
            return file.read(end - start + 1)

        upload_info = self._api.get(Endpoint.PROAPI + "/app/uploadinfo").json()

        payload = {
            "filename": filename,
            "fileid": sha1.upper(),
            "filesize": file_size,
            "target": f"U_1_{dir_id}",
            "appid": 0,
            "sign_key": "",
            "sign_val": "",
            "topupload": "true",
            "appversion": APP_VERSION,
            "userid": upload_info["user_id"],
            "userkey": upload_info["userkey"],
        }
        resp = self._post_upload_init(payload)
        status = int(resp["status"])
        if status == 7:
            sign_check = str(resp.get("sign_check", ""))
            payload["sign_key"] = str(resp.get("sign_key", ""))
            payload["sign_val"] = (
                hashlib.sha1(read_range(sign_check)).hexdigest().upper()
            )
            resp = self._post_upload_init(payload)
            status = int(resp["status"])

        if status != 2:
            raise InstantUploadNotAvailableError(
                "instant upload is not available (file not found on server)",
                response_data=resp,
            )

    def simple_upload(self, file: BinaryIO, *, pid: str, filename: str) -> dict:
        """Upload a file using the simple (sample) upload endpoint.

        Args:
            file: Readable binary file object (seeked to start).
            pid: Parent directory ID on 115.
            filename: Destination filename.

        Returns:
            Parsed JSON response from the upload callback.
        """
        sample_info = self._api.post(
            Endpoint.UPLB + "/3.0/sampleinitupload.php",
            data={"filename": filename, "target": f"U_1_{pid}"},
        ).json()

        fields = {
            "name": filename,
            "key": sample_info["object"],
            "policy": sample_info["policy"],
            "OSSAccessKeyId": sample_info["accessid"],
            "success_action_status": "200",
            "callback": sample_info["callback"],
            "signature": sample_info["signature"],
        }

        request = self._create_multipart_request(
            sample_info["host"],
            data=fields,
            filename=filename,
            file=file,
        )

        with httpcore.ConnectionPool() as pool:
            response = pool.handle_request(request)
            try:
                body = response.read()
            finally:
                response.close()

        if response.status >= 400:
            raise OSError(
                f"sample upload request failed: status={response.status}, body={body!r}"
            )

        data = json.loads(body)
        data["oss_info"] = sample_info
        return data

    def multipart_upload(
        self,
        file: BinaryIO,
        *,
        bucket: str,
        object: str,
        callback: dict,
        file_size: int | None = None,
        on_resume: Callable[[int], None] | None = None,
        remote_path: str | None = None,
        dir_id: str | None = None,
    ) -> dict:
        """Upload a file using the OSS multipart protocol.

        Interrupted uploads can be resumed: while the upload runs, the OSS
        session info and the list of uploaded parts are persisted to a state
        file, and a later call for the same (unmodified) local file continues
        from where the previous attempt stopped.  The STS token is refreshed
        automatically whenever it is about to expire, and a part rejected
        with ``SecurityTokenExpired`` is re-sent with a fresh token.

        Args:
            file: Readable, seekable binary file object (seeked to start).
            bucket: OSS bucket name.
            object: OSS object key.
            callback: Callback payload from the initupload response, with
                ``callback`` and ``callback_var`` fields.
            file_size: Expected size of the file when known; used to
                validate the resume state.
            on_resume: Optional callback invoked with the number of bytes
                already on the server when an upload is resumed.
            remote_path: Destination path of this upload attempt, recorded
                in the resume state so a later resume can detect that the
                target changed.
            dir_id: Destination directory ID of this upload attempt,
                recorded alongside ``remote_path``.

        Returns:
            Parsed JSON response from the OSS complete-upload callback.
        """
        state = self._load_upload_state(file, file_size=file_size)
        if state:
            bucket = state["bucket"]
            object = state["object"]
            callback = state["callback"]
        url = f"https://{bucket}.oss-cn-shenzhen.aliyuncs.com/{object}"
        token = self._get_oss_token()

        upload_id, parts, start_part = self._start_multipart(
            file,
            url,
            bucket,
            object,
            callback,
            state,
            token,
            on_resume,
            remote_path=remote_path,
            dir_id=dir_id,
        )
        # A resumed session may no longer exist on the server (e.g. it was
        # cleaned up while the upload was paused); restart once with a fresh
        # session in that case.
        recovered = False
        while True:
            try:
                resp = self._upload_parts(
                    file, url, upload_id, parts, start_part, token, callback
                )
                break
            except OSSSessionGoneError:
                if recovered:
                    raise
                recovered = True
                logger.warning(
                    "upload session %s no longer exists, starting over", upload_id
                )
                upload_id, parts, start_part = self._start_multipart(
                    file,
                    url,
                    bucket,
                    object,
                    callback,
                    None,
                    token,
                    None,
                    remote_path=remote_path,
                    dir_id=dir_id,
                )

        self._clear_upload_state(file)
        return resp

    def _start_multipart(
        self,
        file: BinaryIO,
        url: str,
        bucket: str,
        object: str,
        callback: dict,
        state: dict | None,
        token: dict,
        on_resume: Callable[[int], None] | None,
        *,
        remote_path: str | None = None,
        dir_id: str | None = None,
    ) -> tuple[str, list[dict], int]:
        """Seek the file to the first unuploaded byte and prepare the session.

        Returns ``(upload_id, parts, start_part)``.  When ``state`` carries
        previously-uploaded parts, the session is resumed: the file is
        rewound to just past the uploaded data and the next part number is
        returned.  Otherwise a fresh OSS session is initiated.
        """
        parts: list[dict] = []
        if state:
            upload_id = state["upload_id"]
            stored = state.get("parts")
            if not isinstance(stored, list):
                stored = []
            # tolerate corrupted entries rather than failing the upload
            stored = [
                part
                for part in stored
                if isinstance(part, dict)
                and isinstance(part.get("PartNumber"), int)
                and isinstance(part.get("Size"), int)
                and "ETag" in part
            ]
            stored.sort(key=lambda p: p["PartNumber"])
            # only the contiguous prefix 1..N is usable (sequential upload)
            usable = 0
            for expected, part in enumerate(stored, start=1):
                if part["PartNumber"] != expected:
                    break
                usable = expected
            parts = stored[:usable]
            start_part = usable + 1
            skipped = sum(part["Size"] for part in parts)
            file.seek(skipped)
            if skipped and on_resume:
                on_resume(skipped)
            if usable:
                logger.info(
                    "resuming upload %s at part %d (%d bytes already uploaded)",
                    upload_id,
                    start_part,
                    skipped,
                )
        else:
            start_part = 1
            file.seek(0)
            upload_id = self._oss_multipart_upload_init(url, token)
            self._save_upload_state(
                file,
                bucket=bucket,
                object=object,
                upload_id=upload_id,
                callback=callback,
                parts=[],
                remote_path=remote_path,
                dir_id=dir_id,
            )
        return upload_id, parts, start_part

    def _upload_parts(
        self,
        file: BinaryIO,
        url: str,
        upload_id: str,
        parts: list[dict],
        start_part: int,
        token: dict,
        callback: dict,
    ) -> dict:
        """Upload remaining parts, then finalize the upload via the callback.

        Appends each uploaded part to ``parts`` and records it in the resume
        state file, so an interruption keeps every finished part usable.
        """
        part_number = start_part
        with httpcore.ConnectionPool() as pool:
            while True:
                token = self._fresh_token(token)
                buf = self._read_part(file)
                if not buf:
                    break

                local_md5 = base64.b64encode(hashlib.md5(buf).digest()).decode()
                retries = 0
                while True:
                    try:
                        part = self._oss_upload_part(
                            url, upload_id, part_number, buf, token, pool
                        )
                        break
                    except OSSTokenExpiredError:
                        retries += 1
                        if retries >= MAX_TOKEN_RETRIES:
                            raise
                        logger.info(
                            "token expired on part %d, refreshing and retrying",
                            part_number,
                        )
                        token = self._get_oss_token()
                part["Size"] = len(buf)

                server_md5 = part.pop("ContentMD5", "")
                if server_md5 and server_md5 != local_md5:
                    raise RuntimeError(
                        f"part {part_number} MD5 mismatch: "
                        f"expected {local_md5!r}, server returned {server_md5!r}"
                    )

                parts.append(part)
                self._update_saved_parts(file, parts)
                part_number += 1

        token = self._fresh_token(token)
        return self._oss_multipart_upload_complete(
            url, upload_id, callback, parts, token
        )

    def _post_upload_init(self, payload: dict) -> dict:
        resp = self._api.post(
            Endpoint.UPLB + "/4.0/initupload.php",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": APP_USER_AGENT,
            },
            **make_upload_payload(payload.copy()),
        )
        return json.loads(ecdh_aes_decrypt(resp.content, decompress=True))

    def _get_oss_token(self) -> dict:
        """Fetch a short-lived STS token from 115 for signing OSS requests."""
        return self._api.get(Endpoint.UPLB + "/3.0/gettoken.php").json()

    def _fresh_token(self, token: dict) -> dict:
        """Return ``token``, or a newly fetched one if it is about to expire."""
        if _token_expires_soon(token):
            logger.debug("OSS token is about to expire, refreshing")
            return self._get_oss_token()
        return token

    @staticmethod
    def _read_part(file: BinaryIO) -> bytes:
        """Read at most one part from the file, buffering it in memory.

        Buffering (instead of streaming) keeps retries simple: a part
        rejected by the server can be re-sent without rewinding the file,
        which also avoids double counting progress for the retried part.
        """
        buf = bytearray()
        while len(buf) < MULTIPART_UPLOAD_PART_SIZE:
            chunk = file.read(min(_CHUNK_SIZE, MULTIPART_UPLOAD_PART_SIZE - len(buf)))
            if not chunk:
                break
            buf += chunk
        return bytes(buf)

    def _oss_multipart_upload_init(self, url: str, token: dict) -> str:
        """Initiate an OSS multipart upload session and return the upload_id."""
        init_url = f"{url}?sequential=1&uploads=1"
        headers = self._oss_sign(init_url, "POST", token)
        resp = self._client.post(init_url, headers=headers)
        resp.raise_for_status()
        upload_id = ET.fromstring(resp.content).findtext("UploadId")
        if not upload_id:
            raise RuntimeError(f"UploadId not found in OSS response: {resp.content!r}")
        return upload_id

    def _oss_upload_part(
        self,
        url: str,
        upload_id: str,
        part_number: int,
        content: bytes,
        token: dict,
        pool: httpcore.ConnectionPool,
    ) -> dict:
        """Upload one part and return its part metadata dict."""
        part_url = f"{url}?partNumber={part_number}&uploadId={upload_id}"
        headers = self._oss_sign(part_url, "PUT", token)
        # handle_request() bypasses httpcore's auto Content-Length (which only
        # runs in the request()/stream() convenience wrappers), and h11 treats
        # a body without framing headers as zero-length, so declare it here.
        headers["content-length"] = str(len(content))
        request = httpcore.Request("PUT", part_url, headers=headers, content=content)
        response = pool.handle_request(request)
        try:
            body = response.read()
        finally:
            response.close()
        if response.status >= 400:
            message = (
                f"part {part_number} upload failed: "
                f"status={response.status}, body={body!r}"
            )
            if _is_token_expired_body(body):
                raise OSSTokenExpiredError(message)
            if _is_no_such_upload_body(body):
                raise OSSSessionGoneError(message)
            raise OSError(message)
        headers_lower = {k.lower(): v for k, v in response.headers}
        etag = headers_lower.get(b"etag", b"").decode().strip('"')
        server_md5 = headers_lower.get(b"content-md5", b"").decode()
        return {"PartNumber": part_number, "ETag": etag, "ContentMD5": server_md5}

    def _oss_multipart_upload_complete(
        self,
        url: str,
        upload_id: str,
        callback: dict,
        parts: list[dict],
        token: dict,
    ) -> dict:
        """Finalize the multipart upload and trigger the 115 server callback."""
        complete_url = f"{url}?uploadId={upload_id}"
        xml_body = (
            b"<CompleteMultipartUpload>"
            + b"".join(
                (
                    f"<Part><PartNumber>{p['PartNumber']}</PartNumber>"
                    f"<ETag>{p['ETag']}</ETag></Part>"
                ).encode()
                for p in parts
            )
            + b"</CompleteMultipartUpload>"
        )
        extra_headers = {
            "x-oss-callback": base64.b64encode(callback["callback"].encode()).decode(),
            "x-oss-callback-var": base64.b64encode(
                callback["callback_var"].encode()
            ).decode(),
            "content-type": "text/xml",
        }
        headers = self._oss_sign(complete_url, "POST", token, extra_headers)
        resp = self._client.post(complete_url, headers=headers, content=xml_body)
        resp.raise_for_status()
        return json.loads(resp.content)

    # -- resume state persistence --

    def resume_state(self, file: BinaryIO) -> dict | None:
        """Return the pending resume state for a local file, if any.

        Cheap to call (stat only): does not read or hash the file.
        """
        return self._load_upload_state(file)

    def _state_path(self, file: BinaryIO) -> str | None:
        """Return the resume-state file path for a local file, else None."""
        ident = _file_identity(file)
        if ident is None:
            return None
        digest = hashlib.sha1(ident[0].encode()).hexdigest()
        return str(_upload_state_dir() / f"{digest}.json")

    def _save_upload_state(
        self,
        file: BinaryIO,
        *,
        bucket: str,
        object: str,
        upload_id: str,
        callback: dict,
        parts: list[dict] | None = None,
        remote_path: str | None = None,
        dir_id: str | None = None,
    ) -> None:
        """Persist session info so an interrupted upload can be resumed.

        ``remote_path`` and ``dir_id`` record the target of the upload
        attempt the session belongs to.  A resumed upload always completes
        into that original target (the saved callback is bound to it), so
        the caller can detect a target change and relocate the file after
        completion.
        """
        path = self._state_path(file)
        if path is None:
            return
        abspath, size, mtime_ns = _file_identity(file)
        try:
            state = {
                "version": 1,
                "path": abspath,
                "size": size,
                "mtime_ns": mtime_ns,
                "part_size": MULTIPART_UPLOAD_PART_SIZE,
                "bucket": bucket,
                "object": object,
                "upload_id": upload_id,
                "callback": callback,
                "parts": parts or [],
                "remote_path": remote_path,
                "dir_id": dir_id,
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = f"{path}.{os.getpid()}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            os.replace(tmp_path, path)
        except (OSError, TypeError):
            # Persistence is best-effort; without it uploads still work,
            # they just cannot be resumed.
            logger.debug("failed to save upload state", exc_info=True)

    def _update_saved_parts(self, file: BinaryIO, parts: list[dict]) -> None:
        """Record the parts uploaded so far in the resume state file."""
        path = self._state_path(file)
        if path is None:
            return
        try:
            with open(path) as f:
                state = json.load(f)
            state["parts"] = parts
            tmp_path = f"{path}.{os.getpid()}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            os.replace(tmp_path, path)
        except (OSError, ValueError):
            logger.debug("failed to update upload state", exc_info=True)

    def _load_upload_state(
        self, file: BinaryIO, *, file_size: int | None = None
    ) -> dict | None:
        """Return a still-valid resume state for the file, else None.

        The state is only reused when the local file is unchanged since the
        state was written (same path, size and mtime).
        """
        path = self._state_path(file)
        if path is None:
            return None
        try:
            with open(path) as f:
                state = json.load(f)
        except (OSError, ValueError):
            return None
        abspath, size, mtime_ns = _file_identity(file)
        if state.get("version") != 1:
            return None
        if (
            state.get("path") != abspath
            or state.get("size") != size
            or state.get("mtime_ns") != mtime_ns
            or state.get("part_size") != MULTIPART_UPLOAD_PART_SIZE
            or (file_size is not None and file_size != size)
        ):
            return None
        if not all(
            state.get(key) for key in ("bucket", "object", "upload_id", "callback")
        ):
            return None
        return state

    def _clear_upload_state(self, file: BinaryIO) -> None:
        """Remove the resume state after a successful upload."""
        path = self._state_path(file)
        if path is None:
            return
        try:
            os.unlink(path)
        except OSError:
            pass

    # -- request signing --

    def _oss_sign(
        self,
        url: str,
        method: str,
        token: dict,
        extra_headers: dict | None = None,
    ) -> dict:
        """Build an OSS v1 (HMAC-SHA1) signed headers dict for the given request."""
        headers: dict[str, str] = {}
        if extra_headers:
            headers.update(extra_headers)
        headers["x-oss-security-token"] = token["SecurityToken"]
        headers.setdefault("content-md5", "")
        headers.setdefault("content-type", "")
        date = headers["date"] = formatdate(usegmt=True)

        urlp = urlsplit(url)
        bucket = urlp.hostname.partition(".")[0]
        headers["host"] = urlp.netloc
        oss_header_lines = "\n".join(
            f"{k}:{v}" for k, v in sorted(headers.items()) if k.startswith("x-oss-")
        )
        canonical_resource = (
            f"/{bucket}{urlunsplit(urlp._replace(scheme='', netloc=''))}"
        )
        string_to_sign = (
            f"{method.upper()}\n"
            f"{headers['content-md5']}\n"
            f"{headers['content-type']}\n"
            f"{date}\n"
            f"{oss_header_lines}\n"
            f"{canonical_resource}"
        )
        signature = base64.b64encode(
            hmac.digest(
                token["AccessKeySecret"].encode(),
                string_to_sign.encode(),
                "sha1",
            )
        ).decode()
        headers["authorization"] = f"OSS {token['AccessKeyId']}:{signature}"
        return headers

    def _create_multipart_request(
        self,
        url: str,
        *,
        data: dict[str, str],
        filename: str,
        file: BinaryIO,
    ) -> httpcore.Request:
        parsed = urlparse(url)
        boundary = uuid.uuid4().hex
        content = self._iter_multipart_content(
            boundary=boundary,
            data=data,
            filename=filename,
            file=file,
        )
        return httpcore.Request(
            method="POST",
            url=url,
            headers={
                "Host": parsed.netloc,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Transfer-Encoding": "chunked",
            },
            content=content,
        )

    def _iter_multipart_content(
        self,
        *,
        boundary: str,
        data: dict[str, str],
        filename: str,
        file: BinaryIO,
    ):
        boundary_bytes = boundary.encode("ascii")

        for key, value in data.items():
            yield b"--" + boundary_bytes + b"\r\n"
            yield (f'Content-Disposition: form-data; name="{key}"\r\n\r\n').encode(
                "utf-8"
            )
            yield str(value).encode("utf-8")
            yield b"\r\n"

        yield b"--" + boundary_bytes + b"\r\n"
        yield (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        ).encode("utf-8")
        yield b"Content-Type: application/octet-stream\r\n\r\n"

        while chunk := file.read(_CHUNK_SIZE):
            yield chunk

        yield b"\r\n"
        yield b"--" + boundary_bytes + b"--\r\n"
