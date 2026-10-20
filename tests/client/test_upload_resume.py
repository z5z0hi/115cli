"""Unit tests for multipart upload token refresh and resume logic."""

import hashlib
import io
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import cli115.client.general.upload as upload_module
from cli115.client.general.upload import (
    MAX_TOKEN_RETRIES,
    OSSSessionGoneError,
    OSSTokenExpiredError,
    UploadClient,
    _is_no_such_upload_body,
    _is_token_expired_body,
    _token_expires_soon,
)


PART_SIZE = 1024


def make_token(expiration: str | None = None) -> dict:
    token = {
        "AccessKeyId": "ak",
        "AccessKeySecret": "sk",
        "SecurityToken": "st",
    }
    if expiration:
        token["Expiration"] = expiration
    return token


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeResponse:
    def __init__(self, status: int, body: bytes, headers=()):
        self.status = status
        self._body = body
        self.headers = list(headers)

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        pass


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLI115_UPLOAD_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


@pytest.fixture
def client(state_dir):
    api = MagicMock()
    api.headers = {}
    with patch("cli115.client.general.upload.httpx.Client"):
        return UploadClient(api)


@pytest.fixture(autouse=True)
def small_part_size(monkeypatch):
    monkeypatch.setattr(upload_module, "MULTIPART_UPLOAD_PART_SIZE", PART_SIZE)


@pytest.fixture
def local_file(tmp_path):
    content = os.urandom(PART_SIZE * 2 + 452)
    path = tmp_path / "big.bin"
    path.write_bytes(content)
    return path, content


def write_state(path, *, content, upload_id="UID", parts=(), **overrides):
    st = os.stat(path)
    state = {
        "version": 1,
        "path": os.path.abspath(path),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "part_size": PART_SIZE,
        "bucket": "bucket",
        "object": "obj",
        "upload_id": upload_id,
        "callback": {"callback": "cb", "callback_var": "cbv"},
        "parts": list(parts),
    }
    state.update(overrides)
    os.makedirs(upload_module._upload_state_dir(), exist_ok=True)
    name = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()
    with open(os.path.join(upload_module._upload_state_dir(), f"{name}.json"), "w") as f:
        json.dump(state, f)
    return state


class TestTokenExpiry:
    def test_no_expiration_field(self):
        assert _token_expires_soon(make_token()) is False

    def test_expired_token(self):
        token = make_token("2020-01-01T00:00:00Z")
        assert _token_expires_soon(token) is True

    def test_future_token(self):
        token = make_token("2999-01-01T00:00:00Z")
        assert _token_expires_soon(token) is False

    def test_within_margin(self):
        soon = datetime.now(timezone.utc) + timedelta(seconds=60)
        assert _token_expires_soon(make_token(iso(soon))) is True

    def test_invalid_format(self):
        assert _token_expires_soon(make_token("not-a-date")) is False

    def test_error_body_detection(self):
        assert _is_token_expired_body(b"<Code>SecurityTokenExpired</Code>") is True
        assert _is_token_expired_body(b"<Code>AccessDenied</Code>") is False
        assert _is_no_such_upload_body(b"<Code>NoSuchUpload</Code>") is True
        assert _is_no_such_upload_body(b"<Code>AccessDenied</Code>") is False


class TestFreshToken:
    def test_refreshes_expiring_token(self, client):
        expired = make_token("2020-01-01T00:00:00Z")
        fresh = make_token("2999-01-01T00:00:00Z")
        client._get_oss_token = MagicMock(return_value=fresh)
        assert client._fresh_token(expired) is fresh
        client._get_oss_token.assert_called_once()

    def test_keeps_valid_token(self, client):
        fresh = make_token("2999-01-01T00:00:00Z")
        client._get_oss_token = MagicMock()
        assert client._fresh_token(fresh) is fresh
        client._get_oss_token.assert_not_called()


class TestUploadPart:
    def test_raises_token_expired_error(self, client):
        pool = MagicMock()
        pool.handle_request.return_value = FakeResponse(
            403, b"<Code>SecurityTokenExpired</Code>"
        )
        with pytest.raises(OSSTokenExpiredError):
            client._oss_upload_part(
                "https://b.o/k", "uid", 1, b"data", make_token(), pool
            )

    def test_raises_generic_oserror(self, client):
        pool = MagicMock()
        pool.handle_request.return_value = FakeResponse(500, b"boom")
        with pytest.raises(OSError, match="part 1 upload failed") as exc_info:
            client._oss_upload_part(
                "https://b.o/k", "uid", 1, b"data", make_token(), pool
            )
        assert not isinstance(exc_info.value, OSSTokenExpiredError)

    def test_returns_etag_and_md5(self, client):
        pool = MagicMock()
        pool.handle_request.return_value = FakeResponse(
            200, b"", headers=[(b"etag", b'"abc"'), (b"content-md5", b"md5")]
        )
        part = client._oss_upload_part(
            "https://b.o/k", "uid", 7, b"data", make_token(), pool
        )
        assert part == {"PartNumber": 7, "ETag": "abc", "ContentMD5": "md5"}

    def test_sends_matching_content_length(self, client):
        # handle_request() bypasses httpcore's auto Content-Length; without an
        # explicit header h11 treats the body as zero-length and rejects it
        mock_backend = pytest.importorskip("httpcore._backends.mock")
        import httpcore

        class RecordingStream(mock_backend.MockStream):
            def __init__(self, buffer):
                super().__init__(buffer)
                self.written = b""

            def write(self, buffer, timeout=None):
                self.written += buffer

        class RecordingBackend(mock_backend.MockBackend):
            def connect_tcp(self, *args, **kwargs):
                self.stream = RecordingStream(list(self._buffer))
                return self.stream

        response = b'HTTP/1.1 200 OK\r\nETag: "abc"\r\nContent-Length: 0\r\n\r\n'
        backend = RecordingBackend([response])
        pool = httpcore.ConnectionPool(network_backend=backend)

        buf = bytes(range(256)) * 4096
        part = client._oss_upload_part(
            "http://oss.local/obj", "uid", 1, buf, make_token(), pool
        )

        head, _, body = backend.stream.written.partition(b"\r\n\r\n")
        headers = {}
        for line in head.split(b"\r\n")[1:]:
            k, _, v = line.partition(b":")
            headers[k.strip().lower()] = v.strip()
        assert headers[b"content-length"] == str(len(buf)).encode()
        assert body == buf
        assert part["ETag"] == "abc"


class TestTokenRetry:
    def test_retries_part_with_new_token(self, client, local_file):
        path, content = local_file
        tokens = [make_token(), make_token(), make_token()]
        client._get_oss_token = MagicMock(side_effect=tokens)
        client._oss_multipart_upload_init = MagicMock(return_value="UID")
        client._oss_multipart_upload_complete = MagicMock(return_value={"state": True})

        calls = []

        def fake_part(url, upload_id, part_number, buf, token, pool):
            calls.append((part_number, buf, token))
            if part_number == 1 and len(calls) == 1:
                raise OSSTokenExpiredError("expired")
            return {"PartNumber": part_number, "ETag": f"e{part_number}"}

        client._oss_upload_part = MagicMock(side_effect=fake_part)

        file = open(path, "rb")
        try:
            resp = client.multipart_upload(
                file,
                bucket="bucket",
                object="obj",
                callback={"callback": "cb", "callback_var": "cbv"},
            )
        finally:
            file.close()

        assert resp == {"state": True}
        # first call failed on the expired token, second used the new one
        assert calls[0][0] == 1 and calls[0][2] is tokens[0]
        assert calls[1][0] == 1 and calls[1][2] is tokens[1]
        assert calls[1][1] == content[:PART_SIZE]
        # remaining parts succeed with the refreshed token
        assert [c[0] for c in calls[1:]] == [1, 2, 3]
        assert calls[-1][1] == content[2 * PART_SIZE :]
        # 1 initial fetch + 1 refresh; no proactive refresh (no Expiration)
        assert client._get_oss_token.call_count == 2
        client._oss_multipart_upload_complete.assert_called_once()

    def test_gives_up_after_max_retries(self, client, local_file):
        path, _ = local_file
        client._get_oss_token = MagicMock(return_value=make_token())
        client._oss_multipart_upload_init = MagicMock(return_value="UID")
        client._oss_upload_part = MagicMock(
            side_effect=OSSTokenExpiredError("expired")
        )
        with open(path, "rb") as file:
            with pytest.raises(OSSTokenExpiredError):
                client.multipart_upload(
                    file,
                    bucket="bucket",
                    object="obj",
                    callback={"callback": "cb", "callback_var": "cbv"},
                )
        assert client._oss_upload_part.call_count == MAX_TOKEN_RETRIES


class TestResume:
    def test_resumes_from_saved_parts(self, client, local_file):
        path, content = local_file
        write_state(
            path,
            content=content,
            upload_id="OLD",
            parts=[{"PartNumber": 1, "ETag": "e1", "Size": PART_SIZE}],
        )
        client._get_oss_token = MagicMock(return_value=make_token())
        client._oss_multipart_upload_init = MagicMock()
        client._oss_multipart_upload_complete = MagicMock(return_value={"state": True})

        calls = []

        def fake_part(url, upload_id, part_number, buf, token, pool):
            calls.append((upload_id, part_number, buf))
            return {"PartNumber": part_number, "ETag": f"e{part_number}"}

        client._oss_upload_part = MagicMock(side_effect=fake_part)

        resumed = []
        with open(path, "rb") as file:
            state_path = client._state_path(file)
            client.multipart_upload(
                file,
                bucket="bucket",
                object="obj",
                callback={"callback": "cb", "callback_var": "cbv"},
                on_resume=resumed.append,
            )

        # no new session was created; the saved upload_id is reused
        client._oss_multipart_upload_init.assert_not_called()
        assert resumed == [PART_SIZE]
        assert [c[1] for c in calls] == [2, 3]
        assert calls[0][2] == content[PART_SIZE : 2 * PART_SIZE]
        assert calls[1][2] == content[2 * PART_SIZE :]
        # the complete request carries the saved part and the two new ones
        parts = client._oss_multipart_upload_complete.call_args[0][3]
        assert [p["PartNumber"] for p in parts] == [1, 2, 3]
        assert parts[0]["ETag"] == "e1"
        # state file is removed after success
        assert state_path is not None
        assert not os.path.exists(state_path)

    def test_resume_uses_only_contiguous_prefix(self, client, local_file):
        path, content = local_file
        # a gap at part 2 makes parts 3+ unusable (sequential upload)
        write_state(
            path,
            content=content,
            upload_id="OLD",
            parts=[
                {"PartNumber": 1, "ETag": "e1", "Size": PART_SIZE},
                {"PartNumber": 3, "ETag": "e3", "Size": PART_SIZE},
            ],
        )
        client._get_oss_token = MagicMock(return_value=make_token())
        client._oss_multipart_upload_init = MagicMock()
        client._oss_multipart_upload_complete = MagicMock(return_value={"state": True})
        client._oss_upload_part = MagicMock(
            side_effect=lambda url, uid, n, buf, token, pool: {
                "PartNumber": n, "ETag": f"e{n}",
            }
        )

        resumed = []
        with open(path, "rb") as file:
            client.multipart_upload(
                file,
                bucket="bucket",
                object="obj",
                callback={"callback": "cb", "callback_var": "cbv"},
                on_resume=resumed.append,
            )

        # resume skips only part 1, then re-uploads 2 and 3
        assert resumed == [PART_SIZE]
        assert client._oss_upload_part.call_count == 2
        parts = client._oss_multipart_upload_complete.call_args[0][3]
        assert [p["PartNumber"] for p in parts] == [1, 2, 3]

    def test_restarts_when_session_gone(self, client, local_file):
        path, content = local_file
        write_state(
            path,
            content=content,
            upload_id="OLD",
            parts=[{"PartNumber": 1, "ETag": "e1", "Size": PART_SIZE}],
        )
        client._get_oss_token = MagicMock(return_value=make_token())
        client._oss_multipart_upload_init = MagicMock(return_value="NEW")
        client._oss_multipart_upload_complete = MagicMock(return_value={"state": True})

        calls = []
        state_ids = []

        def fake_part(url, upload_id, part_number, buf, token, pool):
            calls.append((upload_id, part_number))
            with open(client._state_path(file)) as f:
                saved = json.load(f)
            state_ids.append((saved["upload_id"], len(saved["parts"])))
            if upload_id == "OLD":
                raise OSSSessionGoneError("no such upload")
            return {"PartNumber": part_number, "ETag": f"e{part_number}"}

        client._oss_upload_part = MagicMock(side_effect=fake_part)

        with open(path, "rb") as file:
            state_path = client._state_path(file)
            client.multipart_upload(
                file,
                bucket="bucket",
                object="obj",
                callback={"callback": "cb", "callback_var": "cbv"},
            )

        client._oss_multipart_upload_init.assert_called_once()
        # resumed at part 2 on OLD, rejected, then restarted from part 1 on NEW
        assert calls == [("OLD", 2), ("NEW", 1), ("NEW", 2), ("NEW", 3)]
        parts = client._oss_multipart_upload_complete.call_args[0][3]
        assert [p["PartNumber"] for p in parts] == [1, 2, 3]
        # state pointed at the new session with empty parts while re-uploading
        assert state_ids[0] == ("OLD", 1)
        assert state_ids[1] == ("NEW", 0)
        # and is cleared after success
        assert state_path is not None
        assert not os.path.exists(state_path)

    def test_skips_state_when_file_changed(self, client, local_file):
        path, content = local_file
        write_state(path, content=content, mtime_ns=12345)
        client._get_oss_token = MagicMock(return_value=make_token())
        client._oss_multipart_upload_init = MagicMock(return_value="NEW")
        client._oss_multipart_upload_complete = MagicMock(return_value={"state": True})
        client._oss_upload_part = MagicMock(
            side_effect=lambda url, uid, n, buf, token, pool: {
                "PartNumber": n, "ETag": f"e{n}",
            }
        )

        with open(path, "rb") as file:
            client.multipart_upload(
                file,
                bucket="bucket",
                object="obj",
                callback={"callback": "cb", "callback_var": "cbv"},
            )

        client._oss_multipart_upload_init.assert_called_once()
        assert client._oss_upload_part.call_count == 3

    def test_keeps_state_on_failure(self, client, local_file):
        path, content = local_file
        write_state(path, content=content, upload_id="OLD")
        client._get_oss_token = MagicMock(return_value=make_token())
        client._oss_upload_part = MagicMock(side_effect=OSError("network down"))

        with open(path, "rb") as file:
            state_path = client._state_path(file)
            with pytest.raises(OSError, match="network down"):
                client.multipart_upload(
                    file,
                    bucket="bucket",
                    object="obj",
                    callback={"callback": "cb", "callback_var": "cbv"},
                )
        assert state_path is not None
        assert os.path.exists(state_path)

    def test_state_records_uploaded_parts(self, client, local_file):
        path, content = local_file
        client._get_oss_token = MagicMock(return_value=make_token())
        client._oss_multipart_upload_init = MagicMock(return_value="UID")
        client._oss_multipart_upload_complete = MagicMock(return_value={"state": True})

        parts_seen = []

        def fake_part(url, upload_id, part_number, buf, token, pool):
            with open(client._state_path(file)) as f:
                parts_seen.append(json.load(f)["parts"])
            return {"PartNumber": part_number, "ETag": f"e{part_number}"}

        client._oss_upload_part = MagicMock(side_effect=fake_part)

        with open(path, "rb") as file:
            client.multipart_upload(
                file,
                bucket="bucket",
                object="obj",
                callback={"callback": "cb", "callback_var": "cbv"},
            )

        # each part was recorded in the state file right after it uploaded
        assert [len(parts) for parts in parts_seen] == [0, 1, 2]
        assert parts_seen[-1][-1]["PartNumber"] == 2
        assert parts_seen[-1][-1]["ETag"] == "e2"


class TestStatePersistence:
    def test_no_state_for_nameless_file(self, client):
        buf = io.BytesIO(b"data")
        assert client._state_path(buf) is None
        assert client._load_upload_state(buf) is None

    def test_no_state_for_non_local_file(self, client, tmp_path):
        # a file-like object (e.g. RemoteFile) whose name matches a real
        # local file must not pick up that file's resume state
        real = tmp_path / "nonexistent.bin"
        real.write_bytes(b"data")

        class FakeRemote:
            name = str(real)

        client._save_upload_state(
            open(str(real), "rb"),
            bucket="bucket",
            object="obj",
            upload_id="UID",
            callback={"callback": "cb", "callback_var": "cbv"},
        )
        assert client._load_upload_state(FakeRemote()) is None

    def test_state_roundtrip(self, client, local_file):
        path, content = local_file
        with open(path, "rb") as file:
            client._save_upload_state(
                file,
                bucket="bucket",
                object="obj",
                upload_id="UID",
                callback={"callback": "cb", "callback_var": "cbv"},
                parts=[{"PartNumber": 1, "ETag": "e1", "Size": PART_SIZE}],
            )
            state = client._load_upload_state(file, file_size=len(content))
        assert state is not None
        assert state["upload_id"] == "UID"
        assert state["part_size"] == PART_SIZE
        assert state["parts"] == [{"PartNumber": 1, "ETag": "e1", "Size": PART_SIZE}]

    def test_update_saved_parts(self, client, local_file):
        path, content = local_file
        with open(path, "rb") as file:
            client._save_upload_state(
                file,
                bucket="bucket",
                object="obj",
                upload_id="UID",
                callback={"callback": "cb", "callback_var": "cbv"},
                parts=[],
            )
            client._update_saved_parts(
                file, [{"PartNumber": 1, "ETag": "e1", "Size": PART_SIZE}]
            )
            state = client._load_upload_state(file)
        assert state is not None
        assert state["parts"] == [{"PartNumber": 1, "ETag": "e1", "Size": PART_SIZE}]

    def test_state_with_invalid_parts_is_tolerated(self, client, local_file):
        path, content = local_file
        write_state(path, content=content, upload_id="OLD", parts="not-a-list")
        with open(path, "rb") as file:
            state = client._load_upload_state(file)
            assert state is not None
            # resume proceeds with an empty part list on the saved session
            upload_id, parts, start_part = client._start_multipart(
                file,
                "https://bucket.oss-cn-shenzhen.aliyuncs.com/obj",
                state["bucket"],
                state["object"],
                state["callback"],
                state,
                make_token(),
                None,
            )
        assert upload_id == "OLD"
        assert parts == []
        assert start_part == 1

    def test_state_rejected_on_size_mismatch(self, client, local_file):
        path, content = local_file
        with open(path, "rb") as file:
            client._save_upload_state(
                file,
                bucket="bucket",
                object="obj",
                upload_id="UID",
                callback={"callback": "cb", "callback_var": "cbv"},
            )
            assert client._load_upload_state(file, file_size=len(content) + 1) is None

    def test_clear_state(self, client, local_file):
        path, _ = local_file
        with open(path, "rb") as file:
            client._save_upload_state(
                file,
                bucket="bucket",
                object="obj",
                upload_id="UID",
                callback={"callback": "cb", "callback_var": "cbv"},
            )
            assert os.path.exists(client._state_path(file))
            client._clear_upload_state(file)
            assert not os.path.exists(client._state_path(file))
            # clearing again is a no-op
            client._clear_upload_state(file)
