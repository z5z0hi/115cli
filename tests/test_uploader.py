from unittest.mock import MagicMock

import pytest

from cli115.exceptions import UploadVerificationError
from cli115.uploader import Uploader
from tests.client.conftest import make_dir, make_file


def _make_client():
    mock = MagicMock()
    mock.file.upload.return_value = make_file()
    return mock


class TestUploadFile:
    def test_upload_to_nonexistent_path(self):
        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        uploaded = make_file(name="file.txt")
        client.file.upload.return_value = uploaded

        uploader = Uploader(client)
        uploader.upload("/local/file.txt", "/remote/file.txt")

        assert client.file.upload.call_count == 1
        call_args = client.file.upload.call_args
        assert call_args.args[0] == "/remote/file.txt"
        assert call_args.kwargs["instant_only"] is None
        assert len(uploader.entries) == 1
        assert uploader.entries[0].remote_path == "/remote/file.txt"

    def test_upload_to_existing_directory_appends_filename(self):
        client = _make_client()
        client.file.stat.return_value = make_dir(name="remotedir")
        uploaded = make_file(name="file.txt")
        client.file.upload.return_value = uploaded

        uploader = Uploader(client)
        uploader.upload("/local/path/file.txt", "/remote/dir")

        assert client.file.upload.call_count == 1
        call_args = client.file.upload.call_args
        assert call_args.args[0] == "/remote/dir/file.txt"
        assert call_args.kwargs["instant_only"] is None
        assert len(uploader.entries) == 1

    def test_upload_instant_only_threshold_passed_through(self):
        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        threshold = 100 * 1024 * 1024  # 100 MB

        uploader = Uploader(client)
        uploader.upload("/local/file.txt", "/remote/file.txt", instant_only=threshold)

        assert client.file.upload.call_count == 1
        call_args = client.file.upload.call_args
        assert call_args.kwargs["instant_only"] == threshold

    def test_upload_file_retried_once_on_failure(self):
        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.upload.side_effect = [RuntimeError("boom"), make_file()]

        uploader = Uploader(client)
        result = uploader.upload("/local/file.txt", "/remote/file.txt")

        assert client.file.upload.call_count == 2
        assert uploader.entries[0].error is None
        assert result is not None

    def test_retry_after_verification_error_treats_existing_as_success(self):
        client = _make_client()
        # 1st stat: pre-upload duplicate check passes;
        # 2nd stat: the retry's duplicate check finds the file
        client.file.stat.side_effect = [
            FileNotFoundError("not found"),
            make_file(name="file.txt"),
        ]
        # 1st attempt completes but the file cannot be verified;
        # on retry the duplicate check raises FileExistsError
        client.file.upload.side_effect = [
            UploadVerificationError("did not appear"),
            FileExistsError("remote path already exists"),
        ]

        uploader = Uploader(client)
        result = uploader.upload("/local/file.txt", "/remote/file.txt")

        assert client.file.upload.call_count == 2
        assert uploader.entries[0].error is None
        assert result is not None


class TestUploadDirectory:
    def test_upload_dir_to_nonexistent_remote_creates_it(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        dest_dir = make_dir(name=tmp_path.name)
        client.file.create_directory.return_value = dest_dir

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/newdir")

        assert isinstance(uploader, Uploader)
        client.file.create_directory.assert_any_call("/remote/newdir", parents=True)

    def test_upload_dir_to_existing_remote_dir_creates_subdir(self, tmp_path):
        (tmp_path / "file.txt").write_text("content")

        client = _make_client()
        client.file.stat.return_value = make_dir(name="existing")
        dest_dir = make_dir(name=tmp_path.name)
        client.file.create_directory.return_value = dest_dir

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/existing")

        assert isinstance(uploader, Uploader)
        expected_dest = "/remote/existing/" + tmp_path.name
        client.file.create_directory.assert_any_call(expected_dest, parents=False)

    def test_upload_dir_to_remote_file_raises(self, tmp_path):
        (tmp_path / "file.txt").write_text("content")

        client = _make_client()
        client.file.stat.return_value = make_file(name="remote.txt")

        uploader = Uploader(client)
        with pytest.raises(FileExistsError, match="cannot upload directory"):
            uploader.upload(str(tmp_path), "/remote/file.txt")

        client.file.upload.assert_not_called()

    def test_upload_dir_all_files_are_uploaded(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.txt").write_text("c")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.create_directory.return_value = make_dir()

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/dest")

        assert client.file.upload.call_count == 3
        uploaded_names = {
            c.args[0].rsplit("/", 1)[-1] for c in client.file.upload.call_args_list
        }
        assert uploaded_names == {"a.txt", "b.txt", "c.txt"}
        assert len(uploader.entries) == 3

    def test_upload_dir_continues_and_retries_failed_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.create_directory.return_value = make_dir()
        # a.txt fails once and succeeds on its retry; b.txt succeeds directly
        client.file.upload.side_effect = [
            RuntimeError("network error"),
            make_file(),
            make_file(),
        ]

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/dest")

        assert client.file.upload.call_count == 3
        assert len(uploader.entries) == 2
        failed_entries = [
            entry for entry in uploader.entries if entry.error is not None
        ]
        assert failed_entries == []

    def test_upload_dir_records_permanently_failed_file(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.create_directory.return_value = make_dir()
        # a.txt fails twice (initial + retry); b.txt succeeds
        client.file.upload.side_effect = [
            RuntimeError("network error"),
            RuntimeError("retry failed"),
            make_file(),
        ]

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/dest")

        assert client.file.upload.call_count == 3
        failed_entries = [
            entry for entry in uploader.entries if entry.error is not None
        ]
        assert len(failed_entries) == 1

        failed_entry = failed_entries[0]
        assert failed_entry.remote_path.endswith("/a.txt")
        assert str(failed_entry.error) == "retry failed"

    def test_upload_dir_passes_dest_dir_id_to_uploads(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.create_directory.return_value = make_dir(id="100")

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/dest")

        assert client.file.upload.call_count == 1
        assert client.file.upload.call_args.kwargs["dir_id"] == "100"

    def test_upload_dir_nested_dirs_created_with_parent_id(self, tmp_path):
        # Structure:
        #   tmp/
        #     sub1/
        #       sub2/
        #         deep.txt
        sub1 = tmp_path / "sub1"
        sub1.mkdir()
        sub2 = sub1 / "sub2"
        sub2.mkdir()
        (sub2 / "deep.txt").write_text("deep")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        dest_dir = make_dir(id="100")
        sub1_dir = make_dir(id="201")
        sub2_dir = make_dir(id="202")
        client.file.create_directory.side_effect = [dest_dir, sub1_dir, sub2_dir]

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/dest")

        calls = client.file.create_directory.call_args_list
        assert calls[0].args[0] == "/remote/dest"
        assert calls[1].args[0] == "/remote/dest/sub1"
        assert calls[1].kwargs["pid"] == "100"
        assert calls[2].args[0] == "/remote/dest/sub1/sub2"
        assert calls[2].kwargs["pid"] == "201"

    def test_upload_dir_multilevel_subdirs_created(self, tmp_path):
        # Structure:
        #   tmp/
        #     root.txt
        #     sub1/
        #       mid.txt
        #       sub2/
        #         deep.txt
        (tmp_path / "root.txt").write_text("root")
        sub1 = tmp_path / "sub1"
        sub1.mkdir()
        (sub1 / "mid.txt").write_text("mid")
        sub2 = sub1 / "sub2"
        sub2.mkdir()
        (sub2 / "deep.txt").write_text("deep")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.create_directory.return_value = make_dir()

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/dest")

        # Base dir + sub1 + sub1/sub2
        assert client.file.create_directory.call_count == 3
        create_paths = [c.args[0] for c in client.file.create_directory.call_args_list]
        assert "/remote/dest" in create_paths
        assert any("sub1" in p for p in create_paths)
        assert any("sub2" in p for p in create_paths)

        # All 3 files uploaded
        assert client.file.upload.call_count == 3

    def test_upload_dir_instant_only_passed_to_file_uploads(self, tmp_path):
        (tmp_path / "file.txt").write_text("content")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.create_directory.return_value = make_dir()
        threshold = 50 * 1024 * 1024  # 50 MB

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/dest", instant_only=threshold)

        client.file.upload.assert_called_once()
        assert client.file.upload.call_args.kwargs["instant_only"] == threshold


class TestUploadDirectoryPatterns:
    def test_exclude_pattern_filters_files(self, tmp_path):
        (tmp_path / "app.py").write_text("code")
        (tmp_path / "debug.log").write_text("log")
        (tmp_path / "error.log").write_text("log")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.create_directory.return_value = make_dir()

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/dest", exclude=["**/*.log"])

        client.file.create_directory.assert_called_once_with(
            "/remote/dest", parents=True
        )
        assert client.file.upload.call_count == 1
        uploaded_name = client.file.upload.call_args.args[0].rsplit("/", 1)[-1]
        assert uploaded_name == "app.py"

    def test_include_pattern_filters_files(self, tmp_path):
        (tmp_path / "main.py").write_text("code")
        (tmp_path / "utils.py").write_text("code")
        (tmp_path / "README.md").write_text("docs")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.create_directory.return_value = make_dir()

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/dest", include=["**/*.py"])

        client.file.create_directory.assert_called_once_with(
            "/remote/dest", parents=True
        )
        assert client.file.upload.call_count == 2
        uploaded_names = {
            c.args[0].rsplit("/", 1)[-1] for c in client.file.upload.call_args_list
        }
        assert uploaded_names == {"main.py", "utils.py"}

    def test_exclude_subdirectory_pattern(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("code")
        temp = tmp_path / "temp"
        temp.mkdir()
        (temp / "cache.bin").write_text("cache")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.create_directory.return_value = make_dir()

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/dest", exclude=["temp/**"])

        created_dirs = [c.args[0] for c in client.file.create_directory.call_args_list]
        assert "/remote/dest" in created_dirs
        assert "/remote/dest/src" in created_dirs
        assert not any("temp" in p for p in created_dirs)
        assert client.file.upload.call_count == 1
        uploaded_name = client.file.upload.call_args.args[0].rsplit("/", 1)[-1]
        assert uploaded_name == "main.py"

    def test_include_and_exclude_combined(self, tmp_path):
        (tmp_path / "keep.py").write_text("code")
        (tmp_path / "skip_test.py").write_text("test code")
        (tmp_path / "data.csv").write_text("data")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.create_directory.return_value = make_dir()

        uploader = Uploader(client)
        uploader.upload(
            str(tmp_path),
            "/remote/dest",
            include=["**/*.py"],
            exclude=["**/skip_*"],
        )

        client.file.create_directory.assert_called_once_with(
            "/remote/dest", parents=True
        )
        assert client.file.upload.call_count == 1
        uploaded_name = client.file.upload.call_args.args[0].rsplit("/", 1)[-1]
        assert uploaded_name == "keep.py"

    def test_no_patterns_uploads_all(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.log").write_text("b")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")
        client.file.create_directory.return_value = make_dir()

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/dest")

        client.file.create_directory.assert_called_once_with(
            "/remote/dest", parents=True
        )
        assert client.file.upload.call_count == 2


class TestNoTargetDirectory:
    def test_dir_upload_no_target_dir_uses_remote_as_dest(self, tmp_path):
        (tmp_path / "file.txt").write_text("content")

        client = _make_client()
        client.file.stat.return_value = make_dir(name="existing")
        client.file.create_directory.return_value = make_dir()

        uploader = Uploader(client)
        uploader.upload(str(tmp_path), "/remote/existing", no_target_dir=True)

        client.file.create_directory.assert_any_call("/remote/existing", parents=False)
        assert client.file.upload.call_args.args[0] == "/remote/existing/file.txt"


class TestDryRun:
    def test_dry_run_no_upload_called(self):
        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")

        uploader = Uploader(client, dry_run=True)
        uploader.upload("/local/file.txt", "/remote/file.txt")

        client.file.upload.assert_not_called()
        assert len(uploader.entries) == 1

    def test_dry_run_directory_no_upload_called(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        client = _make_client()
        client.file.stat.side_effect = FileNotFoundError("not found")

        uploader = Uploader(client, dry_run=True)
        uploader.upload(str(tmp_path), "/remote/dest")

        client.file.upload.assert_not_called()
        client.file.create_directory.assert_not_called()
        assert len(uploader.entries) == 2
