"""Process ownership and private, credential-free launch metadata."""

import os
import stat

import pytest

from wavedl_daemon.serve_state import (
    AlreadyRunning,
    check_not_running,
    read_run_url,
    remove_run_url,
    write_run_url,
)
from wavedl_daemon.storage.ownership import acquire_dispatch_lock


def test_already_running_check_does_not_create_state_root(tmp_path):
    root = tmp_path / "absent"
    check_not_running(root)
    assert not root.exists()


def test_held_lock_raises_already_running_with_url(tmp_path):
    with acquire_dispatch_lock(tmp_path / "dispatch.lock"):
        write_run_url(tmp_path, "http://127.0.0.1:8421/")
        with pytest.raises(AlreadyRunning) as error:
            check_not_running(tmp_path)
        assert error.value.url == "http://127.0.0.1:8421/"


def test_held_lock_without_run_file_has_no_url(tmp_path):
    with acquire_dispatch_lock(tmp_path / "dispatch.lock"):
        with pytest.raises(AlreadyRunning) as error:
            check_not_running(tmp_path)
        assert error.value.url is None


def test_stale_run_file_ignored_when_lock_free(tmp_path):
    with acquire_dispatch_lock(tmp_path / "dispatch.lock"):
        write_run_url(tmp_path, "http://127.0.0.1:8420/")
    assert check_not_running(tmp_path) is None


@pytest.mark.parametrize(
    "url", ["http://10.0.0.1:8420/", "http://127.0.0.1:8420/?bootstrap=x"]
)
def test_run_url_rejects_non_loopback_and_codes(tmp_path, url):
    write_run_url(tmp_path, url)
    assert read_run_url(tmp_path) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode confidentiality")
def test_run_file_is_private(tmp_path):
    write_run_url(tmp_path, "http://[::1]:8420/")
    assert stat.S_IMODE((tmp_path / "serve.json").stat().st_mode) == 0o600
    assert read_run_url(tmp_path) == "http://[::1]:8420/"
    remove_run_url(tmp_path)
    remove_run_url(tmp_path)
