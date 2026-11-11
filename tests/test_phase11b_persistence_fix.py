"""Windows persistence tests use temporary files only; no auction simulations."""

import ctypes
import errno
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes

import pytest

from tarneeb import auction_value_recovery as r


def windows_error(code):
    error = PermissionError(errno.EACCES, "injected Windows contention")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_replacement_retries_same_complete_temp(tmp_path, monkeypatch, code):
    path = tmp_path / "budget.json"
    r.atomic(path, {"spent": 4})
    original_replace = r.os.replace
    attempts, sleeps = [], []

    def replace(source, target):
        attempts.append((source, target))
        assert r.read(source) == {"spent": 7}
        if len(attempts) < 3:
            assert r.read(target) == {"spent": 4}
            raise windows_error(code)
        return original_replace(source, target)

    monkeypatch.setattr(r.os, "replace", replace)
    monkeypatch.setattr(r.time, "sleep", sleeps.append)
    r.atomic(path, {"spent": 7}, replace=True)
    assert len(set(attempts)) == 1 and len(attempts) == 3
    assert sleeps == [0.02, 0.04]
    assert r.read(path) == {"spent": 7}


def test_persistent_denial_is_bounded_and_preserves_old_budget(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    r.atomic(path, {"spent": 41})
    before = path.read_bytes()
    attempts, sleeps = [], []

    def denied(*args):
        attempts.append(args)
        raise windows_error(5)

    monkeypatch.setattr(r.os, "replace", denied)
    monkeypatch.setattr(r.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        r.atomic(path, {"spent": 42}, replace=True)
    assert len(attempts) == 8 and sum(sleeps) == pytest.approx(1.90)
    assert path.read_bytes() == before
    temps = list(tmp_path.glob("budget.json.*.tmp"))
    assert len(temps) == 1 and r.read(temps[0]) == {"spent": 42}


def test_unrelated_io_error_is_not_retried(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    r.atomic(path, {"spent": 4})
    attempts = []

    def disk_full(*args):
        attempts.append(args)
        raise OSError(errno.ENOSPC, "injected disk full")

    monkeypatch.setattr(r.os, "replace", disk_full)
    with pytest.raises(OSError, match="disk full"):
        r.atomic(path, {"spent": 5}, replace=True)
    assert len(attempts) == 1 and r.read(path) == {"spent": 4}


def test_real_windows_reader_lock_can_clear_during_retry(tmp_path):
    path = tmp_path / "budget.json"
    r.atomic(path, {"spent": 4})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
    assert handle != wintypes.HANDLE(-1).value
    released = threading.Event()

    def release():
        time.sleep(0.12)
        kernel.CloseHandle(handle)
        released.set()

    thread = threading.Thread(target=release)
    thread.start()
    try:
        r.atomic(path, {"spent": 5}, replace=True)
    finally:
        thread.join()
    assert released.is_set() and r.read(path) == {"spent": 5}


def test_mutable_writers_are_serialized(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    r.atomic(path, {"writer": -1})
    original_replace = r.os.replace
    guard = threading.Lock()
    active, maximum = 0, 0

    def observed_replace(source, target):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.03)
            return original_replace(source, target)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(r.os, "replace", observed_replace)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: r.atomic(path, {"writer": i}, replace=True), range(4)))
    assert maximum == 1
    assert r.read(path)["writer"] in range(4)


def test_amendment_accepts_only_pinned_repair_and_keeps_original_identity(monkeypatch):
    plan = r.read(r.ROOT / "preregistration.json")
    identity = plan["identity"]
    r.verify_plan(plan)
    assert plan["identity"] == identity
    original_sha = r.v1.sha
    monkeypatch.setattr(
        r.v1,
        "sha",
        lambda p: (
            "unapproved"
            if str(p).replace("\\", "/") == "tarneeb/auction_value_recovery.py"
            else original_sha(p)
        ),
    )
    with pytest.raises(ValueError, match="outside approved amendment"):
        r.verify_plan(plan)
