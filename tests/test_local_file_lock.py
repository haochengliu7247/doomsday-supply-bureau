import threading
from pathlib import Path

from src.services.local_file_lock import LocalFileLock


def test_local_file_lock_serializes_threads_and_releases(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.lock"
    first = LocalFileLock(path, poll_seconds=0.01)
    second = LocalFileLock(path, poll_seconds=0.01)
    errors: list[Exception] = []

    def contend() -> None:
        try:
            with second.acquire(0.05):
                raise AssertionError("contending lock unexpectedly acquired")
        except TimeoutError as exc:
            errors.append(exc)

    with first.acquire(1):
        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=1)

    assert len(errors) == 1
    with second.acquire(1):
        assert path.is_file()


def test_local_file_lock_releases_after_exception(tmp_path: Path) -> None:
    lock = LocalFileLock(tmp_path / "pipeline.lock")

    try:
        with lock.acquire(1):
            raise RuntimeError("expected")
    except RuntimeError:
        pass

    with lock.acquire(1):
        assert True
