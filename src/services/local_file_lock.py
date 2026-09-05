from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path.resolve()).casefold()
    with _LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


class LocalFileLock:
    """A small cross-process, one-byte lock for the local AI pipeline."""

    def __init__(self, path: Path, *, poll_seconds: float = 0.1) -> None:
        self.path = path.resolve()
        self.poll_seconds = poll_seconds
        self._thread_lock = _thread_lock(self.path)

    @contextmanager
    def acquire(self, timeout_seconds: float) -> Iterator[None]:
        deadline = time.monotonic() + timeout_seconds
        if not self._thread_lock.acquire(timeout=timeout_seconds):
            raise TimeoutError(f"timed out acquiring thread lock: {self.path}")
        handle: BinaryIO | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            while True:
                try:
                    self._lock_byte(handle)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out acquiring process lock: {self.path}"
                        ) from exc
                    time.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))
            try:
                yield
            finally:
                self._unlock_byte(handle)
        finally:
            if handle is not None:
                handle.close()
            self._thread_lock.release()

    @staticmethod
    def _lock_byte(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock_byte(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
