# SPDX-License-Identifier: Apache-2.0
"""Helpers for transferring large numpy arrays between local scheduler processes."""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MIN_FILE_REF_BYTES = 32 << 20


@dataclass
class NumpyArrayFileRef:
    path: str

    def materialize(self) -> np.ndarray:
        try:
            return np.load(self.path, allow_pickle=False)
        finally:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class PendingOutputFileRef:
    """A local output path published atomically by a worker background task."""

    path: str
    staging_path: str
    ready_path: str
    error_path: str
    timeout_seconds: float = 3600.0

    @classmethod
    def create(
        cls,
        path: str,
        *,
        timeout_seconds: float = 3600.0,
    ) -> PendingOutputFileRef:
        final_path = os.path.abspath(path)
        parent = os.path.dirname(final_path) or "."
        os.makedirs(parent, exist_ok=True)
        token = uuid.uuid4().hex
        stem, suffix = os.path.splitext(final_path)
        staging_path = f"{stem}.{token}.pending{suffix}"
        marker_prefix = f"{final_path}.{token}"
        return cls(
            path=final_path,
            staging_path=staging_path,
            ready_path=f"{marker_prefix}.ready",
            error_path=f"{marker_prefix}.error",
            timeout_seconds=timeout_seconds,
        )

    def publish_success(self) -> None:
        os.replace(self.staging_path, self.path)
        self._publish_marker(self.ready_path, "ready")

    def publish_error(self, error: BaseException | str) -> None:
        try:
            os.unlink(self.staging_path)
        except FileNotFoundError:
            pass
        message = str(error)
        if isinstance(error, BaseException):
            message = f"{type(error).__name__}: {error}"
        self._publish_marker(self.error_path, message)

    def materialize(self) -> str:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if os.path.exists(self.error_path):
                try:
                    with open(self.error_path, encoding="utf-8") as f:
                        message = f.read().strip()
                finally:
                    self._cleanup_markers()
                raise RuntimeError(
                    f"Asynchronous output persistence failed for {self.path}: "
                    f"{message or 'unknown error'}"
                )
            if os.path.exists(self.ready_path):
                if not os.path.isfile(self.path):
                    self._cleanup_markers()
                    raise RuntimeError(
                        "Asynchronous output persistence reported success without "
                        f"publishing {self.path}"
                    )
                self._cleanup_markers()
                return self.path
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for asynchronous output persistence: "
                    f"{self.path}"
                )
            time.sleep(0.01)

    @staticmethod
    def _publish_marker(path: str, content: str) -> None:
        parent = os.path.dirname(path) or "."
        fd, temporary_path = tempfile.mkstemp(prefix=".sglang-marker-", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(temporary_path, path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    def _cleanup_markers(self) -> None:
        for path in (self.ready_path, self.error_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def is_local_endpoint(endpoint: str) -> bool:
    return endpoint.startswith(
        ("tcp://127.0.0.1:", "tcp://localhost:", "ipc://", "inproc://")
    )


def spill_large_arrays_to_file_refs(value: Any) -> Any:
    directory = _array_ipc_dir()
    if directory is None:
        return value
    return _spill_large_arrays_to_file_refs(value, directory)


def _spill_large_arrays_to_file_refs(value: Any, directory: str) -> Any:
    if isinstance(value, np.ndarray) and value.nbytes >= _MIN_FILE_REF_BYTES:
        # only spill if the array size is above the threshold. if not, it's not worth it
        return _spill_array(value, directory)
    if isinstance(value, list):
        return [_spill_large_arrays_to_file_refs(item, directory) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _spill_large_arrays_to_file_refs(item, directory) for item in value
        )
    return value


def materialize_file_refs(value: Any) -> Any:
    if isinstance(value, (NumpyArrayFileRef, PendingOutputFileRef)):
        return value.materialize()
    if isinstance(value, list):
        return [materialize_file_refs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_file_refs(item) for item in value)
    return value


def _spill_array(array: np.ndarray, directory: str) -> NumpyArrayFileRef:
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)

    fd, path = tempfile.mkstemp(
        prefix="sgldiffusion-array-",
        suffix=".npy",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array, allow_pickle=False)
    except Exception:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    return NumpyArrayFileRef(path=path)


def _array_ipc_dir() -> str | None:
    shm_path = Path("/dev/shm")
    if shm_path.is_dir() and os.access(shm_path, os.W_OK):
        return str(shm_path)
    return None
