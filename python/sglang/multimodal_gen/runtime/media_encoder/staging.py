# SPDX-License-Identifier: Apache-2.0
"""Bounded atomic tmpfs staging for decoder-produced RGB24 and PCM."""

from __future__ import annotations

import hashlib
import mmap
import os
import stat
import threading
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .protocol import MediaEncodeRequest, TensorDescriptor


_ALIGNMENT = 4096


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _secure_root(path: str | os.PathLike[str]) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ValueError("media shared-memory root must be an absolute path")
    raw.mkdir(parents=True, exist_ok=True, mode=0o770)
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError("media shared-memory root must be a real directory")
    return raw.resolve(strict=True)


def _write_all(fd: int, data: memoryview) -> None:
    while data:
        written = os.write(fd, data)
        if written <= 0:
            raise OSError("short write while staging media payload")
        data = data[written:]


class StagedMediaPayload:
    """Own one staged payload and release its bounded-capacity slot once."""

    def __init__(
        self,
        path: Path,
        release_slot,
        *,
        request_fields: dict[str, Any],
        video_fields: dict[str, Any],
        audio_fields: dict[str, Any],
    ) -> None:
        self.path = path
        self.request_id = request_fields["request_id"]
        self.attempt_id = request_fields["attempt_id"]
        self._release_slot = release_slot
        self._request_fields = request_fields
        self._video_fields = video_fields
        self._audio_fields = audio_fields
        self._request: MediaEncodeRequest | None = None
        self._request_lock = threading.Lock()
        self._released = False
        self._lock = threading.Lock()

    @property
    def request(self) -> MediaEncodeRequest:
        with self._request_lock:
            if self._request is not None:
                return self._request
            payload_sha256, video_sha256, audio_sha256 = self._hashes()
            request = MediaEncodeRequest(
                **self._request_fields,
                payload_sha256=payload_sha256,
                video=TensorDescriptor(**self._video_fields, sha256=video_sha256),
                audio=TensorDescriptor(**self._audio_fields, sha256=audio_sha256),
            )
            request.validate(max_payload_bytes=self._request_fields["payload_nbytes"])
            self._request = request
            return request

    def _hashes(self) -> tuple[str, str, str]:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.path, flags)
        mapping: mmap.mmap | None = None
        try:
            info = os.fstat(fd)
            expected_size = self._request_fields["payload_nbytes"]
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
                raise ValueError("staged media payload changed before checksum")
            mapping = mmap.mmap(fd, expected_size, access=mmap.ACCESS_READ)
            payload_digest = hashlib.sha256()
            video_digest = hashlib.sha256()
            audio_digest = hashlib.sha256()
            video_start = self._video_fields["offset"]
            video_end = video_start + self._video_fields["nbytes"]
            audio_start = self._audio_fields["offset"]
            audio_end = audio_start + self._audio_fields["nbytes"]
            view = memoryview(mapping)
            try:
                for start in range(0, expected_size, 8 * 1024 * 1024):
                    end = min(start + 8 * 1024 * 1024, expected_size)
                    payload_digest.update(view[start:end])
                    overlap_start = max(start, video_start)
                    overlap_end = min(end, video_end)
                    if overlap_start < overlap_end:
                        video_digest.update(view[overlap_start:overlap_end])
                    overlap_start = max(start, audio_start)
                    overlap_end = min(end, audio_end)
                    if overlap_start < overlap_end:
                        audio_digest.update(view[overlap_start:overlap_end])
            finally:
                view.release()
            return (
                payload_digest.hexdigest(),
                video_digest.hexdigest(),
                audio_digest.hexdigest(),
            )
        finally:
            if mapping is not None:
                mapping.close()
            os.close(fd)

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        try:
            self.path.unlink(missing_ok=True)
        finally:
            self._release_slot()

    def __enter__(self) -> StagedMediaPayload:
        return self

    def __exit__(self, *_args) -> None:
        self.release()


class SharedMemoryMediaStager:
    """Stage canonical media into immutable, bounded same-host payload files."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_payload_bytes: int,
        max_slots: int,
    ) -> None:
        if isinstance(max_payload_bytes, bool) or max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        if isinstance(max_slots, bool) or max_slots <= 0:
            raise ValueError("max_slots must be positive")
        self.root = _secure_root(root)
        self.max_payload_bytes = int(max_payload_bytes)
        self.max_slots = int(max_slots)
        self._slots = threading.BoundedSemaphore(self.max_slots)

    def stage(
        self,
        *,
        request_id: str,
        attempt_id: str,
        frames: Sequence[np.ndarray],
        audio: np.ndarray,
        fps: int,
        audio_sample_rate: int,
        output_compression: int | None,
        model_identity: dict[str, Any] | None,
    ) -> StagedMediaPayload:
        self._slots.acquire()
        try:
            return self._stage_acquired(
                request_id=request_id,
                attempt_id=attempt_id,
                frames=frames,
                audio=audio,
                fps=fps,
                audio_sample_rate=audio_sample_rate,
                output_compression=output_compression,
                model_identity=model_identity,
            )
        except Exception:
            self._slots.release()
            raise

    def _stage_acquired(
        self,
        *,
        request_id: str,
        attempt_id: str,
        frames: Sequence[np.ndarray],
        audio: np.ndarray,
        fps: int,
        audio_sample_rate: int,
        output_compression: int | None,
        model_identity: dict[str, Any] | None,
    ) -> StagedMediaPayload:
        if not frames:
            raise ValueError("media staging requires at least one video frame")
        first = np.asarray(frames[0])
        if first.dtype != np.uint8 or first.ndim != 3 or first.shape[-1] != 3:
            raise ValueError("video frames must be uint8 HWC RGB arrays")
        height, width, channels = (int(value) for value in first.shape)
        frame_nbytes = height * width * channels
        for index, frame in enumerate(frames):
            array = np.asarray(frame)
            if array.dtype != np.uint8 or tuple(array.shape) != tuple(first.shape):
                raise ValueError(
                    f"video frame {index} does not match the canonical frame layout"
                )

        audio_array = np.asarray(audio)
        if audio_array.dtype != np.float32 or audio_array.ndim != 2:
            raise ValueError("audio must be float32 [samples, channels]")
        audio_array = np.ascontiguousarray(audio_array)

        video_nbytes = len(frames) * frame_nbytes
        audio_offset = _align_up(video_nbytes)
        audio_nbytes = int(audio_array.nbytes)
        payload_nbytes = audio_offset + audio_nbytes
        if payload_nbytes > self.max_payload_bytes:
            raise ValueError(
                f"media payload requires {payload_nbytes} bytes, above configured "
                f"limit {self.max_payload_bytes}"
            )
        filesystem = os.statvfs(self.root)
        available = filesystem.f_bavail * filesystem.f_frsize
        if payload_nbytes > available:
            raise OSError(
                f"media shared-memory root has {available} bytes free, "
                f"requires {payload_nbytes}"
            )

        payload_name = f"{uuid.uuid4().hex}.bin"
        final_path = self.root / payload_name
        staging_path = self.root / f".{payload_name}.{uuid.uuid4().hex}.tmp"
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(staging_path, flags, 0o640)
        published = False
        try:
            os.ftruncate(fd, payload_nbytes)
            for frame in frames:
                contiguous = np.ascontiguousarray(frame)
                view = memoryview(contiguous).cast("B")
                _write_all(fd, view)

            padding = audio_offset - video_nbytes
            if padding:
                zero_padding = memoryview(bytes(padding))
                _write_all(fd, zero_padding)

            audio_view = memoryview(audio_array).cast("B")
            _write_all(fd, audio_view)
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(staging_path, final_path)

            staged = StagedMediaPayload(
                final_path,
                self._slots.release,
                request_fields={
                    "request_id": request_id,
                    "attempt_id": attempt_id,
                    "payload_name": payload_name,
                    "payload_nbytes": payload_nbytes,
                    "fps": fps,
                    "audio_sample_rate": audio_sample_rate,
                    "output_compression": output_compression,
                    "model_identity": model_identity,
                },
                video_fields={
                    "offset": 0,
                    "nbytes": video_nbytes,
                    "shape": (len(frames), height, width, channels),
                    "dtype": "uint8",
                    "layout": "THWC",
                },
                audio_fields={
                    "offset": audio_offset,
                    "nbytes": audio_nbytes,
                    "shape": tuple(int(value) for value in audio_array.shape),
                    "dtype": "float32",
                    "layout": "SC",
                },
            )
            published = True
            return staged
        finally:
            if fd >= 0:
                os.close(fd)
            if not published:
                staging_path.unlink(missing_ok=True)
                final_path.unlink(missing_ok=True)
