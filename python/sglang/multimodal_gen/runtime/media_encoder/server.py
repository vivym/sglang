# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU media encoder for a co-located disaggregated decoder."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import mmap
import os
import signal
import stat
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    MaterializedOutput,
    save_materialized_output,
)
from sglang.multimodal_gen.runtime.disaggregation.telemetry import (
    log_disagg_receipt,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.video_adapter import (
    _probe_minimax_h3_output_fields,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    init_logger,
)

from .client import validate_ipc_endpoint
from .protocol import (
    MEDIA_HEALTH_REQUEST,
    MEDIA_HEALTH_RESPONSE,
    MAX_CONTROL_MESSAGE_BYTES,
    MediaEncodeManifest,
    MediaEncodeRequest,
    MediaEncodeResponse,
)

logger = init_logger(__name__)
_MAX_DUPLICATE_WAITERS = 8


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_range(view: mmap.mmap, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    end = offset + nbytes
    buffer = memoryview(view)
    try:
        for start in range(offset, end, 8 * 1024 * 1024):
            digest.update(buffer[start : min(start + 8 * 1024 * 1024, end)])
    finally:
        buffer.release()
    return digest.hexdigest()


def _safe_error_message(error: BaseException) -> str:
    message = str(error).replace("\n", " ").strip()
    return message[:512] or type(error).__name__


@dataclass
class _OwnedPayload:
    request: MediaEncodeRequest
    fd: int

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


@dataclass
class _PendingJob:
    digest: str
    future: Future[MediaEncodeResponse]
    waiters: list[bytes] = field(default_factory=list)


class MediaEncoderServer:
    """Bounded ROUTER service with exact-once attempt handling."""

    def __init__(
        self,
        *,
        endpoint: str,
        shared_memory_root: str | os.PathLike[str],
        output_root: str | os.PathLike[str],
        max_payload_bytes: int,
        max_pending: int,
        workers: int,
        publish_mode: str = "local",
        completed_cache_size: int = 256,
        context: zmq.Context | None = None,
    ) -> None:
        self.endpoint = validate_ipc_endpoint(endpoint)
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        if max_pending <= 0 or workers <= 0 or workers > max_pending:
            raise ValueError("workers must be positive and no greater than max_pending")
        if completed_cache_size <= 0:
            raise ValueError("completed_cache_size must be positive")
        if publish_mode not in {"local", "s3"}:
            raise ValueError("publish_mode must be local or s3")
        self.shared_memory_root = self._prepare_root(
            shared_memory_root, "shared-memory"
        )
        self.output_root = self._prepare_root(output_root, "output")
        if (
            self.shared_memory_root == self.output_root
            or self.shared_memory_root in self.output_root.parents
            or self.output_root in self.shared_memory_root.parents
        ):
            raise ValueError(
                "shared-memory and output roots must be separate directory trees"
            )
        self._prepare_endpoint_parent(self.endpoint)
        self.max_payload_bytes = int(max_payload_bytes)
        self.max_pending = int(max_pending)
        self.workers = int(workers)
        self.publish_mode = publish_mode
        self.completed_cache_size = int(completed_cache_size)
        self._context = context or zmq.Context.instance()
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="sglang-media-encoder"
        )
        self._pending: dict[tuple[str, str], _PendingJob] = {}
        self._completed: OrderedDict[
            tuple[str, str], tuple[str, MediaEncodeResponse]
        ] = OrderedDict()
        self._stop_requested = threading.Event()
        self._ready = threading.Event()
        self._socket: zmq.Socket | None = None

        if self.publish_mode == "s3":
            from sglang.multimodal_gen.runtime.entrypoints.openai.storage import (
                CloudStorage,
            )

            self._cloud_storage = CloudStorage()
            if not self._cloud_storage.is_enabled():
                raise ValueError(
                    "publish_mode=s3 requires SGLANG_CLOUD_STORAGE_TYPE=s3, "
                    "bucket configuration, and boto3"
                )
        else:
            self._cloud_storage = None

    @staticmethod
    def _prepare_root(path: str | os.PathLike[str], label: str) -> Path:
        raw = Path(path).expanduser()
        if not raw.is_absolute():
            raise ValueError(f"media encoder {label} root must be absolute")
        raw.mkdir(parents=True, exist_ok=True, mode=0o770)
        if raw.is_symlink() or not raw.is_dir():
            raise ValueError(f"media encoder {label} root must be a real directory")
        return raw.resolve(strict=True)

    @staticmethod
    def _prepare_endpoint_parent(endpoint: str) -> None:
        socket_path = Path(endpoint.removeprefix("ipc://"))
        parent = socket_path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o770)
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(
                "media encoder ipc endpoint parent must be a real directory"
            )

    def request_stop(self) -> None:
        self._stop_requested.set()

    def wait_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    def _open_payload(self, request: MediaEncodeRequest) -> _OwnedPayload:
        request.validate(max_payload_bytes=self.max_payload_bytes)
        payload_path = self.shared_memory_root / request.payload_name
        if payload_path.parent != self.shared_memory_root:
            raise ValueError("payload path escapes the configured shared-memory root")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(payload_path, flags)
        except FileNotFoundError as exc:
            raise ValueError("media payload is unavailable") from exc
        except OSError as exc:
            raise ValueError("media payload cannot be opened safely") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("media payload must be a regular file")
            if info.st_size != request.payload_nbytes:
                raise ValueError(
                    "media payload size does not match its control descriptor"
                )
            # Taking ownership at accept time makes client timeout cleanup safe.
            payload_path.unlink()
            return _OwnedPayload(request=request, fd=fd)
        except Exception:
            os.close(fd)
            raise

    def _output_name(self, request: MediaEncodeRequest) -> str:
        identity = (
            f"{request.request_id}\0{request.attempt_id}\0{request.payload_sha256}"
        ).encode("utf-8")
        return f"h3-{hashlib.sha256(identity).hexdigest()}.mp4"

    def _publish(self, local_path: Path) -> tuple[str, str]:
        if self.publish_mode == "local":
            return local_path.as_uri(), local_path.name
        assert self._cloud_storage is not None
        uri = asyncio.run(self._cloud_storage.upload_and_cleanup(str(local_path)))
        if not uri:
            raise RuntimeError("S3 media publication failed")
        return uri, local_path.name

    def _encode_owned_payload(self, owned: _OwnedPayload) -> MediaEncodeResponse:
        request = owned.request
        total_started = time.monotonic()
        phase_times: dict[str, float] = {}
        output_bytes = 0
        mapping: mmap.mmap | None = None
        temporary_path: Path | None = None
        materialized: MaterializedOutput | None = None
        video: np.ndarray | None = None
        audio: np.ndarray | None = None
        try:
            if request.fps != 24:
                raise ValueError("MiniMax H3 media fps must be 24")
            if request.audio_sample_rate != 32_000 or request.audio.shape[-1] != 2:
                raise ValueError("MiniMax H3 media audio must be 32 kHz stereo")

            mapping = mmap.mmap(
                owned.fd, request.payload_nbytes, access=mmap.ACCESS_READ
            )
            checksum_started = time.monotonic()
            if (
                _sha256_range(mapping, 0, request.payload_nbytes)
                != request.payload_sha256
            ):
                raise ValueError("media payload checksum mismatch")
            if (
                _sha256_range(mapping, request.video.offset, request.video.nbytes)
                != request.video.sha256
            ):
                raise ValueError("video payload checksum mismatch")
            if (
                _sha256_range(mapping, request.audio.offset, request.audio.nbytes)
                != request.audio.sha256
            ):
                raise ValueError("audio payload checksum mismatch")
            phase_times["checksum_s"] = time.monotonic() - checksum_started

            video = np.ndarray(
                request.video.shape,
                dtype=np.uint8,
                buffer=mapping,
                offset=request.video.offset,
            )
            audio = np.ndarray(
                request.audio.shape,
                dtype=np.float32,
                buffer=mapping,
                offset=request.audio.offset,
            )
            frame_count, height, width, _channels = request.video.shape
            materialized = MaterializedOutput(
                sample=None,
                frames=list(video),
                audio=audio,
                fps=request.fps,
            )

            final_path = self.output_root / self._output_name(request)
            temporary_path = self.output_root / (
                f".{final_path.stem}.{uuid.uuid4().hex}.tmp.mp4"
            )
            encode_started = time.monotonic()
            save_materialized_output(
                materialized,
                DataType.VIDEO,
                str(temporary_path),
                save_output=True,
                audio_sample_rate=request.audio_sample_rate,
                output_compression=request.output_compression,
            )
            phase_times["ffmpeg_s"] = time.monotonic() - encode_started
            probe_started = time.monotonic()
            _probe_minimax_h3_output_fields(
                str(temporary_path),
                expected_frame_count=frame_count,
                expected_size=(width, height),
            )
            phase_times["ffprobe_s"] = time.monotonic() - probe_started
            output_hash_started = time.monotonic()
            byte_size = temporary_path.stat().st_size
            output_bytes = byte_size
            output_sha256 = _sha256_file(temporary_path)
            phase_times["output_hash_s"] = time.monotonic() - output_hash_started
            os.replace(temporary_path, final_path)
            temporary_path = None
            publish_started = time.monotonic()
            uri, object_key = self._publish(final_path)
            phase_times["publish_s"] = time.monotonic() - publish_started

            manifest = MediaEncodeManifest(
                request_id=request.request_id,
                attempt_id=request.attempt_id,
                uri=uri,
                object_key=object_key,
                storage=self.publish_mode,
                byte_size=byte_size,
                sha256=output_sha256,
                container="mp4",
                video_codec="h264",
                pixel_format="yuv420p",
                audio_codec="aac",
                width=width,
                height=height,
                frame_count=frame_count,
                fps=request.fps,
                duration_seconds=frame_count / request.fps,
                audio_sample_rate=request.audio_sample_rate,
                audio_channels=request.audio.shape[-1],
                source_payload_sha256=request.payload_sha256,
                model_identity=request.model_identity,
            )
            response = MediaEncodeResponse(
                request_id=request.request_id,
                attempt_id=request.attempt_id,
                status="ok",
                manifest=manifest,
            )
            log_disagg_receipt(
                logger,
                "cpu_media_encode",
                request.request_id,
                transfer_id=request.attempt_id,
                payload_bytes=request.payload_nbytes,
                output_bytes=output_bytes,
                total_s=time.monotonic() - total_started,
                status="ok",
                **phase_times,
            )
            return response
        except Exception as error:
            logger.exception(
                "Media encoding failed for request=%s attempt=%s",
                request.request_id,
                request.attempt_id,
            )
            log_disagg_receipt(
                logger,
                "cpu_media_encode",
                request.request_id,
                transfer_id=request.attempt_id,
                payload_bytes=request.payload_nbytes,
                output_bytes=output_bytes,
                total_s=time.monotonic() - total_started,
                status="error",
                error_type=type(error).__name__,
                **phase_times,
            )
            return MediaEncodeResponse(
                request_id=request.request_id,
                attempt_id=request.attempt_id,
                status="error",
                error_code="media_encode_failed",
                error=_safe_error_message(error),
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            # Drop numpy views before closing their backing mmap.
            materialized = None
            audio = None
            video = None
            if mapping is not None:
                mapping.close()
            owned.close()

    def _error_response(
        self,
        request_id: str,
        attempt_id: str,
        *,
        status: str,
        code: str,
        error: str,
    ) -> MediaEncodeResponse:
        return MediaEncodeResponse(
            request_id=request_id,
            attempt_id=attempt_id,
            status=status,
            error_code=code,
            error=error,
        )

    def _send_response(
        self, socket: zmq.Socket, identity: bytes, response: MediaEncodeResponse
    ) -> None:
        socket.send_multipart([identity, b"", response.encode()])

    def _accept(self, socket: zmq.Socket, identity: bytes, payload: bytes) -> None:
        if payload == MEDIA_HEALTH_REQUEST:
            socket.send_multipart([identity, b"", MEDIA_HEALTH_RESPONSE])
            return
        request_id = "unknown"
        attempt_id = "unknown"
        try:
            request = MediaEncodeRequest.decode(
                payload, max_payload_bytes=self.max_payload_bytes
            )
            request_id = request.request_id
            attempt_id = request.attempt_id
            key = request.operation_key
            digest = request.digest()

            completed = self._completed.get(key)
            if completed is not None:
                completed_digest, response = completed
                if completed_digest != digest:
                    response = self._error_response(
                        request_id,
                        attempt_id,
                        status="error",
                        code="attempt_conflict",
                        error="request/attempt identity was reused with different content",
                    )
                self._send_response(socket, identity, response)
                return

            pending = self._pending.get(key)
            if pending is not None:
                if pending.digest != digest:
                    self._send_response(
                        socket,
                        identity,
                        self._error_response(
                            request_id,
                            attempt_id,
                            status="error",
                            code="attempt_conflict",
                            error="request/attempt identity was reused with different content",
                        ),
                    )
                else:
                    if len(pending.waiters) >= _MAX_DUPLICATE_WAITERS:
                        self._send_response(
                            socket,
                            identity,
                            self._error_response(
                                request_id,
                                attempt_id,
                                status="busy",
                                code="duplicate_waiters_full",
                                error="too many duplicate waiters for this attempt",
                            ),
                        )
                    else:
                        pending.waiters.append(identity)
                return

            if len(self._pending) >= self.max_pending:
                self._send_response(
                    socket,
                    identity,
                    self._error_response(
                        request_id,
                        attempt_id,
                        status="busy",
                        code="queue_full",
                        error="media encoder pending queue is full",
                    ),
                )
                return

            owned = self._open_payload(request)
            try:
                future = self._executor.submit(self._encode_owned_payload, owned)
            except Exception:
                owned.close()
                raise
            self._pending[key] = _PendingJob(
                digest=digest, future=future, waiters=[identity]
            )
        except Exception as error:
            logger.warning("Rejected media request: %s", error)
            self._send_response(
                socket,
                identity,
                self._error_response(
                    request_id,
                    attempt_id,
                    status="error",
                    code="invalid_request",
                    error=_safe_error_message(error),
                ),
            )

    def _drain_completed(self, socket: zmq.Socket) -> None:
        for key, pending in list(self._pending.items()):
            if not pending.future.done():
                continue
            try:
                response = pending.future.result()
            except Exception as error:
                response = self._error_response(
                    key[0],
                    key[1],
                    status="error",
                    code="internal_error",
                    error=_safe_error_message(error),
                )
            for identity in pending.waiters:
                try:
                    self._send_response(socket, identity, response)
                except zmq.ZMQError:
                    logger.warning(
                        "Failed to send media response for request=%s attempt=%s",
                        *key,
                        exc_info=True,
                    )
            self._completed[key] = (pending.digest, response)
            self._completed.move_to_end(key)
            while len(self._completed) > self.completed_cache_size:
                self._completed.popitem(last=False)
            self._pending.pop(key, None)

    def serve_forever(self) -> None:
        socket = self._context.socket(zmq.ROUTER)
        self._socket = socket
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVHWM, self.max_pending)
        socket.setsockopt(zmq.SNDHWM, self.max_pending * 2)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.setsockopt(zmq.MAXMSGSIZE, MAX_CONTROL_MESSAGE_BYTES)
        socket.bind(self.endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        self._ready.set()
        logger.info(
            "Media encoder listening at %s (workers=%d, max_pending=%d)",
            self.endpoint,
            self.workers,
            self.max_pending,
        )
        try:
            while not self._stop_requested.is_set() or self._pending:
                self._drain_completed(socket)
                if self._stop_requested.is_set():
                    time.sleep(0.01)
                    continue
                events = dict(poller.poll(timeout=25))
                if socket not in events:
                    continue
                frames = socket.recv_multipart()
                if len(frames) != 3 or frames[1] != b"":
                    logger.warning("Rejected malformed media control envelope")
                    continue
                self._accept(socket, frames[0], frames[2])
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._drain_completed(socket)
            socket.close(linger=0)
            self._socket = None
            self._ready.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the co-located MiniMax H3 CPU media encoder service."
    )
    parser.add_argument("--endpoint", required=True, help="Absolute ipc:// endpoint")
    parser.add_argument("--shared-memory-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-payload-bytes", type=int, default=1536 * 1024**2)
    parser.add_argument("--max-pending", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--completed-cache-size", type=int, default=256)
    parser.add_argument("--publish-mode", choices=["local", "s3"], default="local")
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="info",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logger(args)
    service = MediaEncoderServer(
        endpoint=args.endpoint,
        shared_memory_root=args.shared_memory_root,
        output_root=args.output_root,
        max_payload_bytes=args.max_payload_bytes,
        max_pending=args.max_pending,
        workers=args.workers,
        publish_mode=args.publish_mode,
        completed_cache_size=args.completed_cache_size,
    )

    def request_stop(_signum: int, _frame: Any) -> None:
        service.request_stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    service.serve_forever()


if __name__ == "__main__":
    main()
