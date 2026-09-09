from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import sglang.multimodal_gen.runtime.media_encoder.server as media_encoder_server
from sglang.multimodal_gen.runtime.media_encoder.client import (
    MediaEncoderClient,
    MediaEncoderError,
)
from sglang.multimodal_gen.runtime.media_encoder.protocol import (
    MediaEncodeRequest,
    MediaEncodeResponse,
)
from sglang.multimodal_gen.runtime.media_encoder.server import (
    _MAX_DUPLICATE_WAITERS,
    MediaEncoderServer,
)
from sglang.multimodal_gen.runtime.media_encoder.staging import (
    SharedMemoryMediaStager,
)


def _media_arrays(frame_count: int = 2) -> tuple[list[np.ndarray], np.ndarray]:
    frames = [
        np.full((16, 32, 3), index * 31, dtype=np.uint8) for index in range(frame_count)
    ]
    sample_count = round(frame_count / 24 * 32_000)
    phase = np.linspace(0, 8 * np.pi, sample_count, dtype=np.float32)
    mono = np.sin(phase).astype(np.float32) * np.float32(0.1)
    audio = np.stack((mono, -mono), axis=1)
    return frames, audio


def _stage(tmp_path: Path, name: str = "request"):
    frames, audio = _media_arrays()
    stager = SharedMemoryMediaStager(
        tmp_path / "shm", max_payload_bytes=16 * 1024 * 1024, max_slots=2
    )
    return stager.stage(
        request_id=name,
        attempt_id=f"{name}-attempt",
        frames=frames,
        audio=audio,
        fps=24,
        audio_sample_rate=32_000,
        output_compression=50,
        model_identity={"partition": "fl2va", "manifest": "test"},
    )


def _endpoint() -> tuple[str, Path]:
    path = Path(f"/tmp/sglang-media-{uuid.uuid4().hex[:12]}.sock")
    return f"ipc://{path}", path


def _start_server(server: MediaEncoderServer):
    errors: list[BaseException] = []

    def run() -> None:
        try:
            server.serve_forever()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert server.wait_ready(2.0)
    return thread, errors


def _stop_server(
    server: MediaEncoderServer,
    thread: threading.Thread,
    errors: list[BaseException],
    socket_path: Path,
) -> None:
    server.request_stop()
    thread.join(timeout=10.0)
    socket_path.unlink(missing_ok=True)
    assert not thread.is_alive()
    assert errors == []


def test_media_service_creates_missing_ipc_parent(tmp_path):
    endpoint_parent = Path(f"/tmp/sglang-media-{uuid.uuid4().hex[:12]}")
    socket_path = endpoint_parent / "media.sock"
    service = None
    thread = None
    errors = []
    try:
        service = MediaEncoderServer(
            endpoint=f"ipc://{socket_path}",
            shared_memory_root=tmp_path / "shm",
            output_root=tmp_path / "output",
            max_payload_bytes=16 * 1024 * 1024,
            max_pending=1,
            workers=1,
        )
        assert endpoint_parent.is_dir()
        thread, errors = _start_server(service)
    finally:
        if service is not None and thread is not None:
            _stop_server(service, thread, errors, socket_path)
        elif service is not None:
            service._executor.shutdown(wait=True)
        socket_path.unlink(missing_ok=True)
        if endpoint_parent.exists():
            endpoint_parent.rmdir()


def test_media_encoder_main_configures_info_logging(monkeypatch):
    observed = {}

    class _Service:
        def __init__(self, **kwargs):
            observed["service_kwargs"] = kwargs

        def request_stop(self):
            pass

        def serve_forever(self):
            observed["served"] = True

    monkeypatch.setattr(media_encoder_server, "MediaEncoderServer", _Service)
    monkeypatch.setattr(
        media_encoder_server,
        "configure_logger",
        lambda args: observed.setdefault("log_level", args.log_level),
    )
    monkeypatch.setattr(media_encoder_server.signal, "signal", lambda *_args: None)

    media_encoder_server.main(
        [
            "--endpoint",
            "ipc:///tmp/media.sock",
            "--shared-memory-root",
            "/dev/shm/sglang-h3-media",
            "--output-root",
            "/tmp/sglang-h3-output",
        ]
    )

    assert observed["log_level"] == "info"
    assert observed["service_kwargs"]["endpoint"] == "ipc:///tmp/media.sock"
    assert observed["served"] is True


def test_media_request_rejects_path_traversal_and_shape_size_mismatch(tmp_path):
    staged = _stage(tmp_path)
    try:
        value = staged.request.to_dict()
        value["payload_name"] = "../escape.bin"
        with pytest.raises(ValueError, match="payload_name"):
            MediaEncodeRequest.decode(
                json.dumps(value).encode(),
                max_payload_bytes=16 * 1024 * 1024,
            )

        invalid_video = replace(
            staged.request.video,
            shape=(
                staged.request.video.shape[0] + 1,
                *staged.request.video.shape[1:],
            ),
        )
        with pytest.raises(ValueError, match="shape, dtype, and nbytes disagree"):
            replace(staged.request, video=invalid_video).validate(
                max_payload_bytes=16 * 1024 * 1024
            )
    finally:
        staged.release()


def test_staged_payload_has_exact_descriptors_and_idempotent_cleanup(tmp_path):
    staged = _stage(tmp_path)
    request = staged.request
    payload = staged.path.read_bytes()
    video = payload[request.video.offset : request.video.offset + request.video.nbytes]
    audio = payload[request.audio.offset : request.audio.offset + request.audio.nbytes]

    assert len(payload) == request.payload_nbytes
    assert hashlib.sha256(payload).hexdigest() == request.payload_sha256
    assert hashlib.sha256(video).hexdigest() == request.video.sha256
    assert hashlib.sha256(audio).hexdigest() == request.audio.sha256

    staged.release()
    staged.release()
    assert not staged.path.exists()


def test_media_service_rejects_checksum_corruption_and_cleans_payload(tmp_path):
    staged = _stage(tmp_path)
    with staged.path.open("r+b") as payload:
        payload.seek(staged.request.video.offset)
        payload.write(b"\xff")

    endpoint, socket_path = _endpoint()
    service = MediaEncoderServer(
        endpoint=endpoint,
        shared_memory_root=tmp_path / "shm",
        output_root=tmp_path / "output",
        max_payload_bytes=16 * 1024 * 1024,
        max_pending=1,
        workers=1,
    )
    owned = service._open_payload(staged.request)
    response = service._encode_owned_payload(owned)
    service._executor.shutdown(wait=True)
    staged.release()
    socket_path.unlink(missing_ok=True)

    assert response.status == "error"
    assert "checksum mismatch" in response.error
    assert not staged.path.exists()


def test_real_ffmpeg_ipc_round_trip_is_valid_and_idempotent(tmp_path):
    staged = _stage(tmp_path)
    endpoint, socket_path = _endpoint()
    service = MediaEncoderServer(
        endpoint=endpoint,
        shared_memory_root=tmp_path / "shm",
        output_root=tmp_path / "output",
        max_payload_bytes=16 * 1024 * 1024,
        max_pending=2,
        workers=1,
    )
    thread, errors = _start_server(service)
    try:
        client = MediaEncoderClient(endpoint, timeout_s=10.0, retries=0)
        assert client.health(timeout_s=1.0)
        first = client.encode(staged.request)
        second = client.encode(staged.request)

        assert first == second
        assert first.storage == "local"
        assert first.frame_count == 2
        assert first.width == 32
        assert first.height == 16
        output = Path(first.uri.removeprefix("file://"))
        assert output.is_file()
        assert output.stat().st_size == first.byte_size
        assert hashlib.sha256(output.read_bytes()).hexdigest() == first.sha256
        assert not staged.path.exists()
    finally:
        staged.release()
        _stop_server(service, thread, errors, socket_path)


def test_media_client_timeout_leaves_cleanup_with_staging_owner(tmp_path):
    staged = _stage(tmp_path)
    endpoint, socket_path = _endpoint()
    client = MediaEncoderClient(endpoint, timeout_s=0.05, retries=0)
    try:
        with pytest.raises(MediaEncoderError) as caught:
            client.encode(staged.request)
        assert caught.value.code == "timeout"
        assert staged.path.exists()
    finally:
        staged.release()
        socket_path.unlink(missing_ok=True)
    assert not staged.path.exists()


def test_media_client_retries_busy_response_before_returning_error(tmp_path):
    staged = _stage(tmp_path)
    responses = [
        MediaEncodeResponse(
            request_id=staged.request_id,
            attempt_id=staged.attempt_id,
            status="busy",
            error_code="queue_full",
            error="queue is full",
        ).encode(),
        MediaEncodeResponse(
            request_id=staged.request_id,
            attempt_id=staged.attempt_id,
            status="error",
            error_code="second_attempt",
            error="retry reached the service",
        ).encode(),
    ]

    class _Socket:
        def __init__(self, response):
            self.response = response

        def setsockopt(self, *_args):
            pass

        def connect(self, _endpoint):
            pass

        def send(self, _payload):
            pass

        def recv(self):
            return self.response

        def close(self, **_kwargs):
            pass

    class _Context:
        def __init__(self):
            self.socket_count = 0

        def socket(self, _socket_type):
            response = responses[self.socket_count]
            self.socket_count += 1
            return _Socket(response)

    context = _Context()
    try:
        with pytest.raises(MediaEncoderError) as caught:
            MediaEncoderClient(
                "ipc:///tmp/media-retry.sock",
                timeout_s=1.0,
                retries=1,
                context=context,
            ).encode(staged.request)
        assert caught.value.code == "second_attempt"
        assert context.socket_count == 2
    finally:
        staged.release()


def test_media_service_bounds_duplicate_waiters(tmp_path, monkeypatch):
    staged = _stage(tmp_path)
    request_payload = staged.request.encode()
    endpoint, socket_path = _endpoint()
    service = MediaEncoderServer(
        endpoint=endpoint,
        shared_memory_root=tmp_path / "shm",
        output_root=tmp_path / "output",
        max_payload_bytes=16 * 1024 * 1024,
        max_pending=1,
        workers=1,
    )
    encode_started = threading.Event()
    allow_encode = threading.Event()

    def blocked_encode(owned):
        encode_started.set()
        assert allow_encode.wait(timeout=5.0)
        owned.close()
        return service._error_response(
            staged.request_id,
            staged.attempt_id,
            status="error",
            code="test_complete",
            error="test completed",
        )

    class _Router:
        def __init__(self):
            self.messages = []

        def send_multipart(self, frames):
            self.messages.append(frames)

    monkeypatch.setattr(service, "_encode_owned_payload", blocked_encode)
    router = _Router()
    try:
        service._accept(router, b"original", request_payload)
        assert encode_started.wait(timeout=2.0)
        for index in range(_MAX_DUPLICATE_WAITERS - 1):
            service._accept(router, f"duplicate-{index}".encode(), request_payload)
        assert router.messages == []

        service._accept(router, b"overflow", request_payload)
        response = MediaEncodeResponse.decode(router.messages[-1][-1])
        assert response.status == "busy"
        assert response.error_code == "duplicate_waiters_full"
    finally:
        allow_encode.set()
        service._executor.shutdown(wait=True)
        service._drain_completed(router)
        staged.release()
        socket_path.unlink(missing_ok=True)


def test_queue_saturation_preserves_unaccepted_payload_and_shutdown_drains(
    tmp_path, monkeypatch
):
    first = _stage(tmp_path, "first")
    second = _stage(tmp_path, "second")
    endpoint, socket_path = _endpoint()
    service = MediaEncoderServer(
        endpoint=endpoint,
        shared_memory_root=tmp_path / "shm",
        output_root=tmp_path / "output",
        max_payload_bytes=16 * 1024 * 1024,
        max_pending=1,
        workers=1,
    )
    encode_started = threading.Event()
    allow_encode = threading.Event()
    original_encode = service._encode_owned_payload

    def blocked_encode(owned):
        encode_started.set()
        assert allow_encode.wait(timeout=5.0)
        return original_encode(owned)

    monkeypatch.setattr(service, "_encode_owned_payload", blocked_encode)
    thread, errors = _start_server(service)
    first_result: list[object] = []

    def encode_first() -> None:
        try:
            first_result.append(
                MediaEncoderClient(endpoint, timeout_s=10.0, retries=0).encode(
                    first.request
                )
            )
        except BaseException as error:
            first_result.append(error)

    client_thread = threading.Thread(target=encode_first)
    client_thread.start()
    assert encode_started.wait(timeout=2.0)
    try:
        with pytest.raises(MediaEncoderError) as caught:
            MediaEncoderClient(endpoint, timeout_s=2.0, retries=0).encode(
                second.request
            )
        assert caught.value.code == "queue_full"
        assert second.path.exists()

        service.request_stop()
        thread.join(timeout=0.05)
        assert thread.is_alive()

        allow_encode.set()
        client_thread.join(timeout=10.0)
        thread.join(timeout=10.0)
        assert not client_thread.is_alive()
        assert not thread.is_alive()
        assert len(first_result) == 1
        assert not isinstance(first_result[0], BaseException)
        assert errors == []
    finally:
        allow_encode.set()
        first.release()
        second.release()
        service.request_stop()
        client_thread.join(timeout=10.0)
        thread.join(timeout=10.0)
        socket_path.unlink(missing_ok=True)

    assert not first.path.exists()
    assert not second.path.exists()
    assert not any((tmp_path / "shm").iterdir())
