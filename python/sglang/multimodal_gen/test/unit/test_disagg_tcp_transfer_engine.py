from __future__ import annotations

import ctypes
import hashlib
import json

import pytest
import zmq

from sglang.multimodal_gen.runtime.disaggregation.transport import (
    engine as engine_module,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    MockTransferEngine,
    ZmqTcpTransferEngine,
    create_transfer_engine,
)


def _buffer(payload: bytes):
    value = ctypes.create_string_buffer(payload, len(payload))
    return value, ctypes.addressof(value)


def _empty_buffer(size: int):
    value = ctypes.create_string_buffer(size)
    return value, ctypes.addressof(value)


def _raw_tcp_request(engine, header: dict, payload: bytes) -> dict:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, 1000)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.connect(engine._endpoint_from_session_id(engine.session_id))
    try:
        socket.send_multipart(
            [json.dumps(header, separators=(",", ":")).encode(), payload]
        )
        return json.loads(socket.recv())
    finally:
        socket.close(linger=0)
        context.term()


def test_zmq_tcp_engine_transfers_registered_host_memory():
    sender = ZmqTcpTransferEngine("127.0.0.1", timeout_s=1, max_retries=0)
    receiver = ZmqTcpTransferEngine("127.0.0.1", timeout_s=1, max_retries=0)
    payload = bytes(range(251)) * 17
    source, source_ptr = _buffer(payload)
    destination, destination_ptr = _empty_buffer(len(payload))
    sender.register_buffer(source_ptr, len(payload))
    receiver.register_buffer(destination_ptr, len(payload))

    try:
        result = sender.transfer_sync(
            receiver.session_id,
            source_ptr,
            destination_ptr,
            len(payload),
            transfer_id="request-a:encoder-to-denoiser:1",
        )

        assert result == 0
        assert sender.last_error is None
        assert destination.raw == payload
        assert source.raw == payload
    finally:
        sender.close()
        receiver.close()


def test_zmq_tcp_engine_rejects_unregistered_source_and_destination_ranges():
    sender = ZmqTcpTransferEngine("127.0.0.1", timeout_s=1, max_retries=0)
    receiver = ZmqTcpTransferEngine("127.0.0.1", timeout_s=1, max_retries=0)
    source, source_ptr = _buffer(b"abcdefgh")
    destination, destination_ptr = _empty_buffer(8)
    receiver.register_buffer(destination_ptr, 8)

    try:
        assert (
            sender.transfer_sync(receiver.session_id, source_ptr, destination_ptr, 8)
            < 0
        )
        assert "source range is not registered" in sender.last_error

        sender.register_buffer(source_ptr, 8)
        assert (
            sender.transfer_sync(
                receiver.session_id, source_ptr, destination_ptr + 4, 8
            )
            < 0
        )
        assert "destination range is not registered" in sender.last_error
        assert destination.raw == b"\0" * 8
    finally:
        sender.close()
        receiver.close()


def test_zmq_tcp_engine_rejects_payload_over_configured_limit_before_send():
    sender = ZmqTcpTransferEngine(
        "127.0.0.1", timeout_s=1, max_payload_bytes=4, max_retries=0
    )
    receiver = ZmqTcpTransferEngine(
        "127.0.0.1", timeout_s=1, max_payload_bytes=4, max_retries=0
    )
    source, source_ptr = _buffer(b"abcde")
    destination, destination_ptr = _empty_buffer(5)
    sender.register_buffer(source_ptr, 5)
    receiver.register_buffer(destination_ptr, 5)

    try:
        assert (
            sender.transfer_sync(receiver.session_id, source_ptr, destination_ptr, 5)
            < 0
        )
        assert "exceeds TCP limit 4" in sender.last_error
        assert destination.raw == b"\0" * 5
    finally:
        sender.close()
        receiver.close()


def test_zmq_tcp_receiver_rejects_checksum_mismatch_without_writing():
    receiver = ZmqTcpTransferEngine("127.0.0.1", timeout_s=1, max_retries=0)
    destination, destination_ptr = _empty_buffer(7)
    receiver.register_buffer(destination_ptr, 7)
    header = {
        "schema": "sglang.diffusion-zmq-tcp/v1",
        "transfer_id": "corrupt-transfer",
        "dst_addr": destination_ptr,
        "length": 7,
        "sha256": "0" * 64,
    }

    try:
        ack = _raw_tcp_request(receiver, header, b"payload")

        assert ack["status"] == "error"
        assert "checksum mismatch" in ack["error"]
        assert destination.raw == b"\0" * 7
    finally:
        receiver.close()


def test_zmq_tcp_receiver_idempotently_acknowledges_duplicate_transfer():
    receiver = ZmqTcpTransferEngine("127.0.0.1", timeout_s=1, max_retries=0)
    payload = b"immutable"
    destination, destination_ptr = _empty_buffer(len(payload))
    receiver.register_buffer(destination_ptr, len(payload))
    header = {
        "schema": "sglang.diffusion-zmq-tcp/v1",
        "transfer_id": "duplicate-transfer",
        "dst_addr": destination_ptr,
        "length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    try:
        first = _raw_tcp_request(receiver, header, payload)
        assert first["status"] == "ok"
        assert destination.raw == payload

        ctypes.memset(destination_ptr, ord("x"), len(payload))
        duplicate = _raw_tcp_request(receiver, header, payload)
        assert duplicate["status"] == "duplicate"
        assert destination.raw == b"x" * len(payload)
    finally:
        receiver.close()


def test_zmq_tcp_engine_timeout_is_bounded():
    sender = ZmqTcpTransferEngine(
        "127.0.0.1", timeout_s=0.02, max_payload_bytes=16, max_retries=1
    )
    source, source_ptr = _buffer(b"timeout")
    sender.register_buffer(source_ptr, 7)

    try:
        result = sender.transfer_sync(
            "zmq+tcp://127.0.0.1:1", source_ptr, source_ptr, 7
        )

        assert result < 0
        assert "timed out after 2 attempt(s)" in sender.last_error
    finally:
        sender.close()


def test_mock_engine_is_explicit_and_checks_both_registered_ranges():
    sender = create_transfer_engine(backend="mock")
    receiver = create_transfer_engine(backend="mock")
    assert isinstance(sender, MockTransferEngine)
    source, source_ptr = _buffer(b"mock")
    destination, destination_ptr = _empty_buffer(4)
    sender.register_buffer(source_ptr, 4)
    receiver.register_buffer(destination_ptr, 4)

    try:
        assert (
            sender.transfer_sync(receiver.session_id, source_ptr, destination_ptr, 4)
            == 0
        )
        assert destination.raw == b"mock"
    finally:
        sender.close()
        receiver.close()


def test_transfer_engine_factory_auto_falls_back_to_tcp(monkeypatch):
    monkeypatch.setattr(engine_module, "_check_mooncake", lambda: False)

    engine = create_transfer_engine(
        hostname="127.0.0.1",
        backend="auto",
        timeout_s=1,
        max_payload_bytes=1024,
        max_retries=0,
    )

    try:
        assert isinstance(engine, ZmqTcpTransferEngine)
        assert engine.backend_name == "tcp"
    finally:
        engine.close()


def test_transfer_engine_factory_explicit_mooncake_fails_closed(monkeypatch):
    monkeypatch.setattr(engine_module, "_check_mooncake", lambda: False)

    with pytest.raises(RuntimeError, match="requested but is not installed"):
        create_transfer_engine(backend="mooncake")
