# SPDX-License-Identifier: Apache-2.0
"""Transfer engines for tensor movement between disaggregated roles."""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import zmq

logger = logging.getLogger(__name__)

_MOONCAKE_AVAILABLE = None
_TCP_PROTOCOL_SCHEMA = "sglang.diffusion-zmq-tcp/v1"
_TCP_SESSION_PREFIX = "zmq+tcp://"
_TCP_HEADER_LIMIT_BYTES = 16 * 1024
_TCP_RECENT_TRANSFER_LIMIT = 1024


def _check_mooncake() -> bool:
    global _MOONCAKE_AVAILABLE
    if _MOONCAKE_AVAILABLE is None:
        try:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (  # noqa: F401
                MooncakeTransferEngine as _MTE,
            )

            _MOONCAKE_AVAILABLE = True
        except ImportError:
            _MOONCAKE_AVAILABLE = False
    return _MOONCAKE_AVAILABLE


def _validate_region(ptr: int, length: int, *, name: str) -> None:
    if isinstance(ptr, bool) or not isinstance(ptr, int) or ptr <= 0:
        raise ValueError(f"{name} pointer must be a positive integer")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError(f"{name} length must be a positive integer")


def _region_contains(base: int, size: int, ptr: int, length: int) -> bool:
    return base <= ptr and length <= size and ptr - base <= size - length


class BaseTransferEngine(ABC):
    """Abstract transfer engine for data movement between roles."""

    @property
    def supports_gpu_direct(self) -> bool:
        return False

    @property
    def backend_name(self) -> str:
        return type(self).__name__.lower()

    @property
    @abstractmethod
    def session_id(self) -> str: ...

    @property
    def last_error(self) -> str | None:
        return None

    @abstractmethod
    def register_buffer(self, ptr: int, length: int) -> None: ...

    @abstractmethod
    def deregister_buffer(self, ptr: int) -> None: ...

    @abstractmethod
    def transfer_sync(
        self,
        dst_session_id: str,
        src_addr: int,
        dst_addr: int,
        length: int,
        *,
        transfer_id: str | None = None,
    ) -> int:
        """Return zero on success and a negative value on failure."""

    @abstractmethod
    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
        *,
        transfer_id: str | None = None,
    ) -> int: ...

    def close(self) -> None:
        """Release engine-owned resources."""


class MooncakeDiffusionEngine(BaseTransferEngine):
    """Production engine backed by MooncakeTransferEngine (RDMA)."""

    @property
    def backend_name(self) -> str:
        return "mooncake"

    @property
    def supports_gpu_direct(self) -> bool:
        return True

    def __init__(
        self,
        hostname: str,
        gpu_id: int = 0,
        ib_device: str | None = None,
    ):
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            MooncakeTransferEngine,
        )

        self._engine = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=ib_device,
        )
        logger.info(
            "MooncakeDiffusionEngine initialized: session_id=%s",
            self._engine.session_id,
        )

    @property
    def session_id(self) -> str:
        return self._engine.session_id

    def register_buffer(self, ptr: int, length: int) -> None:
        self._engine.register(ptr, length)

    def deregister_buffer(self, ptr: int) -> None:
        self._engine.deregister(ptr)

    def transfer_sync(
        self,
        dst_session_id: str,
        src_addr: int,
        dst_addr: int,
        length: int,
        *,
        transfer_id: str | None = None,
    ) -> int:
        del transfer_id
        return self._engine.transfer_sync(dst_session_id, src_addr, dst_addr, length)

    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
        *,
        transfer_id: str | None = None,
    ) -> int:
        del transfer_id
        return self._engine.batch_transfer_sync(
            dst_session_id, src_addrs, dst_addrs, lengths
        )


class ZmqTcpTransferEngine(BaseTransferEngine):
    """Bounded, checksummed host-memory transport for ordinary Ethernet.

    The receiver owns a REP socket and accepts writes only into buffers that it
    registered locally. A transfer is acknowledged only after the complete
    payload has arrived, its SHA256 has matched, and it has been copied into the
    registered destination range. Retries reuse a transfer ID and are
    idempotently acknowledged by the receiver.
    """

    def __init__(
        self,
        hostname: str,
        *,
        timeout_s: float = 60.0,
        max_payload_bytes: int = 256 * 1024 * 1024,
        max_retries: int = 1,
    ) -> None:
        if not isinstance(hostname, str) or not hostname.strip():
            raise ValueError("TCP transfer hostname must be a non-empty string")
        hostname = hostname.strip()
        if "://" in hostname or "/" in hostname:
            raise ValueError(
                "TCP transfer hostname must not contain a URL scheme or path"
            )
        if hostname in {"0.0.0.0", "::"}:
            raise ValueError("TCP transfer hostname must be reachable by peer hosts")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("TCP transfer timeout must be a positive number")
        if timeout_s <= 0:
            raise ValueError("TCP transfer timeout must be positive")
        if (
            isinstance(max_payload_bytes, bool)
            or not isinstance(max_payload_bytes, int)
            or max_payload_bytes <= 0
        ):
            raise ValueError("TCP max payload bytes must be a positive integer")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("TCP transfer retries must be a non-negative integer")

        self._hostname = hostname
        self._timeout_ms = max(1, int(float(timeout_s) * 1000))
        self._max_payload_bytes = max_payload_bytes
        self._max_retries = max_retries
        self._registered: dict[int, int] = {}
        self._registered_lock = threading.Lock()
        self._recent: OrderedDict[str, tuple[int, int, str]] = OrderedDict()
        self._last_error: str | None = None
        self._context = zmq.Context(io_threads=1)
        self._running = threading.Event()
        self._running.set()
        self._ready = threading.Event()
        self._session_id = ""
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._serve,
            daemon=True,
            name="diffusion-zmq-tcp-receiver",
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            self.close()
            raise RuntimeError("timed out starting TCP transfer receiver")
        if self._startup_error is not None:
            error = self._startup_error
            self.close()
            raise RuntimeError("failed to start TCP transfer receiver") from error

    @property
    def backend_name(self) -> str:
        return "tcp"

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def register_buffer(self, ptr: int, length: int) -> None:
        _validate_region(ptr, length, name="registered buffer")
        with self._registered_lock:
            existing = self._registered.get(ptr)
            if existing is not None and existing != length:
                raise ValueError(
                    "buffer pointer is already registered with another length"
                )
            self._registered[ptr] = length

    def deregister_buffer(self, ptr: int) -> None:
        with self._registered_lock:
            self._registered.pop(ptr, None)

    def _contains_registered_region(self, ptr: int, length: int) -> bool:
        return any(
            _region_contains(base, size, ptr, length)
            for base, size in self._registered.items()
        )

    @staticmethod
    def _endpoint_from_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.startswith(
            _TCP_SESSION_PREFIX
        ):
            raise ValueError("destination is not a ZMQ TCP transfer session")
        return "tcp://" + session_id[len(_TCP_SESSION_PREFIX) :]

    def transfer_sync(
        self,
        dst_session_id: str,
        src_addr: int,
        dst_addr: int,
        length: int,
        *,
        transfer_id: str | None = None,
    ) -> int:
        try:
            _validate_region(src_addr, length, name="source")
            _validate_region(dst_addr, length, name="destination")
            if length > self._max_payload_bytes:
                raise ValueError(
                    f"payload size {length} exceeds TCP limit {self._max_payload_bytes}"
                )
            with self._registered_lock:
                if not self._contains_registered_region(src_addr, length):
                    raise ValueError("source range is not registered by this engine")
            endpoint = self._endpoint_from_session_id(dst_session_id)
            if transfer_id is None:
                transfer_id = uuid.uuid4().hex
            if not isinstance(transfer_id, str) or not transfer_id:
                raise ValueError("transfer_id must be a non-empty string")
            payload = ctypes.string_at(src_addr, length)
            digest = hashlib.sha256(payload).hexdigest()
            header = json.dumps(
                {
                    "schema": _TCP_PROTOCOL_SCHEMA,
                    "transfer_id": transfer_id,
                    "dst_addr": dst_addr,
                    "length": length,
                    "sha256": digest,
                },
                separators=(",", ":"),
            ).encode("utf-8")

            for attempt in range(self._max_retries + 1):
                socket = self._context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDHWM, 1)
                socket.setsockopt(zmq.RCVHWM, 1)
                socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
                socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
                socket.connect(endpoint)
                try:
                    socket.send_multipart([header, payload], copy=True)
                    raw_ack = socket.recv()
                    ack = json.loads(raw_ack)
                    self._validate_ack(
                        ack,
                        transfer_id=transfer_id,
                        length=length,
                        digest=digest,
                    )
                    self._last_error = None
                    return 0
                except zmq.Again:
                    if attempt >= self._max_retries:
                        raise TimeoutError(
                            f"TCP transfer {transfer_id!r} timed out after "
                            f"{self._max_retries + 1} attempt(s)"
                        )
                finally:
                    socket.close(linger=0)
            raise AssertionError("unreachable TCP transfer retry state")
        except Exception as exc:
            self._last_error = str(exc)
            logger.error("ZMQ TCP transfer failed: %s", exc)
            return -1

    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
        *,
        transfer_id: str | None = None,
    ) -> int:
        if not (len(src_addrs) == len(dst_addrs) == len(lengths)):
            self._last_error = "batch transfer address and length counts differ"
            return -1
        batch_id = transfer_id or uuid.uuid4().hex
        for index, (src_addr, dst_addr, length) in enumerate(
            zip(src_addrs, dst_addrs, lengths, strict=True)
        ):
            result = self.transfer_sync(
                dst_session_id,
                src_addr,
                dst_addr,
                length,
                transfer_id=f"{batch_id}:{index}",
            )
            if result != 0:
                return result
        return 0

    @staticmethod
    def _validate_ack(ack: Any, *, transfer_id: str, length: int, digest: str) -> None:
        if not isinstance(ack, dict) or ack.get("schema") != _TCP_PROTOCOL_SCHEMA:
            raise ValueError("TCP transfer peer returned an invalid acknowledgement")
        if ack.get("transfer_id") != transfer_id:
            raise ValueError("TCP transfer acknowledgement ID mismatch")
        if ack.get("status") not in {"ok", "duplicate"}:
            raise RuntimeError(
                str(ack.get("error") or "TCP transfer peer rejected data")
            )
        if ack.get("length") != length or not hmac.compare_digest(
            str(ack.get("sha256", "")), digest
        ):
            raise ValueError("TCP transfer acknowledgement integrity mismatch")

    def _serve(self) -> None:
        socket = None
        try:
            socket = self._context.socket(zmq.REP)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.RCVTIMEO, 100)
            socket.setsockopt(
                zmq.MAXMSGSIZE,
                max(self._max_payload_bytes, _TCP_HEADER_LIMIT_BYTES),
            )
            port = socket.bind_to_random_port("tcp://0.0.0.0")
            host = self._hostname
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            self._session_id = f"zmq+tcp://{host}:{port}"
            self._ready.set()

            while self._running.is_set():
                try:
                    frames = socket.recv_multipart(copy=True)
                except zmq.Again:
                    continue
                ack = self._receive_transfer(frames)
                try:
                    socket.send(
                        json.dumps(ack, separators=(",", ":")).encode("utf-8"),
                        copy=True,
                    )
                except zmq.Again:
                    logger.warning(
                        "TCP transfer acknowledgement timed out for %s",
                        ack.get("transfer_id", ""),
                    )
        except BaseException as exc:
            if not self._ready.is_set():
                self._startup_error = exc
                self._ready.set()
            elif self._running.is_set():
                logger.exception("TCP transfer receiver stopped unexpectedly")
        finally:
            if socket is not None:
                socket.close(linger=0)

    def _receive_transfer(self, frames: list[bytes]) -> dict[str, Any]:
        transfer_id = ""
        try:
            if len(frames) != 2:
                raise ValueError("TCP transfer must contain one header and one payload")
            header_bytes, payload = frames
            if len(header_bytes) > _TCP_HEADER_LIMIT_BYTES:
                raise ValueError("TCP transfer header exceeds the size limit")
            header = json.loads(header_bytes)
            if (
                not isinstance(header, dict)
                or header.get("schema") != _TCP_PROTOCOL_SCHEMA
            ):
                raise ValueError("unsupported TCP transfer protocol schema")
            transfer_id = header.get("transfer_id")
            if not isinstance(transfer_id, str) or not transfer_id:
                raise ValueError("TCP transfer_id must be a non-empty string")
            dst_addr = header.get("dst_addr")
            length = header.get("length")
            _validate_region(dst_addr, length, name="destination")
            if length > self._max_payload_bytes:
                raise ValueError(
                    f"payload size {length} exceeds TCP limit {self._max_payload_bytes}"
                )
            if len(payload) != length:
                raise ValueError(
                    f"TCP payload length mismatch: expected {length}, got {len(payload)}"
                )
            expected_digest = header.get("sha256")
            if not isinstance(expected_digest, str) or len(expected_digest) != 64:
                raise ValueError("TCP transfer SHA256 is invalid")
            actual_digest = hashlib.sha256(payload).hexdigest()
            if not hmac.compare_digest(actual_digest, expected_digest):
                raise ValueError("TCP transfer checksum mismatch")

            signature = (dst_addr, length, actual_digest)
            with self._registered_lock:
                if not self._contains_registered_region(dst_addr, length):
                    raise ValueError(
                        "destination range is not registered by this engine"
                    )
                previous = self._recent.get(transfer_id)
                if previous is not None:
                    if previous != signature:
                        raise ValueError(
                            "duplicate TCP transfer ID has conflicting metadata"
                        )
                    self._recent.move_to_end(transfer_id)
                    status = "duplicate"
                else:
                    ctypes.memmove(dst_addr, payload, length)
                    self._recent[transfer_id] = signature
                    while len(self._recent) > _TCP_RECENT_TRANSFER_LIMIT:
                        self._recent.popitem(last=False)
                    status = "ok"
            return {
                "schema": _TCP_PROTOCOL_SCHEMA,
                "transfer_id": transfer_id,
                "status": status,
                "length": length,
                "sha256": actual_digest,
            }
        except Exception as exc:
            return {
                "schema": _TCP_PROTOCOL_SCHEMA,
                "transfer_id": transfer_id,
                "status": "error",
                "error": str(exc),
            }

    def close(self) -> None:
        self._running.clear()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._context.term()


class MockTransferEngine(BaseTransferEngine):
    """In-process pointer-copy engine used only by tests."""

    _registry_lock = threading.Lock()
    _registry: dict[str, "MockTransferEngine"] = {}

    def __init__(self) -> None:
        self._session_id = f"mock://{uuid.uuid4().hex}"
        self._registered: dict[int, int] = {}
        self._last_error: str | None = None
        with self._registry_lock:
            self._registry[self._session_id] = self

    @property
    def backend_name(self) -> str:
        return "mock"

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def register_buffer(self, ptr: int, length: int) -> None:
        _validate_region(ptr, length, name="registered buffer")
        self._registered[ptr] = length

    def deregister_buffer(self, ptr: int) -> None:
        self._registered.pop(ptr, None)

    def _contains(self, ptr: int, length: int) -> bool:
        return any(
            _region_contains(base, size, ptr, length)
            for base, size in self._registered.items()
        )

    def transfer_sync(
        self,
        dst_session_id: str,
        src_addr: int,
        dst_addr: int,
        length: int,
        *,
        transfer_id: str | None = None,
    ) -> int:
        del transfer_id
        try:
            _validate_region(src_addr, length, name="source")
            _validate_region(dst_addr, length, name="destination")
            with self._registry_lock:
                peer = self._registry.get(dst_session_id)
                if peer is None:
                    raise ValueError("mock destination session is unavailable")
                if not self._contains(src_addr, length):
                    raise ValueError("mock source range is not registered")
                if not peer._contains(dst_addr, length):
                    raise ValueError("mock destination range is not registered")
                ctypes.memmove(dst_addr, src_addr, length)
            self._last_error = None
            return 0
        except Exception as exc:
            self._last_error = str(exc)
            return -1

    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
        *,
        transfer_id: str | None = None,
    ) -> int:
        del transfer_id
        if not (len(src_addrs) == len(dst_addrs) == len(lengths)):
            self._last_error = "batch transfer address and length counts differ"
            return -1
        for src_addr, dst_addr, length in zip(
            src_addrs, dst_addrs, lengths, strict=True
        ):
            result = self.transfer_sync(dst_session_id, src_addr, dst_addr, length)
            if result != 0:
                return result
        return 0

    def close(self) -> None:
        with self._registry_lock:
            self._registry.pop(self._session_id, None)


def create_transfer_engine(
    hostname: str = "127.0.0.1",
    gpu_id: int = 0,
    ib_device: str | None = None,
    *,
    backend: str = "auto",
    timeout_s: float = 60.0,
    max_payload_bytes: int = 256 * 1024 * 1024,
    max_retries: int = 1,
) -> BaseTransferEngine:
    """Create the selected disaggregated diffusion transfer engine."""
    if backend not in {"auto", "mock", "mooncake", "tcp"}:
        raise ValueError(f"unsupported disaggregated transfer backend {backend!r}")
    if backend == "mock":
        return MockTransferEngine()
    if backend == "tcp":
        return ZmqTcpTransferEngine(
            hostname,
            timeout_s=timeout_s,
            max_payload_bytes=max_payload_bytes,
            max_retries=max_retries,
        )
    if backend == "mooncake" or _check_mooncake():
        if not _check_mooncake():
            raise RuntimeError(
                "Mooncake transfer backend was requested but is not installed"
            )
        return MooncakeDiffusionEngine(
            hostname=hostname, gpu_id=gpu_id, ib_device=ib_device
        )

    logger.info("Mooncake is unavailable; using the bounded ZMQ TCP transfer backend")
    return ZmqTcpTransferEngine(
        hostname,
        timeout_s=timeout_s,
        max_payload_bytes=max_payload_bytes,
        max_retries=max_retries,
    )


__all__ = [
    "BaseTransferEngine",
    "MockTransferEngine",
    "MooncakeDiffusionEngine",
    "ZmqTcpTransferEngine",
    "create_transfer_engine",
]
