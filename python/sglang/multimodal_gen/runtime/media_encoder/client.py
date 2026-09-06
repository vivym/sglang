# SPDX-License-Identifier: Apache-2.0
"""Thread-confined ZMQ client for the local media encoder service."""

from __future__ import annotations

import time

import zmq

from .protocol import (
    MEDIA_HEALTH_REQUEST,
    MEDIA_HEALTH_RESPONSE,
    MediaEncodeManifest,
    MediaEncodeRequest,
    MediaEncodeResponse,
)


class MediaEncoderError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def validate_ipc_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.startswith("ipc:///"):
        raise ValueError("media encoder endpoint must be an absolute ipc:// endpoint")
    if len(endpoint.encode("utf-8")) > 100:
        raise ValueError("media encoder ipc endpoint is too long")
    return endpoint


class MediaEncoderClient:
    """Synchronous client intended to live entirely on one background thread."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float,
        retries: int,
        context: zmq.Context | None = None,
    ) -> None:
        self.endpoint = validate_ipc_endpoint(endpoint)
        if timeout_s <= 0:
            raise ValueError("media encoder timeout must be positive")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("media encoder retries must be a non-negative integer")
        self.timeout_ms = max(1, int(timeout_s * 1000))
        self.retries = retries
        self._context = context or zmq.Context.instance()

    def _new_socket(self, *, timeout_ms: int | None = None) -> zmq.Socket:
        timeout_ms = self.timeout_ms if timeout_ms is None else timeout_ms
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.connect(self.endpoint)
        return socket

    def health(self, *, timeout_s: float = 1.0) -> bool:
        timeout_ms = max(1, int(timeout_s * 1000))
        socket = self._new_socket(timeout_ms=timeout_ms)
        try:
            socket.send(MEDIA_HEALTH_REQUEST)
            return socket.recv() == MEDIA_HEALTH_RESPONSE
        except (zmq.Again, zmq.ZMQError):
            return False
        finally:
            socket.close(linger=0)

    def encode(self, request: MediaEncodeRequest) -> MediaEncodeManifest:
        payload = request.encode()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            socket = self._new_socket()
            try:
                socket.send(payload)
                response = MediaEncodeResponse.decode(socket.recv())
                if (
                    response.request_id != request.request_id
                    or response.attempt_id != request.attempt_id
                ):
                    raise MediaEncoderError(
                        "identity_mismatch",
                        "media encoder response identity does not match request",
                    )
                if response.status != "ok":
                    response_error = MediaEncoderError(
                        response.error_code or "media_encode_failed",
                        response.error or "media encoder rejected the request",
                    )
                    if response.status != "busy":
                        raise response_error
                    last_error = response_error
                else:
                    assert response.manifest is not None
                    return response.manifest
            except zmq.Again:
                last_error = MediaEncoderError(
                    "timeout",
                    f"media encoder did not respond within {self.timeout_ms / 1000:g}s",
                )
            except (ValueError, zmq.ZMQError) as exc:
                last_error = exc
            finally:
                socket.close(linger=0)
            if attempt < self.retries:
                time.sleep(min(0.25 * (2**attempt), 1.0))

        if isinstance(last_error, MediaEncoderError):
            raise last_error
        raise MediaEncoderError("transport_error", str(last_error)) from last_error
