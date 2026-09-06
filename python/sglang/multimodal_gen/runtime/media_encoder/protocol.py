# SPDX-License-Identifier: Apache-2.0
"""Versioned control protocol for same-host H3 media encoding."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal, Mapping
from urllib.parse import urlparse


MEDIA_PROTOCOL_SCHEMA = "sglang.minimax-h3.media/v1"
MAX_CONTROL_MESSAGE_BYTES = 64 * 1024
MAX_ID_BYTES = 192
MAX_PAYLOAD_BYTES = 8 * 1024**3

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9_.:@+-]+$")
_PAYLOAD_NAME_RE = re.compile(r"^[0-9a-f]{32}\.bin$")

MEDIA_HEALTH_REQUEST = json.dumps(
    {"op": "health", "schema": MEDIA_PROTOCOL_SCHEMA},
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
MEDIA_HEALTH_RESPONSE = json.dumps(
    {"op": "health", "schema": MEDIA_PROTOCOL_SCHEMA, "status": "ok"},
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, required: set[str], optional: set[str], name: str
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(missing)!r}, "
            f"unknown={sorted(unknown)!r}"
        )


def _require_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _require_identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if len(value.encode("utf-8")) > MAX_ID_BYTES or not _IDENTITY_RE.fullmatch(value):
        raise ValueError(f"{name} contains unsupported characters or is too long")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _json_copy(value: Any, name: str) -> Any:
    try:
        encoded = json.dumps(
            value, separators=(",", ":"), sort_keys=True, allow_nan=False
        )
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be JSON-safe") from exc


@dataclass(frozen=True)
class TensorDescriptor:
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: Literal["uint8", "float32"]
    layout: Literal["THWC", "SC"]
    sha256: str

    _ITEM_SIZES: ClassVar[dict[str, int]] = {"uint8": 1, "float32": 4}

    def validate(self, *, payload_nbytes: int, name: str) -> None:
        _require_int(self.offset, f"{name}.offset", minimum=0, maximum=payload_nbytes)
        _require_int(
            self.nbytes,
            f"{name}.nbytes",
            minimum=1,
            maximum=payload_nbytes,
        )
        if self.offset + self.nbytes > payload_nbytes:
            raise ValueError(f"{name} byte range exceeds the payload")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError(f"{name}.shape must be a non-empty integer sequence")
        element_count = 1
        for index, dimension in enumerate(self.shape):
            dimension = _require_int(
                dimension,
                f"{name}.shape[{index}]",
                minimum=1,
                maximum=1_000_000_000,
            )
            element_count *= dimension
            if element_count > MAX_PAYLOAD_BYTES:
                raise ValueError(f"{name}.shape is too large")
        if self.dtype not in self._ITEM_SIZES:
            raise ValueError(f"unsupported {name}.dtype {self.dtype!r}")
        if element_count * self._ITEM_SIZES[self.dtype] != self.nbytes:
            raise ValueError(f"{name} shape, dtype, and nbytes disagree")
        _require_sha256(self.sha256, f"{name}.sha256")

        if name == "video":
            if self.dtype != "uint8" or self.layout != "THWC":
                raise ValueError("video must use uint8 THWC")
            if len(self.shape) != 4 or self.shape[-1] != 3:
                raise ValueError("video must have shape [frames, height, width, 3]")
        elif name == "audio":
            if self.dtype != "float32" or self.layout != "SC":
                raise ValueError("audio must use float32 samples-by-channels layout")
            if len(self.shape) != 2 or not 1 <= self.shape[-1] <= 8:
                raise ValueError("audio must have shape [samples, channels]")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        return value

    @classmethod
    def from_dict(cls, value: Any, *, name: str) -> TensorDescriptor:
        mapping = _require_mapping(value, name)
        _require_exact_keys(
            mapping,
            required={"offset", "nbytes", "shape", "dtype", "layout", "sha256"},
            optional=set(),
            name=name,
        )
        shape = mapping["shape"]
        if not isinstance(shape, (list, tuple)):
            raise ValueError(f"{name}.shape must be an integer sequence")
        return cls(
            offset=mapping["offset"],
            nbytes=mapping["nbytes"],
            shape=tuple(shape),
            dtype=mapping["dtype"],
            layout=mapping["layout"],
            sha256=mapping["sha256"],
        )


@dataclass(frozen=True)
class MediaEncodeRequest:
    request_id: str
    attempt_id: str
    payload_name: str
    payload_nbytes: int
    payload_sha256: str
    video: TensorDescriptor
    audio: TensorDescriptor
    fps: int
    audio_sample_rate: int
    output_compression: int | None = None
    model_identity: dict[str, Any] | None = None
    schema: str = MEDIA_PROTOCOL_SCHEMA

    def validate(self, *, max_payload_bytes: int = MAX_PAYLOAD_BYTES) -> None:
        if self.schema != MEDIA_PROTOCOL_SCHEMA:
            raise ValueError(f"unsupported media protocol schema {self.schema!r}")
        _require_identity(self.request_id, "request_id")
        _require_identity(self.attempt_id, "attempt_id")
        if (
            not isinstance(self.payload_name, str)
            or _PAYLOAD_NAME_RE.fullmatch(self.payload_name) is None
        ):
            raise ValueError("payload_name must be a generated local payload name")
        _require_int(
            self.payload_nbytes,
            "payload_nbytes",
            minimum=1,
            maximum=min(max_payload_bytes, MAX_PAYLOAD_BYTES),
        )
        _require_sha256(self.payload_sha256, "payload_sha256")
        self.video.validate(payload_nbytes=self.payload_nbytes, name="video")
        self.audio.validate(payload_nbytes=self.payload_nbytes, name="audio")
        video_range = range(self.video.offset, self.video.offset + self.video.nbytes)
        audio_range = range(self.audio.offset, self.audio.offset + self.audio.nbytes)
        if (
            video_range.start < audio_range.stop
            and audio_range.start < video_range.stop
        ):
            raise ValueError("video and audio byte ranges overlap")
        _require_int(self.fps, "fps", minimum=1, maximum=240)
        _require_int(
            self.audio_sample_rate,
            "audio_sample_rate",
            minimum=8_000,
            maximum=192_000,
        )
        if self.output_compression is not None:
            _require_int(
                self.output_compression,
                "output_compression",
                minimum=0,
                maximum=100,
            )
        if self.model_identity is not None:
            copied = _json_copy(self.model_identity, "model_identity")
            if not isinstance(copied, dict):
                raise ValueError("model_identity must be an object")

    @property
    def operation_key(self) -> tuple[str, str]:
        return self.request_id, self.attempt_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
            "payload_name": self.payload_name,
            "payload_nbytes": self.payload_nbytes,
            "payload_sha256": self.payload_sha256,
            "video": self.video.to_dict(),
            "audio": self.audio.to_dict(),
            "fps": self.fps,
            "audio_sample_rate": self.audio_sample_rate,
            "output_compression": self.output_compression,
            "model_identity": self.model_identity,
        }

    def encode(self) -> bytes:
        self.validate()
        encoded = json.dumps(
            self.to_dict(), separators=(",", ":"), sort_keys=True, allow_nan=False
        ).encode("utf-8")
        if len(encoded) > MAX_CONTROL_MESSAGE_BYTES:
            raise ValueError("media request control message is too large")
        return encoded

    def digest(self) -> str:
        return hashlib.sha256(self.encode()).hexdigest()

    @classmethod
    def decode(
        cls, payload: bytes, *, max_payload_bytes: int = MAX_PAYLOAD_BYTES
    ) -> MediaEncodeRequest:
        if not isinstance(payload, bytes) or len(payload) > MAX_CONTROL_MESSAGE_BYTES:
            raise ValueError("invalid media request control message size")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("media request must be valid UTF-8 JSON") from exc
        mapping = _require_mapping(value, "media request")
        _require_exact_keys(
            mapping,
            required={
                "schema",
                "request_id",
                "attempt_id",
                "payload_name",
                "payload_nbytes",
                "payload_sha256",
                "video",
                "audio",
                "fps",
                "audio_sample_rate",
                "output_compression",
                "model_identity",
            },
            optional=set(),
            name="media request",
        )
        request = cls(
            schema=mapping["schema"],
            request_id=mapping["request_id"],
            attempt_id=mapping["attempt_id"],
            payload_name=mapping["payload_name"],
            payload_nbytes=mapping["payload_nbytes"],
            payload_sha256=mapping["payload_sha256"],
            video=TensorDescriptor.from_dict(mapping["video"], name="video"),
            audio=TensorDescriptor.from_dict(mapping["audio"], name="audio"),
            fps=mapping["fps"],
            audio_sample_rate=mapping["audio_sample_rate"],
            output_compression=mapping["output_compression"],
            model_identity=mapping["model_identity"],
        )
        request.validate(max_payload_bytes=max_payload_bytes)
        return request


@dataclass(frozen=True)
class MediaEncodeManifest:
    request_id: str
    attempt_id: str
    uri: str
    object_key: str
    storage: Literal["local", "s3"]
    byte_size: int
    sha256: str
    container: Literal["mp4"]
    video_codec: Literal["h264"]
    pixel_format: Literal["yuv420p"]
    audio_codec: Literal["aac"]
    width: int
    height: int
    frame_count: int
    fps: int
    duration_seconds: float
    audio_sample_rate: int
    audio_channels: int
    source_payload_sha256: str
    model_identity: dict[str, Any] | None = None
    schema: str = MEDIA_PROTOCOL_SCHEMA

    def validate(self) -> None:
        if self.schema != MEDIA_PROTOCOL_SCHEMA:
            raise ValueError("unsupported media manifest schema")
        _require_identity(self.request_id, "manifest.request_id")
        _require_identity(self.attempt_id, "manifest.attempt_id")
        if (
            not isinstance(self.uri, str)
            or not self.uri
            or len(self.uri.encode("utf-8")) > 2048
        ):
            raise ValueError("manifest.uri must be a non-empty string")
        if (
            not isinstance(self.object_key, str)
            or not self.object_key
            or len(self.object_key.encode("utf-8")) > 255
            or self.object_key in {".", ".."}
            or "/" in self.object_key
            or "\\" in self.object_key
        ):
            raise ValueError("manifest.object_key must be one safe basename")
        if self.storage not in {"local", "s3"}:
            raise ValueError("manifest.storage must be local or s3")
        parsed_uri = urlparse(self.uri)
        allowed_schemes = (
            {"file"} if self.storage == "local" else {"http", "https", "s3"}
        )
        if parsed_uri.scheme not in allowed_schemes:
            raise ValueError(
                f"manifest URI scheme is invalid for storage={self.storage!r}"
            )
        if self.storage == "local" and (
            parsed_uri.netloc or not parsed_uri.path.startswith("/")
        ):
            raise ValueError("local manifest URI must be an absolute file URI")
        _require_int(self.byte_size, "manifest.byte_size", minimum=1, maximum=2**63 - 1)
        _require_sha256(self.sha256, "manifest.sha256")
        _require_sha256(self.source_payload_sha256, "manifest.source_payload_sha256")
        if (
            self.container != "mp4"
            or self.video_codec != "h264"
            or self.pixel_format != "yuv420p"
            or self.audio_codec != "aac"
        ):
            raise ValueError("manifest codec contract is invalid")
        _require_int(self.width, "manifest.width", minimum=1, maximum=16_384)
        _require_int(self.height, "manifest.height", minimum=1, maximum=16_384)
        _require_int(
            self.frame_count, "manifest.frame_count", minimum=1, maximum=1_000_000
        )
        _require_int(self.fps, "manifest.fps", minimum=1, maximum=240)
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
        ):
            raise ValueError("manifest.duration_seconds must be positive and finite")
        _require_int(
            self.audio_sample_rate,
            "manifest.audio_sample_rate",
            minimum=8_000,
            maximum=192_000,
        )
        _require_int(
            self.audio_channels, "manifest.audio_channels", minimum=1, maximum=8
        )
        if self.model_identity is not None:
            copied = _json_copy(self.model_identity, "manifest.model_identity")
            if not isinstance(copied, dict):
                raise ValueError("manifest.model_identity must be an object")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> MediaEncodeManifest:
        mapping = _require_mapping(value, "media manifest")
        expected = {
            "schema",
            "request_id",
            "attempt_id",
            "uri",
            "object_key",
            "storage",
            "byte_size",
            "sha256",
            "container",
            "video_codec",
            "pixel_format",
            "audio_codec",
            "width",
            "height",
            "frame_count",
            "fps",
            "duration_seconds",
            "audio_sample_rate",
            "audio_channels",
            "source_payload_sha256",
            "model_identity",
        }
        _require_exact_keys(
            mapping,
            required=expected,
            optional=set(),
            name="media manifest",
        )
        manifest = cls(**mapping)
        manifest.validate()
        return manifest


@dataclass(frozen=True)
class MediaEncodeResponse:
    request_id: str
    attempt_id: str
    status: Literal["ok", "error", "busy"]
    manifest: MediaEncodeManifest | None = None
    error_code: str | None = None
    error: str | None = None
    schema: str = MEDIA_PROTOCOL_SCHEMA

    def validate(self) -> None:
        if self.schema != MEDIA_PROTOCOL_SCHEMA:
            raise ValueError("unsupported media response schema")
        _require_identity(self.request_id, "response.request_id")
        _require_identity(self.attempt_id, "response.attempt_id")
        if self.status not in {"ok", "error", "busy"}:
            raise ValueError("invalid media response status")
        if self.status == "ok":
            if (
                self.manifest is None
                or self.error_code is not None
                or self.error is not None
            ):
                raise ValueError(
                    "successful media response must contain only a manifest"
                )
            self.manifest.validate()
            if (
                self.manifest.request_id != self.request_id
                or self.manifest.attempt_id != self.attempt_id
            ):
                raise ValueError("media response and manifest identities disagree")
        else:
            if self.manifest is not None:
                raise ValueError("failed media response must not contain a manifest")
            if not isinstance(self.error_code, str) or not self.error_code:
                raise ValueError("failed media response requires an error_code")
            if not isinstance(self.error, str) or not self.error:
                raise ValueError("failed media response requires an error")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
            "status": self.status,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "error_code": self.error_code,
            "error": self.error,
        }

    def encode(self) -> bytes:
        encoded = json.dumps(
            self.to_dict(), separators=(",", ":"), sort_keys=True, allow_nan=False
        ).encode("utf-8")
        if len(encoded) > MAX_CONTROL_MESSAGE_BYTES:
            raise ValueError("media response control message is too large")
        return encoded

    @classmethod
    def decode(cls, payload: bytes) -> MediaEncodeResponse:
        if not isinstance(payload, bytes) or len(payload) > MAX_CONTROL_MESSAGE_BYTES:
            raise ValueError("invalid media response control message size")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("media response must be valid UTF-8 JSON") from exc
        mapping = _require_mapping(value, "media response")
        _require_exact_keys(
            mapping,
            required={
                "schema",
                "request_id",
                "attempt_id",
                "status",
                "manifest",
                "error_code",
                "error",
            },
            optional=set(),
            name="media response",
        )
        response = cls(
            schema=mapping["schema"],
            request_id=mapping["request_id"],
            attempt_id=mapping["attempt_id"],
            status=mapping["status"],
            manifest=(
                MediaEncodeManifest.from_dict(mapping["manifest"])
                if mapping["manifest"] is not None
                else None
            ),
            error_code=mapping["error_code"],
            error=mapping["error"],
        )
        response.validate()
        return response
