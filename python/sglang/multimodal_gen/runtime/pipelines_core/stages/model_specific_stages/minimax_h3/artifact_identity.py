# SPDX-License-Identifier: Apache-2.0
"""Role-local production artifact verification for MiniMax H3 disaggregation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from safetensors import safe_open

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

MINIMAX_H3_PRODUCTION_MANIFEST_SCHEMA = "h3-production-model-manifest-v1"
MINIMAX_H3_DISAGG_IDENTITY_SCHEMA = "minimax-h3.disagg-release/v1"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ENCODER_LAYER_RE = re.compile(r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_REQUIRED_ARTIFACTS = frozenset(
    {
        "fl2va_model_index",
        "fl2va_processor",
        "fl2va_tokenizer",
        "fl2va_transformer_config",
        "video_vae",
        "audio_vae",
        "dit",
        "encoder",
        "adaln",
    }
)
_ROLE_ARTIFACTS = {
    RoleType.ENCODER: (
        "fl2va_model_index",
        "fl2va_processor",
        "fl2va_tokenizer",
        "encoder",
    ),
    RoleType.DENOISER: (
        "fl2va_model_index",
        "fl2va_transformer_config",
        "dit",
        "adaln",
    ),
    RoleType.DECODER: (
        "fl2va_model_index",
        "video_vae",
        "audio_vae",
    ),
}


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(rendered).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\n" in value or "\0" in value:
        raise ValueError(f"MiniMax H3 manifest {field} must be a safe relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"MiniMax H3 manifest {field} must stay below artifact root")
    return path


def _load_and_validate_manifest(path: str) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read MiniMax H3 production manifest: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("MiniMax H3 production manifest must be a JSON object")
    if payload.get("schema") != MINIMAX_H3_PRODUCTION_MANIFEST_SCHEMA:
        raise ValueError("unsupported MiniMax H3 production manifest schema")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _REQUIRED_ARTIFACTS:
        raise ValueError("MiniMax H3 production manifest artifact set is incomplete")

    for name, artifact in artifacts.items():
        _validate_artifact_declaration(name, artifact)
    content_identity = {
        "schema": payload["schema"],
        "artifacts": artifacts,
        "required_runtime_directories": payload.get("required_runtime_directories", []),
    }
    expected_digest = _canonical_sha256(content_identity)
    if payload.get("content_sha256") != expected_digest:
        raise ValueError(
            "MiniMax H3 production manifest content_sha256 does not match its payload"
        )
    return payload


def _validate_artifact_declaration(name: str, artifact: Any) -> None:
    if not isinstance(artifact, dict):
        raise ValueError(f"MiniMax H3 manifest artifact {name!r} must be an object")
    relative_path = _safe_relative_path(
        artifact.get("relative_path"), field=f"artifacts.{name}.relative_path"
    )
    files = artifact.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"MiniMax H3 manifest artifact {name!r} has no files")
    declared_files: list[str] = []
    declared_size = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError(
                f"MiniMax H3 manifest artifact {name!r} has an invalid file entry"
            )
        file_path = _safe_relative_path(
            entry.get("path"), field=f"artifacts.{name}.files.path"
        )
        if file_path != relative_path and relative_path not in file_path.parents:
            raise ValueError(
                f"MiniMax H3 manifest file {file_path} is outside artifact {name!r}"
            )
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"MiniMax H3 manifest file {file_path} has invalid size_bytes"
            )
        digest = entry.get("sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"MiniMax H3 manifest file {file_path} has invalid sha256")
        declared_files.append(file_path.as_posix())
        declared_size += size
    if declared_files != sorted(set(declared_files)):
        raise ValueError(
            f"MiniMax H3 manifest artifact {name!r} files must be unique and sorted"
        )
    if artifact.get("file_count") != len(files):
        raise ValueError(
            f"MiniMax H3 manifest artifact {name!r} file_count is inconsistent"
        )
    if artifact.get("size_bytes") != declared_size:
        raise ValueError(
            f"MiniMax H3 manifest artifact {name!r} size_bytes is inconsistent"
        )
    identity = {
        "relative_path": artifact["relative_path"],
        "file_count": artifact["file_count"],
        "size_bytes": artifact["size_bytes"],
        "files": files,
    }
    if artifact.get("content_sha256") != _canonical_sha256(identity):
        raise ValueError(
            f"MiniMax H3 manifest artifact {name!r} content_sha256 is inconsistent"
        )


def _artifact_files(path: Path) -> list[Path]:
    if not path.exists():
        raise ValueError(f"MiniMax H3 production artifact is missing: {path}")
    if path.is_symlink():
        raise ValueError(f"MiniMax H3 production artifact cannot be a symlink: {path}")
    if path.is_file():
        return [path]
    files = []
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(
                f"MiniMax H3 production artifact cannot contain a symlink: {candidate}"
            )
        if candidate.is_file():
            files.append(candidate)
    if not files:
        raise ValueError(f"MiniMax H3 production artifact contains no files: {path}")
    return sorted(files)


def _verify_artifact(root: Path, name: str, artifact: Mapping[str, Any]) -> None:
    relative_path = _safe_relative_path(
        artifact["relative_path"], field=f"artifacts.{name}.relative_path"
    )
    expected_entries = {entry["path"]: entry for entry in artifact["files"]}
    actual_paths = _artifact_files(root / relative_path)
    actual_names = {path.relative_to(root).as_posix() for path in actual_paths}
    if actual_names != set(expected_entries):
        missing = sorted(set(expected_entries) - actual_names)
        unexpected = sorted(actual_names - set(expected_entries))
        raise ValueError(
            f"MiniMax H3 artifact {name!r} file set mismatch: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    def inspect(path: Path) -> tuple[str, int, str]:
        relative = path.relative_to(root).as_posix()
        return relative, path.stat().st_size, _file_sha256(path)

    with ThreadPoolExecutor(max_workers=min(2, len(actual_paths))) as executor:
        actual_entries = list(executor.map(inspect, actual_paths))
    for relative, size, digest in actual_entries:
        expected = expected_entries[relative]
        if size != expected["size_bytes"]:
            raise ValueError(f"MiniMax H3 artifact {name!r} size mismatch: {relative}")
        if digest != expected["sha256"]:
            raise ValueError(
                f"MiniMax H3 artifact {name!r} sha256 mismatch: {relative}"
            )


def _configured_component_path(
    server_args: Any, model_path: Path, component: str
) -> Path:
    weight_override = getattr(server_args, "component_weights_paths", {}).get(component)
    if weight_override is not None:
        return Path(weight_override).expanduser().resolve()
    override = getattr(server_args, "component_paths", {}).get(component)
    if override is not None:
        return Path(override).expanduser().resolve()
    return (model_path / component).resolve()


def _expected_artifact_path(root: Path, artifact: Mapping[str, Any]) -> Path:
    return (
        root / _safe_relative_path(artifact["relative_path"], field="relative_path")
    ).resolve()


def _require_exact_path(name: str, actual: Path, expected: Path) -> None:
    if actual != expected:
        raise ValueError(
            f"MiniMax H3 disagg artifact path for {name!r} does not match the "
            f"production manifest: configured={str(actual)!r}, expected={str(expected)!r}"
        )


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read MiniMax H3 {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"MiniMax H3 {name} must be a JSON object: {path}")
    return value


def _validate_model_index(model_path: Path, partition: str) -> None:
    model_index = _read_json(model_path / "model_index.json", name="model index")
    release = model_index.get("_minimax_h3")
    if model_index.get("_class_name") != "MiniMaxH3Pipeline" or not isinstance(
        release, dict
    ):
        raise ValueError("MiniMax H3 production model index identity is invalid")
    if release.get("schema_version") != 1 or release.get("partition") != partition:
        raise ValueError("MiniMax H3 production model index partition is invalid")


def _validate_encoder_identity(encoder_path: Path) -> None:
    config = _read_json(encoder_path / "config.json", name="encoder config")
    quant = config.get("quantization_config")
    text_config = config.get("text_config")
    if not isinstance(quant, dict) or (
        quant.get("quant_method") != "int8" or quant.get("convrot") is not True
    ):
        raise ValueError("MiniMax H3 encoder must be serialized ConvRot INT8")
    if not isinstance(text_config, dict) or (
        text_config.get("hidden_size") != 5120
        or text_config.get("num_hidden_layers") != 64
    ):
        raise ValueError("MiniMax H3 encoder architecture identity is invalid")
    if MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER != 50:
        raise ValueError("MiniMax H3 runtime must select exactly 50 encoder layers")
    index = _read_json(
        encoder_path / "model.safetensors.index.json", name="encoder weight index"
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("MiniMax H3 encoder weight index has no weight_map")
    layers = {
        int(match.group(1))
        for name in weight_map
        if isinstance(name, str) and (match := _ENCODER_LAYER_RE.match(name))
    }
    if not set(range(50)) <= layers:
        raise ValueError(
            "MiniMax H3 encoder checkpoint does not cover layers 0 through 49"
        )


def _validate_denoiser_identity(dit_path: Path, adaln_path: Path) -> None:
    config = _read_json(dit_path / "config.json", name="DiT config")
    quant = config.get("quantization_config")
    if (
        config.get("num_layers") != 50
        or config.get("hidden_size") != 5376
        or not isinstance(quant, dict)
        or quant.get("quant_method") != "int8"
        or quant.get("convrot") is not True
        or set(quant.get("ignored_layers", ())) != {"condition_proj", "token_refiner"}
    ):
        raise ValueError("MiniMax H3 DiT must be the v5 ConvRot INT8 contract")
    provenance = quant.get("minimax_h3_adaln_table")
    if not isinstance(provenance, dict) or (
        provenance.get("format_version") != "1"
        or provenance.get("table_layout") != "full"
        or _SHA256_RE.fullmatch(str(provenance.get("source_fingerprint", ""))) is None
    ):
        raise ValueError("MiniMax H3 DiT has invalid hardened AdaLN provenance")

    try:
        with safe_open(adaln_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception as exc:
        raise ValueError(
            f"cannot inspect MiniMax H3 AdaLN sidecar: {adaln_path}"
        ) from exc
    expected_metadata = {
        "format_version": "1",
        "table_layout": "full",
        "num_blocks": "50",
        "num_steps": "20",
        "num_timesteps": "39",
        "flow_shift": "12.0",
        "audio_flow_shift": "3.0",
        "source_fingerprint": provenance["source_fingerprint"],
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("MiniMax H3 hardened AdaLN sidecar metadata is incompatible")


def _validate_decoder_identity(video_vae_path: Path, audio_vae_path: Path) -> None:
    video = _read_json(video_vae_path / "config.json", name="Video VAE config")
    audio = _read_json(audio_vae_path / "config.json", name="Audio VAE config")
    if (
        video.get("_class_name") != "MiniMaxH3VideoVAE"
        or video.get("latent_channels") != 24
    ):
        raise ValueError("MiniMax H3 released Video VAE identity is invalid")
    if (
        audio.get("_class_name") != "MiniMaxH3AudioVAE"
        or audio.get("latent_channels") != 32
    ):
        raise ValueError("MiniMax H3 released Audio VAE identity is invalid")


def verify_minimax_h3_disagg_artifacts(
    *,
    manifest_path: str | None,
    artifact_root: str | None,
    model_path: str,
    partition: str,
    role: RoleType,
    server_args: Any,
) -> dict[str, str]:
    """Hash and semantically verify the artifacts owned by one H3 role."""
    if role not in _ROLE_ARTIFACTS:
        raise ValueError(f"MiniMax H3 artifact verification has no role {role.value!r}")
    if not manifest_path:
        raise ValueError(
            "MiniMax H3 disaggregation requires --minimax-h3-disagg-manifest-path"
        )
    manifest = _load_and_validate_manifest(manifest_path)
    root_value = artifact_root or manifest.get("source_root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(
            "MiniMax H3 disaggregation requires --minimax-h3-disagg-artifact-root "
            "when the manifest has no source_root"
        )
    root = Path(root_value).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    artifacts = manifest["artifacts"]

    _require_exact_path(
        "fl2va_model_index",
        model / "model_index.json",
        _expected_artifact_path(root, artifacts["fl2va_model_index"]),
    )
    if role is RoleType.ENCODER:
        for name, component in (
            ("fl2va_processor", "processor"),
            ("fl2va_tokenizer", "tokenizer"),
            ("encoder", "text_encoder"),
        ):
            _require_exact_path(
                name,
                _configured_component_path(server_args, model, component),
                _expected_artifact_path(root, artifacts[name]),
            )
    elif role is RoleType.DENOISER:
        _require_exact_path(
            "fl2va_transformer_config",
            model / "transformer" / "config.json",
            _expected_artifact_path(root, artifacts["fl2va_transformer_config"]),
        )
        dit_override = getattr(server_args, "component_weights_paths", {}).get(
            "transformer"
        ) or getattr(server_args, "transformer_weights_path", None)
        if not dit_override:
            raise ValueError(
                "MiniMax H3 denoiser role requires the production INT8 "
                "--transformer-weights-path"
            )
        dit_path = Path(dit_override).expanduser().resolve()
        _require_exact_path(
            "dit", dit_path, _expected_artifact_path(root, artifacts["dit"])
        )
        if getattr(server_args, "minimax_h3_adaln_online", False):
            raise ValueError(
                "MiniMax H3 production disaggregation requires the hardened AdaLN "
                "sidecar, not online reconstruction"
            )
        adaln_override = getattr(server_args, "minimax_h3_adaln_cache_path", None)
        if adaln_override is None:
            adaln_override = os.environ.get("MINIMAX_H3_ADALN_TABLE_PATH")
        if not adaln_override:
            raise ValueError(
                "MiniMax H3 denoiser role requires the hardened AdaLN sidecar"
            )
        adaln_path = Path(adaln_override).expanduser().resolve()
        adaln_files = artifacts["adaln"]["files"]
        if len(adaln_files) != 1:
            raise ValueError(
                "MiniMax H3 production AdaLN artifact must contain one file"
            )
        _require_exact_path(
            "adaln", adaln_path, (root / adaln_files[0]["path"]).resolve()
        )
    else:
        for name, component in (
            ("video_vae", "video_vae"),
            ("audio_vae", "audio_vae"),
        ):
            _require_exact_path(
                name,
                _configured_component_path(server_args, model, component),
                _expected_artifact_path(root, artifacts[name]),
            )

    started = time.perf_counter()
    for name in _ROLE_ARTIFACTS[role]:
        _verify_artifact(root, name, artifacts[name])
    _validate_model_index(model, partition)
    if role is RoleType.ENCODER:
        _validate_encoder_identity(_expected_artifact_path(root, artifacts["encoder"]))
    elif role is RoleType.DENOISER:
        _validate_denoiser_identity(dit_path, adaln_path)
    else:
        _validate_decoder_identity(
            _expected_artifact_path(root, artifacts["video_vae"]),
            _expected_artifact_path(root, artifacts["audio_vae"]),
        )
    logger.info(
        "Verified MiniMax H3 role=%s production artifacts against %s in %.3f seconds",
        role.value,
        manifest["content_sha256"],
        time.perf_counter() - started,
    )
    return {
        "schema": MINIMAX_H3_DISAGG_IDENTITY_SCHEMA,
        "manifest_content_sha256": manifest["content_sha256"],
        "partition": partition,
    }


__all__ = [
    "MINIMAX_H3_DISAGG_IDENTITY_SCHEMA",
    "MINIMAX_H3_PRODUCTION_MANIFEST_SCHEMA",
    "verify_minimax_h3_disagg_artifacts",
]
