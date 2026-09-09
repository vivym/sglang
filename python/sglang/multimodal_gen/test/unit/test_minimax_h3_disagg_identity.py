from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.artifact_identity import (
    MINIMAX_H3_DISAGG_IDENTITY_SCHEMA,
    MINIMAX_H3_PRODUCTION_MANIFEST_SCHEMA,
    verify_minimax_h3_disagg_artifacts,
)


_ARTIFACT_PATHS = {
    "fl2va_model_index": "MiniMaxAI/MiniMax-H3/FL2VA/model_index.json",
    "fl2va_processor": "MiniMaxAI/MiniMax-H3/FL2VA/processor",
    "fl2va_tokenizer": "MiniMaxAI/MiniMax-H3/FL2VA/tokenizer",
    "fl2va_transformer_config": ("MiniMaxAI/MiniMax-H3/FL2VA/transformer/config.json"),
    "video_vae": "MiniMaxAI/MiniMax-H3/FL2VA/video_vae",
    "audio_vae": "MiniMaxAI/MiniMax-H3/FL2VA/audio_vae",
    "dit": "MiniMaxAI/MiniMax-H3-int8-v5-convrot-hardened/transformer",
    "encoder": "MiniMax-H3-encoder-int8",
    "adaln": "MiniMax-H3-adaln-table-hardened",
}


def _canonical_sha256(value) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_artifact(root: Path, relative: str) -> dict:
    path = root / relative
    paths = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    files = [
        {
            "path": item.relative_to(root).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": _file_sha256(item),
        }
        for item in paths
    ]
    identity = {
        "relative_path": relative,
        "file_count": len(files),
        "size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    return {**identity, "content_sha256": _canonical_sha256(identity)}


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "models"
    model = root / "MiniMaxAI/MiniMax-H3/FL2VA"
    _write_json(
        model / "model_index.json",
        {
            "_class_name": "MiniMaxH3Pipeline",
            "_minimax_h3": {"schema_version": 1, "partition": "fl2va"},
        },
    )
    _write_json(model / "processor/config.json", {"kind": "processor"})
    _write_json(model / "tokenizer/config.json", {"kind": "tokenizer"})
    _write_json(model / "transformer/config.json", {"hidden_size": 5376})
    _write_json(
        model / "video_vae/config.json",
        {"_class_name": "MiniMaxH3VideoVAE", "latent_channels": 24},
    )
    _write_json(
        model / "audio_vae/config.json",
        {"_class_name": "MiniMaxH3AudioVAE", "latent_channels": 32},
    )

    fingerprint = "sha256:" + "3" * 64
    dit = root / "MiniMaxAI/MiniMax-H3-int8-v5-convrot-hardened/transformer"
    _write_json(
        dit / "config.json",
        {
            "num_layers": 50,
            "hidden_size": 5376,
            "quantization_config": {
                "quant_method": "int8",
                "convrot": True,
                "ignored_layers": ["condition_proj", "token_refiner"],
                "minimax_h3_adaln_table": {
                    "format_version": "1",
                    "table_layout": "full",
                    "source_fingerprint": fingerprint,
                },
            },
        },
    )
    _write_json(dit / "model.safetensors.index.json", {"weight_map": {}})

    encoder = root / "MiniMax-H3-encoder-int8"
    _write_json(
        encoder / "config.json",
        {
            "text_config": {"hidden_size": 5120, "num_hidden_layers": 64},
            "quantization_config": {"quant_method": "int8", "convrot": True},
        },
    )
    _write_json(
        encoder / "model.safetensors.index.json",
        {
            "weight_map": {
                f"model.language_model.layers.{layer}.weight": "model.safetensors"
                for layer in range(50)
            }
        },
    )

    adaln = root / "MiniMax-H3-adaln-table-hardened/steps20.safetensors"
    adaln.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"timesteps": torch.arange(39, dtype=torch.float32)},
        adaln,
        metadata={
            "format_version": "1",
            "table_layout": "full",
            "num_blocks": "50",
            "num_steps": "20",
            "num_timesteps": "39",
            "flow_shift": "12.0",
            "audio_flow_shift": "3.0",
            "source_fingerprint": fingerprint,
        },
    )

    artifacts = {
        name: _build_artifact(root, relative)
        for name, relative in _ARTIFACT_PATHS.items()
    }
    content = {
        "schema": MINIMAX_H3_PRODUCTION_MANIFEST_SCHEMA,
        "artifacts": artifacts,
        "required_runtime_directories": [],
    }
    manifest = {
        **content,
        "source_root": str(root),
        "content_sha256": _canonical_sha256(content),
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return root, model, manifest_path


def _server_args(root: Path):
    return SimpleNamespace(
        component_paths={"text_encoder": str(root / "MiniMax-H3-encoder-int8")},
        component_weights_paths={},
        transformer_weights_path=str(
            root / "MiniMaxAI/MiniMax-H3-int8-v5-convrot-hardened/transformer"
        ),
        minimax_h3_adaln_cache_path=str(
            root / "MiniMax-H3-adaln-table-hardened/steps20.safetensors"
        ),
        minimax_h3_adaln_online=False,
    )


@pytest.mark.parametrize(
    "role", [RoleType.ENCODER, RoleType.DENOISER, RoleType.DECODER]
)
def test_h3_disagg_role_verifies_owned_artifacts(tmp_path, role):
    root, model, manifest = _build_fixture(tmp_path)

    identity = verify_minimax_h3_disagg_artifacts(
        manifest_path=str(manifest),
        artifact_root=str(root),
        model_path=str(model),
        partition="fl2va",
        role=role,
        server_args=_server_args(root),
    )

    assert identity["schema"] == MINIMAX_H3_DISAGG_IDENTITY_SCHEMA
    assert identity["manifest_content_sha256"].startswith("sha256:")
    assert identity["partition"] == "fl2va"


def test_h3_disagg_role_rejects_owned_file_tamper(tmp_path):
    root, model, manifest = _build_fixture(tmp_path)
    encoder_config = root / "MiniMax-H3-encoder-int8/config.json"
    encoder_config.write_text(encoder_config.read_text() + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="size mismatch|sha256 mismatch"):
        verify_minimax_h3_disagg_artifacts(
            manifest_path=str(manifest),
            artifact_root=str(root),
            model_path=str(model),
            partition="fl2va",
            role=RoleType.ENCODER,
            server_args=_server_args(root),
        )


def test_h3_disagg_role_rejects_semantically_wrong_manifested_dit(tmp_path):
    root, model, _ = _build_fixture(tmp_path)
    dit_config = (
        root / "MiniMaxAI/MiniMax-H3-int8-v5-convrot-hardened/transformer/config.json"
    )
    config = json.loads(dit_config.read_text(encoding="utf-8"))
    config["quantization_config"]["convrot"] = False
    _write_json(dit_config, config)
    artifacts = {
        name: _build_artifact(root, relative)
        for name, relative in _ARTIFACT_PATHS.items()
    }
    content = {
        "schema": MINIMAX_H3_PRODUCTION_MANIFEST_SCHEMA,
        "artifacts": artifacts,
        "required_runtime_directories": [],
    }
    manifest = {**content, "content_sha256": _canonical_sha256(content)}
    manifest_path = tmp_path / "wrong-dit-manifest.json"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="v5 ConvRot INT8"):
        verify_minimax_h3_disagg_artifacts(
            manifest_path=str(manifest_path),
            artifact_root=str(root),
            model_path=str(model),
            partition="fl2va",
            role=RoleType.DENOISER,
            server_args=_server_args(root),
        )


def test_h3_disagg_role_rejects_unmanifested_component_path(tmp_path):
    root, model, manifest = _build_fixture(tmp_path)
    args = _server_args(root)
    args.component_paths["text_encoder"] = str(tmp_path / "other-encoder")

    with pytest.raises(ValueError, match="artifact path.*does not match"):
        verify_minimax_h3_disagg_artifacts(
            manifest_path=str(manifest),
            artifact_root=str(root),
            model_path=str(model),
            partition="fl2va",
            role=RoleType.ENCODER,
            server_args=args,
        )
