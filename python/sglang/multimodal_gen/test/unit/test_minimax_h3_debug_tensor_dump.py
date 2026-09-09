from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.disaggregation.boundary import (
    DISAGG_ATTEMPT_ID_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.debug_tensor_dump import (
    MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV,
    MINIMAX_H3_DEBUG_TENSOR_DUMP_SCHEMA,
    MINIMAX_H3_DEBUG_TENSOR_METADATA_KEY,
    MINIMAX_H3_DEBUG_TENSOR_NAMES,
    write_minimax_h3_debug_tensor_dump,
)


def _batch(*, request_id="request-1", attempt_id=None):
    extra = {}
    if attempt_id is not None:
        extra[DISAGG_ATTEMPT_ID_EXTRA_KEY] = attempt_id
    return SimpleNamespace(
        request_id=request_id,
        extra=extra,
        sampling_params=SimpleNamespace(prompt="a test prompt"),
        prompt="a test prompt",
        task="t2va",
        width=1344,
        height=768,
        fps=24,
        num_frames=124,
        num_inference_steps=2,
        flow_shift=12.0,
        audio_flow_shift=3.0,
        seed=6101,
    )


def _tensors():
    return {
        name: torch.arange(6, dtype=torch.float32).reshape(2, 3)
        for name in MINIMAX_H3_DEBUG_TENSOR_NAMES
    }


def _server_args(role):
    return SimpleNamespace(
        disagg_role=role,
        model_path="/models/MiniMax-H3",
        transformer_weights_path="/models/MiniMax-H3-int8-v5-convrot-hardened",
        component_paths={"text_encoder": "/models/MiniMax-H3-encoder-int8"},
        attention_backend="fa",
    )


def _pipeline(*, release_identity=None):
    if release_identity is None:
        release_identity = {
            "schema": "minimax-h3.disagg-release/v1",
            "manifest_content_sha256": "sha256:" + "1" * 64,
            "partition": "fl2va",
        }
    return SimpleNamespace(
        debug_release_identity=release_identity,
        disagg_release_identity=release_identity,
        release_metadata=SimpleNamespace(partition="fl2va"),
        model_path="/models/MiniMax-H3",
    )


def test_monolithic_debug_dump_is_identity_bound_and_immutable(tmp_path, monkeypatch):
    monkeypatch.setenv(MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV, str(tmp_path))
    tensors = _tensors()
    path = write_minimax_h3_debug_tensor_dump(
        batch=_batch(),
        tensors=tensors,
        server_args=_server_args(RoleType.MONOLITHIC),
        pipeline=_pipeline(),
    )

    assert path == tmp_path / "request-1/monolithic/direct/denoiser-output.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload[MINIMAX_H3_DEBUG_TENSOR_METADATA_KEY]
    assert metadata["schema"] == MINIMAX_H3_DEBUG_TENSOR_DUMP_SCHEMA
    assert metadata["request_id"] == "request-1"
    assert metadata["attempt_id"] == "direct"
    assert metadata["role"] == "monolithic"
    assert metadata["seed"] == 6101
    assert metadata["workload"]["num_frames"] == 124
    assert metadata["model_identity"] == {
        "kind": "verified_monolithic_release",
        "release_identity": {
            "schema": "minimax-h3.disagg-release/v1",
            "manifest_content_sha256": "sha256:" + "1" * 64,
            "partition": "fl2va",
        },
    }
    assert metadata["tensor_names"] == list(MINIMAX_H3_DEBUG_TENSOR_NAMES)
    for name in MINIMAX_H3_DEBUG_TENSOR_NAMES:
        assert torch.equal(payload[name], tensors[name])

    with pytest.raises(FileExistsError):
        write_minimax_h3_debug_tensor_dump(
            batch=_batch(),
            tensors=tensors,
            server_args=_server_args(RoleType.MONOLITHIC),
            pipeline=_pipeline(),
        )
    assert list(tmp_path.rglob("*.tmp")) == []


def test_disagg_dump_requires_and_separates_transfer_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv(MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV, str(tmp_path))
    release_identity = {
        "schema": "minimax-h3.disagg-release/v1",
        "manifest_content_sha256": "sha256:" + "1" * 64,
        "partition": "fl2va",
    }
    server_args = _server_args(RoleType.DENOISER)
    pipeline = _pipeline(release_identity=release_identity)

    with pytest.raises(ValueError, match="attempt_id"):
        write_minimax_h3_debug_tensor_dump(
            batch=_batch(),
            tensors=_tensors(),
            server_args=server_args,
            pipeline=pipeline,
        )

    first = write_minimax_h3_debug_tensor_dump(
        batch=_batch(attempt_id="transfer-a"),
        tensors=_tensors(),
        server_args=server_args,
        pipeline=pipeline,
    )
    second = write_minimax_h3_debug_tensor_dump(
        batch=_batch(attempt_id="transfer-b"),
        tensors=_tensors(),
        server_args=server_args,
        pipeline=pipeline,
    )
    assert first != second
    first_payload = torch.load(first, map_location="cpu", weights_only=True)
    first_metadata = first_payload[MINIMAX_H3_DEBUG_TENSOR_METADATA_KEY]
    assert first_metadata["attempt_id"] == "transfer-a"
    assert first_metadata["model_identity"] == {
        "kind": "verified_disaggregated_release",
        "release_identity": release_identity,
    }


def test_debug_dump_rejects_unsafe_identity_and_relative_root(tmp_path, monkeypatch):
    monkeypatch.setenv(MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV, "relative/root")
    with pytest.raises(ValueError, match="absolute path"):
        write_minimax_h3_debug_tensor_dump(
            batch=_batch(),
            tensors=_tensors(),
            server_args=_server_args(RoleType.MONOLITHIC),
            pipeline=_pipeline(),
        )

    monkeypatch.setenv(MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="request_id"):
        write_minimax_h3_debug_tensor_dump(
            batch=_batch(request_id="../escape"),
            tensors=_tensors(),
            server_args=_server_args(RoleType.MONOLITHIC),
            pipeline=_pipeline(),
        )


def test_debug_dump_rejects_non_denoising_role(tmp_path, monkeypatch):
    monkeypatch.setenv(MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="monolithic or denoiser"):
        write_minimax_h3_debug_tensor_dump(
            batch=_batch(attempt_id="transfer-1"),
            tensors=_tensors(),
            server_args=_server_args(RoleType.DECODER),
            pipeline=_pipeline(),
        )


def test_debug_dump_keeps_nonfinite_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv(MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV, str(tmp_path))
    tensors = _tensors()
    tensors["latents"][0, 0] = float("nan")
    path = write_minimax_h3_debug_tensor_dump(
        batch=_batch(),
        tensors=tensors,
        server_args=_server_args(RoleType.MONOLITHIC),
        pipeline=_pipeline(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.isnan(payload["latents"][0, 0])


def test_debug_dump_only_writes_global_rank_zero(tmp_path, monkeypatch):
    monkeypatch.setenv(MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    assert (
        write_minimax_h3_debug_tensor_dump(
            batch=_batch(),
            tensors=_tensors(),
            server_args=_server_args(RoleType.MONOLITHIC),
            pipeline=_pipeline(),
        )
        is None
    )
    assert list(tmp_path.rglob("*.pt")) == []
