# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    _adaln_table_covers_timesteps,
)


def _fake_loader_model(
    fingerprint: str = "sha256:test",
    adapter_identity: tuple[str, float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        _adaln_artifact_config={
            "format_version": "1",
            "table_layout": "full",
            "source_fingerprint": fingerprint,
        },
        arch=SimpleNamespace(adaln_out_features=6, final_adaln_out_features=2),
        blocks=[object(), object()],
        video_patch_proj=SimpleNamespace(weight=torch.empty(0)),
        _expected_adaln_adapter_identity=adapter_identity,
        _loaded_adaln_adapter_identity=None,
    )


def _write_table(
    path,
    fingerprint: str = "sha256:test",
    adapter_identity: tuple[str, float] | None = None,
    *,
    adapter_application: str = "table_delta",
    adapter_tensors: str = "6",
) -> None:
    metadata = {
        "format_version": "1",
        "table_layout": "full",
        "source_fingerprint": fingerprint,
    }
    if adapter_identity is not None:
        metadata["adaln_adapter_sha256"] = adapter_identity[0]
        metadata["adaln_adapter_scale"] = str(adapter_identity[1])
        metadata["adaln_adapter_application"] = adapter_application
        metadata["adaln_adapter_tensors"] = adapter_tensors
    save_file(
        {
            "timesteps": torch.tensor([0.0, 0.5], dtype=torch.float32),
            "blocks.0": torch.zeros(2, 6, dtype=torch.bfloat16),
            "blocks.1": torch.ones(2, 6, dtype=torch.bfloat16),
            "final_layer": torch.zeros(2, 2, dtype=torch.bfloat16),
        },
        str(path),
        metadata=metadata,
    )


def test_offline_adaln_table_validates_provenance_and_shapes(tmp_path):
    path = tmp_path / "adaln.safetensors"
    _write_table(path)
    model = _fake_loader_model()
    MiniMaxH3DiTModel.load_adaln_table(model, str(path))
    assert len(model._adaln_table) == 2
    assert tuple(model._adaln_table[0].shape) == (2, 6)

    mismatched = _fake_loader_model("sha256:other")
    with pytest.raises(ValueError, match="source checkpoint mismatch"):
        MiniMaxH3DiTModel.load_adaln_table(mismatched, str(path))

    bad_contract = _fake_loader_model()
    bad_contract._adaln_artifact_config["table_layout"] = "tp-sharded"
    with pytest.raises(ValueError, match="checkpoint AdaLN table layout"):
        MiniMaxH3DiTModel.load_adaln_table(bad_contract, str(path))


def test_offline_adaln_table_requires_exact_adapter_identity(tmp_path):
    base_path = tmp_path / "base.safetensors"
    turbo_path = tmp_path / "turbo.safetensors"
    adapter = ("sha256:adapter", 1.0)
    _write_table(base_path)
    _write_table(turbo_path, adapter_identity=adapter)

    with pytest.raises(ValueError, match="adapter mismatch"):
        MiniMaxH3DiTModel.load_adaln_table(
            _fake_loader_model(adapter_identity=adapter), str(base_path)
        )
    with pytest.raises(ValueError, match="adapter mismatch"):
        MiniMaxH3DiTModel.load_adaln_table(_fake_loader_model(), str(turbo_path))

    matching = _fake_loader_model(adapter_identity=adapter)
    MiniMaxH3DiTModel.load_adaln_table(matching, str(turbo_path))
    assert matching._loaded_adaln_adapter_identity == adapter


@pytest.mark.parametrize(
    ("application", "tensor_count", "message"),
    [
        ("base_only", "6", "table_delta application"),
        ("table_delta", "4", "tensor count mismatch"),
        ("table_delta", "unknown", "invalid AdaLN table adapter tensor count"),
    ],
)
def test_offline_adaln_table_requires_adapter_application_contract(
    tmp_path, application, tensor_count, message
):
    path = tmp_path / "turbo.safetensors"
    adapter = ("sha256:adapter", 1.0)
    _write_table(
        path,
        adapter_identity=adapter,
        adapter_application=application,
        adapter_tensors=tensor_count,
    )

    with pytest.raises(ValueError, match=message):
        MiniMaxH3DiTModel.load_adaln_table(
            _fake_loader_model(adapter_identity=adapter), str(path)
        )


def test_offline_adaln_table_rejects_adapter_change_after_load():
    model = SimpleNamespace(
        _adaln_table=[torch.empty(0)],
        _loaded_adaln_adapter_identity=("sha256:first", 1.0),
        _expected_adaln_adapter_identity=("sha256:first", 1.0),
    )

    with pytest.raises(ValueError, match="changed after"):
        MiniMaxH3DiTModel.set_expected_adaln_adapter_identity(
            model, "sha256:second", 1.0
        )


def test_offline_adaln_table_is_not_all_gathered_again():
    class Projection:
        @staticmethod
        def split_output(value):
            return (value,)

    model = SimpleNamespace(
        _adaln_table=[torch.arange(12).reshape(2, 6)],
        blocks=[SimpleNamespace(adaln_proj=Projection())],
    )
    with patch(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
        "tensor_model_parallel_all_gather",
        side_effect=AssertionError("offline full-width table must not be gathered"),
    ):
        params = MiniMaxH3DiTModel._prepare_block_adaln_params(
            model, torch.empty(0), torch.tensor([1])
        )
    assert torch.equal(params[0][0], model._adaln_table[0][1:2])


def test_adaln_lookup_requires_exact_timestep_match():
    model = SimpleNamespace(_adaln_timesteps=torch.tensor([0.0, 0.5, 1.0]))
    indices = MiniMaxH3DiTModel._adaln_global_indices(model, torch.tensor([0.5]))
    assert indices.tolist() == [1]
    with pytest.raises(ValueError, match="does not contain"):
        MiniMaxH3DiTModel._adaln_global_indices(model, torch.tensor([0.25]))
    with pytest.raises(ValueError, match="does not contain"):
        MiniMaxH3DiTModel._adaln_global_indices(model, torch.tensor([1.1]))


def test_adaln_schedule_accepts_exact_table():
    timesteps = torch.tensor([0.0, 0.5, 1.0])
    assert _adaln_table_covers_timesteps(timesteps, timesteps.clone())


def test_adaln_schedule_accepts_exact_subset():
    cached = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert _adaln_table_covers_timesteps(cached, torch.tensor([0.25, 0.75]))


def test_adaln_schedule_rejects_missing_timestep():
    cached = torch.tensor([0.0, 0.5, 1.0])
    assert not _adaln_table_covers_timesteps(cached, torch.tensor([0.0, 0.25]))


def test_adaln_schedule_rejects_out_of_range_timestep():
    cached = torch.tensor([0.0, 0.5, 1.0])
    assert not _adaln_table_covers_timesteps(cached, torch.tensor([0.5, 1.1]))
