# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.layers.quantization import get_quantization_config
from sglang.multimodal_gen.runtime.layers.quantization.int8 import (
    Int8Config,
    Int8LinearMethod,
    _convrot_enabled,
)
from sglang.multimodal_gen.runtime.loader.minimax_h3_weights import (
    inspect_minimax_h3_safetensors,
    is_legacy_minimax_h3_int8_config,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    TransformerQuantLoadSpec,
)
from sglang.multimodal_gen.runtime.models.parameter import ChannelQuantScaleParameter


def test_serialized_int8_uses_output_channel_scale_parameter():
    layer = torch.nn.Module()
    method = Int8LinearMethod(Int8Config(is_checkpoint_int8_serialized=True))

    method.create_weights(
        layer=layer,
        input_size_per_partition=4,
        output_partition_sizes=[8],
        input_size=4,
        output_size=8,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *_args, **_kwargs: None,
    )

    assert isinstance(layer.weight_scale, ChannelQuantScaleParameter)
    assert layer.weight_scale.shape == (8, 1)
    assert not hasattr(layer.weight_scale, "input_dim")


def test_legacy_int8_is_registered_and_marked_serialized():
    assert get_quantization_config("int8") is Int8Config
    config = Int8Config.from_config({"quant_method": "int8", "convrot": True})
    spec = TransformerQuantLoadSpec(
        safetensors_list=[],
        quant_config=config,
        nunchaku_config=None,
        param_dtype=None,
    )
    assert spec.is_serialized_legacy_int8
    assert not spec.is_serialized_kitchen_int8
    assert not spec.uses_comfy_layer_markers


def _write_quant_config(tmp_path, quantization_config):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"quantization_config": quantization_config}), encoding="utf-8"
    )
    return str(config_path)


def _write_int8_checkpoint(tmp_path, *, include_scale=True, include_marker=False):
    prefix = "blocks.0.mlp.fc1"
    tensors = {f"{prefix}.weight": torch.ones((2, 4), dtype=torch.int8)}
    if include_scale:
        tensors[f"{prefix}.weight_scale"] = torch.ones((2, 1), dtype=torch.float32)
    if include_marker:
        marker = json.dumps({"format": "int8_tensorwise", "convrot": True}).encode()
        tensors[f"{prefix}.comfy_quant"] = torch.tensor(list(marker), dtype=torch.uint8)
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(tensors, checkpoint_path)
    return str(checkpoint_path)


def test_legacy_int8_checkpoint_requires_exact_config_and_tensor_contract(tmp_path):
    config_path = _write_quant_config(
        tmp_path, {"quant_method": "int8", "convrot": True}
    )
    checkpoint_path = _write_int8_checkpoint(tmp_path)

    assert is_legacy_minimax_h3_int8_config(config_path)
    curve_shape, markers = inspect_minimax_h3_safetensors(
        [checkpoint_path], legacy_int8=True
    )
    assert curve_shape is None
    assert markers == {}


def test_legacy_int8_checkpoint_rejects_convrot_false(tmp_path):
    config_path = _write_quant_config(
        tmp_path, {"quant_method": "int8", "convrot": False}
    )

    with pytest.raises(ValueError, match="requires convrot=true"):
        is_legacy_minimax_h3_int8_config(config_path)


def test_legacy_int8_checkpoint_rejects_missing_scale(tmp_path):
    checkpoint_path = _write_int8_checkpoint(tmp_path, include_scale=False)

    with pytest.raises(ValueError, match="is missing.*weight_scale"):
        inspect_minimax_h3_safetensors([checkpoint_path], legacy_int8=True)


def test_legacy_int8_checkpoint_rejects_comfy_marker(tmp_path):
    checkpoint_path = _write_int8_checkpoint(tmp_path, include_marker=True)

    with pytest.raises(ValueError, match="cannot mix .comfy_quant markers"):
        inspect_minimax_h3_safetensors([checkpoint_path], legacy_int8=True)


def test_unannounced_int8_checkpoint_still_requires_comfy_marker(tmp_path):
    config_path = _write_quant_config(
        tmp_path, {"quant_method": "kitchen_int8", "convrot": True}
    )
    checkpoint_path = _write_int8_checkpoint(tmp_path)

    assert not is_legacy_minimax_h3_int8_config(config_path)
    with pytest.raises(ValueError, match="missing comfy_quant metadata"):
        inspect_minimax_h3_safetensors([checkpoint_path])


def test_convrot_config_requires_json_boolean():
    with pytest.raises(ValueError, match="JSON boolean"):
        Int8Config.from_config({"quant_method": "int8", "convrot": "false"})


def test_adaln_provenance_flows_through_int8_override_config():
    artifact = {
        "format_version": "1",
        "table_layout": "full",
        "source_fingerprint": "sha256:test",
    }
    config = Int8Config.from_config(
        {
            "quant_method": "int8",
            "convrot": True,
            "minimax_h3_adaln_table": artifact,
        }
    )
    assert config.minimax_h3_adaln_table == artifact

    with pytest.raises(ValueError, match="must be an object"):
        Int8Config.from_config(
            {
                "quant_method": "int8",
                "minimax_h3_adaln_table": "sha256:test",
            }
        )


def test_convrot_env_cannot_override_checkpoint_marker(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_CONVROT", "0")
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        _convrot_enabled(True)

    monkeypatch.setenv("MINIMAX_H3_CONVROT", "1")
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        _convrot_enabled(False)

    assert _convrot_enabled(True)


def test_convrot_env_conflict_fails_during_config_construction(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_CONVROT", "0")
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        Int8Config.from_config({"quant_method": "int8", "convrot": True})
