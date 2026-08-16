from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest

from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import (
    LoRAPipeline,
    _swap_peft_swiglu_fc1_lora_b,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_lora

_RANK_PATCH = "sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline.dist.get_rank"


class _TestLoRAPipeline(LoRAPipeline):
    def create_pipeline_stages(self, server_args):
        return None


def _make_layer() -> BaseLayerWithLoRA:
    return BaseLayerWithLoRA(torch.nn.Linear(2, 2, bias=False))


def _make_pipeline(layer: BaseLayerWithLoRA) -> _TestLoRAPipeline:
    pipeline = object.__new__(_TestLoRAPipeline)
    pipeline.modules = {"transformer": torch.nn.Module()}
    pipeline.server_args = SimpleNamespace(lora_merge_mode="dynamic")
    pipeline.lora_initialized = True
    pipeline.lora_adapters = defaultdict(dict)
    pipeline.loaded_adapter_paths = {"adapter": "/adapter"}
    pipeline.loaded_adapter_alphas = {"adapter": None}
    pipeline.loaded_adapter_sha256 = {}
    pipeline.loaded_adapter_has_adaln = {}
    pipeline.cur_adapter_name = {}
    pipeline.cur_adapter_path = {}
    pipeline.cur_adapter_strength = {}
    pipeline.cur_adapter_config = {}
    pipeline.lora_layers = {"linear": layer}
    pipeline.lora_layers_transformer_2 = {}
    pipeline.lora_layers_critic = {}
    pipeline.is_lora_merged = {}

    pipeline.lora_adapters["adapter"]["linear.lora_A"] = torch.ones(1, 2)
    pipeline.lora_adapters["adapter"]["linear.lora_B"] = torch.ones(2, 1)
    return pipeline


def test_h3_peft_swiglu_lora_b_swaps_only_external_fc1_rows():
    source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    swapped = _swap_peft_swiglu_fc1_lora_b(
        "transformer_blocks.0.ff.net.0.proj.lora_B",
        "blocks.0.mlp.fc1.lora_B",
        source,
    )
    assert torch.equal(swapped, torch.cat([source[3:], source[:3]], dim=0))
    assert (
        _swap_peft_swiglu_fc1_lora_b(
            "blocks.0.mlp.fc1.lora_B", "blocks.0.mlp.fc1.lora_B", source
        )
        is source
    )
    assert (
        _swap_peft_swiglu_fc1_lora_b(
            "transformer_blocks.0.ff.net.0.proj.lora_A",
            "blocks.0.mlp.fc1.lora_A",
            source,
        )
        is source
    )


def test_h3_peft_swiglu_lora_b_rejects_odd_output_rows():
    with torch.no_grad(), pytest.raises(ValueError, match="output rows must be even"):
        _swap_peft_swiglu_fc1_lora_b(
            "transformer_blocks.0.ff.net.0.proj.lora_B",
            "blocks.0.mlp.fc1.lora_B",
            torch.zeros(5, 2),
        )


def test_dynamic_lora_reactivates_cached_layers_without_weight_update_context():
    layer = _make_layer()
    pipeline = _make_pipeline(layer)
    context_calls = 0

    @contextmanager
    def counted_context(*args, **kwargs):
        nonlocal context_calls
        context_calls += 1
        yield []

    pipeline._temporarily_disable_offload = counted_context

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            "/adapter",
            target="transformer",
            strength=0.75,
            merge_mode="dynamic",
        )

    first_lora_a = layer.lora_A
    first_lora_b = layer.lora_B
    assert context_calls == 0
    assert not layer.disable_lora

    pipeline._temporarily_disable_offload = lambda *args, **kwargs: nullcontext([])
    pipeline.deactivate_lora_weights("transformer")
    assert layer.disable_lora

    def fail_apply(*args, **kwargs):
        raise AssertionError("cached dynamic LoRA should not rebuild weights")

    context_calls = 0
    pipeline._temporarily_disable_offload = counted_context
    pipeline._apply_lora_to_layers = fail_apply

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            None,
            target="transformer",
            strength=0.75,
            merge_mode="dynamic",
        )

    assert context_calls == 0
    assert not layer.disable_lora
    assert layer.lora_A is first_lora_a
    assert layer.lora_B is first_lora_b


def test_merged_lora_still_uses_weight_update_context():
    layer = _make_layer()
    pipeline = _make_pipeline(layer)
    context_calls = 0

    @contextmanager
    def counted_context(*args, **kwargs):
        nonlocal context_calls
        context_calls += 1
        yield []

    pipeline._temporarily_disable_offload = counted_context

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            "/adapter",
            target="transformer",
            strength=1.0,
            merge_mode="merge",
        )

    assert context_calls == 1
    assert layer.merged
    assert pipeline.is_lora_merged["transformer"]


def test_lora_alpha_override_updates_cached_adapter_scale():
    layer = _make_layer()
    pipeline = _make_pipeline(layer)

    with patch(_RANK_PATCH, return_value=0):
        pipeline.set_lora(
            "adapter",
            None,
            target="transformer",
            strength=1.0,
            merge_mode="dynamic",
            lora_alpha=8,
        )

    assert pipeline.loaded_adapter_alphas["adapter"] == 8
    assert layer.lora_rank == 1
    assert layer.lora_alpha == 8


def test_pinned_lora_weight_limits_snapshot_download(tmp_path):
    weight_name = "adapter-v4.safetensors"
    weight_path = tmp_path / weight_name
    weight_path.touch()

    download_target = (
        "sglang.multimodal_gen.runtime.utils.hf_diffusers_utils.maybe_download_model"
    )
    with patch(download_target, return_value=str(tmp_path)) as download:
        actual = maybe_download_lora("org/multi-adapter", weight_name=weight_name)

    assert actual == str(weight_path)
    assert download.call_args.kwargs["allow_patterns"] == [
        "*.json",
        weight_name,
        f"**/{weight_name}",
    ]
