import pytest
import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)


def _per_tensor_scale(values: list[float]) -> PerTensorScaleParameter:
    return PerTensorScaleParameter(
        data=torch.tensor(values, dtype=torch.float32),
        weight_loader=lambda *_args, **_kwargs: None,
    )


def _per_channel_scale(rows: int) -> ChannelQuantScaleParameter:
    return ChannelQuantScaleParameter(
        data=torch.empty((rows, 1), dtype=torch.float32),
        output_dim=0,
        weight_loader=lambda *_args, **_kwargs: None,
    )


def test_column_parallel_per_channel_scale_shards_output_channels():
    layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
    layer.tp_rank = 1
    param = _per_channel_scale(4)
    loaded_weight = torch.arange(8, dtype=torch.float32).reshape(8, 1)

    layer.weight_loader(param, loaded_weight)

    assert torch.equal(param.data, loaded_weight[4:])


def test_row_parallel_per_channel_scale_is_replicated():
    layer = RowParallelLinear.__new__(RowParallelLinear)
    layer.tp_rank = 1
    param = _per_channel_scale(8)
    loaded_weight = torch.arange(8, dtype=torch.float32).reshape(8, 1)

    layer.weight_loader(param, loaded_weight)

    assert torch.equal(param.data, loaded_weight)


@pytest.mark.parametrize("layer_cls", [ColumnParallelLinear, RowParallelLinear])
def test_tp1_channel_scale_matches_legacy_model_weight_parameter(layer_cls):
    layer = layer_cls.__new__(layer_cls)
    layer.tp_rank = 0
    loaded_weight = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    channel_scale = _per_channel_scale(8)
    legacy_scale = ModelWeightParameter(
        data=torch.empty_like(loaded_weight),
        input_dim=1,
        output_dim=0,
        weight_loader=lambda *_args, **_kwargs: None,
    )

    layer.weight_loader(channel_scale, loaded_weight)
    layer.weight_loader(legacy_scale, loaded_weight)

    assert torch.equal(channel_scale.data, loaded_weight)
    assert torch.equal(channel_scale.data, legacy_scale.data)


@pytest.mark.parametrize("loaded_weight", [torch.tensor(0.25), torch.tensor([0.25])])
def test_merged_column_parallel_scalar_scale_load_fills_fused_slots(loaded_weight):
    layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    layer.tp_size = 2
    param = _per_tensor_scale([-1.0, -2.0])

    layer.weight_loader_v2(param, loaded_weight)

    assert torch.equal(param.data, torch.tensor([0.25, 0.25]))


@pytest.mark.parametrize("loaded_weight", [torch.tensor(0.5), torch.tensor([0.5])])
def test_qkv_parallel_scalar_scale_load_fills_fused_slots(loaded_weight):
    layer = QKVParallelLinear.__new__(QKVParallelLinear)
    layer.tp_size = 2
    param = _per_tensor_scale([-1.0, -2.0, -3.0])

    layer.weight_loader_v2(param, loaded_weight)

    assert torch.equal(param.data, torch.tensor([0.5, 0.5, 0.5]))


def test_merged_column_parallel_full_scale_vector_loads_all_fused_slots():
    layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    layer.tp_size = 1
    param = _per_tensor_scale([-1.0, -2.0])

    layer.weight_loader_v2(param, torch.tensor([0.25, 0.75]))

    assert torch.equal(param.data, torch.tensor([0.25, 0.75]))


def test_qkv_parallel_full_scale_vector_loads_all_fused_slots():
    layer = QKVParallelLinear.__new__(QKVParallelLinear)
    layer.tp_size = 1
    param = _per_tensor_scale([-1.0, -2.0, -3.0])

    layer.weight_loader_v2(param, torch.tensor([0.25, 0.5, 0.75]))

    assert torch.equal(param.data, torch.tensor([0.25, 0.5, 0.75]))
