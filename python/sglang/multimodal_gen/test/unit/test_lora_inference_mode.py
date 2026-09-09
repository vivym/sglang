import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.lora.linear import (
    LinearWithLoRA,
    _apply_lora_delta,
    _compute_lora_delta,
    _contracted_lora_gemm_epilogue_enabled,
)


def test_stacked_lora_delta_preserves_projection_order():
    x = torch.tensor([[2.0, 3.0]])
    lora_a = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    lora_b = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    actual = _compute_lora_delta(x, lora_a, lora_b)

    torch.testing.assert_close(actual, torch.tensor([[2.0, 4.0, 9.0, 12.0]]))


def test_dynamic_lora_delta_chunks_output_in_place():
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(2, 7, 5, generator=generator, dtype=torch.bfloat16)
    lora_a = torch.randn(3, 5, generator=generator, dtype=torch.bfloat16)
    lora_b = torch.randn(11, 3, generator=generator, dtype=torch.bfloat16)
    base = torch.randn(2, 7, 11, generator=generator, dtype=torch.bfloat16)
    expected = base + (x @ lora_a.T @ lora_b.T) * 0.75
    actual = base.clone()
    data_ptr = actual.data_ptr()

    result = _apply_lora_delta(
        actual,
        x,
        lora_a,
        lora_b,
        0.75,
        chunk_bytes=3 * 11 * actual.element_size(),
    )

    assert result.data_ptr() == data_ptr
    assert torch.equal(result, expected)


def test_dynamic_lora_delta_keeps_stacked_fallback():
    x = torch.tensor([[2.0, 3.0]])
    lora_a = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    lora_b = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    base = torch.ones(1, 4)

    actual = _apply_lora_delta(base, x, lora_a, lora_b, 0.5)

    torch.testing.assert_close(actual, torch.tensor([[2.0, 3.0, 5.5, 7.0]]))


def test_dynamic_stacked_lora_delta_chunks_in_place_with_sequential_rounding():
    generator = torch.Generator().manual_seed(43)
    x = torch.randn(2, 7, 5, generator=generator, dtype=torch.bfloat16)
    lora_a = torch.randn(3, 4, 5, generator=generator, dtype=torch.bfloat16)
    lora_b = torch.randn(3, 6, 4, generator=generator, dtype=torch.bfloat16)
    base = torch.randn(2, 7, 18, generator=generator, dtype=torch.bfloat16)
    delta = _compute_lora_delta(x, lora_a, lora_b)
    delta.mul_(0.0625)
    delta.mul_(0.75)
    expected = base + delta
    actual = base.clone()
    data_ptr = actual.data_ptr()

    result = _apply_lora_delta(
        actual,
        x,
        lora_a,
        lora_b,
        0.0625,
        post_scale=0.75,
        chunk_bytes=3 * 18 * actual.element_size(),
    )

    assert result.data_ptr() == data_ptr
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", False), ("false", False), ("1", True), ("TRUE", True), ("on", True)],
)
def test_contracted_lora_gemm_epilogue_env(monkeypatch, raw, expected):
    monkeypatch.setenv("SGLANG_LORA_CONTRACTED_GEMM_EPILOGUE", raw)
    assert _contracted_lora_gemm_epilogue_enabled() is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.no_grad()
@pytest.mark.parametrize("stacked", [False, True])
def test_dynamic_lora_delta_dispatches_contracted_epilogue(monkeypatch, stacked):
    monkeypatch.setenv("SGLANG_LORA_CONTRACTED_GEMM_EPILOGUE", "1")
    generator = torch.Generator(device="cuda").manual_seed(44)
    x = torch.randn((17, 7), generator=generator, device="cuda", dtype=torch.bfloat16)
    if stacked:
        lora_a = torch.randn(
            (3, 5, 7), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        lora_b = torch.randn(
            (3, 11, 5), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        base = torch.randn(
            (17, 33), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        hidden = torch.einsum("mi,nri->mnr", x, lora_a)
        expected = base.clone()
        expected_grouped = expected.view(17, 3, 11).permute(1, 0, 2)
        torch.baddbmm(
            expected_grouped,
            hidden.permute(1, 0, 2),
            lora_b.transpose(1, 2),
            beta=1.0,
            alpha=0.046875,
            out=expected_grouped,
        )
    else:
        lora_a = torch.randn(
            (5, 7), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        lora_b = torch.randn(
            (11, 5), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        base = torch.randn(
            (17, 11), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        hidden = x @ lora_a.T
        expected = torch.addmm(base, hidden, lora_b.T, beta=1.0, alpha=0.046875)
    actual = base.clone()

    result = _apply_lora_delta(
        actual,
        x,
        lora_a,
        lora_b,
        0.0625,
        post_scale=0.75,
        chunk_bytes=actual.numel() * actual.element_size(),
    )

    assert result.data_ptr() == actual.data_ptr()
    assert torch.equal(result, expected)


def test_lora_merge_unmerge_handles_inference_base_weight():
    with torch.inference_mode():
        base_layer = nn.Linear(4, 3, bias=False)

    layer = LinearWithLoRA(base_layer, lora_rank=2, lora_alpha=2)
    base_weight = layer.cpu_weight.clone()

    assert layer.base_layer.weight.is_inference()
    assert not base_weight.is_inference()

    lora_a = torch.ones(2, 4)
    lora_b = torch.full((3, 2), 0.5)
    expected_merged = base_weight + lora_b @ lora_a

    with torch.inference_mode(False):
        layer.set_lora_weights(
            lora_a,
            lora_b,
            clear_existing=True,
            merge_weights=True,
        )

    assert layer.merged
    assert not layer.base_layer.weight.is_inference()
    assert torch.allclose(layer.base_layer.weight, expected_merged)

    with torch.inference_mode(False):
        layer.unmerge_lora_weights()

    assert not layer.merged
    assert not layer.base_layer.weight.is_inference()
    assert torch.allclose(layer.base_layer.weight, base_weight)
