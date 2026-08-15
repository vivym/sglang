# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.base_module import (
    FeedForward,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp16_gated_ffn_preserves_large_product_cancellation(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_VAE_FFN_FP32_ACT", "1")
    monkeypatch.setenv("MINIMAX_H3_VAE_FFN_FP32_PRODUCT_SCALE", "1")
    module = FeedForward(2, mult=1, use_gated=True, bias=True).cuda().half()
    with torch.no_grad():
        module.w1.weight.zero_()
        module.w1.bias.fill_(512.0)
        module.w2.weight.copy_(
            torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device="cuda")
        )
        module.w2.bias.fill_(7.0)

    inputs = torch.zeros((1, 2), device="cuda", dtype=torch.float16)
    gate_and_value = module.w1(inputs)
    gate, value = gate_and_value.chunk(2, dim=-1)
    product = torch.nn.functional.silu(gate.float()) * value.float()
    assert product.max().item() == 262144.0
    expected = torch.nn.functional.linear(
        product, module.w2.weight.float(), module.w2.bias.float()
    )

    actual = module(inputs)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp16_gated_ffn_keeps_large_restored_output_in_fp32(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_VAE_FFN_FP32_ACT", "1")
    monkeypatch.setenv("MINIMAX_H3_VAE_FFN_FP32_PRODUCT_SCALE", "1")
    module = FeedForward(2, mult=1, use_gated=True, bias=True).cuda().half()
    with torch.no_grad():
        module.w1.weight.zero_()
        module.w1.bias.fill_(512.0)
        module.w2.weight.fill_(1.0)
        module.w2.bias.zero_()

    inputs = torch.zeros((1, 2), device="cuda", dtype=torch.float16)
    gate_and_value = module.w1(inputs)
    gate, value = gate_and_value.chunk(2, dim=-1)
    product = torch.nn.functional.silu(gate.float()) * value.float()
    expected = torch.nn.functional.linear(product, module.w2.weight.float())
    assert expected.max().item() > torch.finfo(torch.float16).max

    actual = module(inputs)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp16_gated_ffn_scales_for_w2_output_bound(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_VAE_FFN_FP32_ACT", "1")
    monkeypatch.setenv("MINIMAX_H3_VAE_FFN_FP32_PRODUCT_SCALE", "1")
    monkeypatch.setenv("MINIMAX_H3_VAE_FFN_OUTPUT_BOUND_SCALE", "1")
    module = FeedForward(2, mult=1, use_gated=True, bias=True).cuda().half()
    with torch.no_grad():
        module.w1.weight.zero_()
        module.w1.bias.fill_(64.0)
        module.w2.weight.fill_(20.0)
        module.w2.bias.zero_()

    inputs = torch.zeros((1, 2), device="cuda", dtype=torch.float16)
    gate_and_value = module.w1(inputs)
    gate, value = gate_and_value.chunk(2, dim=-1)
    product = torch.nn.functional.silu(gate.float()) * value.float()
    expected = torch.nn.functional.linear(product, module.w2.weight.float())
    assert expected.max().item() > torch.finfo(torch.float16).max

    actual = module(inputs)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
