# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.base_module import (
    FeedForward,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp16_gated_ffn_preserves_large_product_cancellation(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_VAE_FFN_FP32_ACT", "1")
    module = FeedForward(2, mult=1, use_gated=True, bias=True).cuda().half()
    with torch.no_grad():
        module.w1.weight.zero_()
        module.w1.bias.fill_(512.0)
        module.w2.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device="cuda"))
        module.w2.bias.fill_(7.0)

    inputs = torch.zeros((1, 2), device="cuda", dtype=torch.float16)
    gate_and_value = module.w1(inputs)
    gate, value = gate_and_value.chunk(2, dim=-1)
    product = torch.nn.functional.silu(gate.float()) * value.float()
    assert product.max().item() == 262144.0
    expected = torch.nn.functional.linear(
        product, module.w2.weight.float(), module.w2.bias.float()
    ).half()

    actual = module(inputs)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
