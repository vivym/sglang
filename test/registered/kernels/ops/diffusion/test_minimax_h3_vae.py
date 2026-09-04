import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.triton.minimax_h3_vae import (
    try_mul_reduce_max_f32_exact,
    try_restore_scale_add_bias_f32_exact,
    try_scale_cast_f16_exact,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.no_grad()
@pytest.mark.parametrize("strided_value", [False, True])
def test_mul_reduce_max_matches_released_vae_shape(strided_value):
    torch.cuda.manual_seed(0)
    shape = (1, 1797, 8192)
    gate = torch.randn(shape, device="cuda", dtype=torch.float32)
    if strided_value:
        value = torch.randn(
            (*shape[:-1], shape[-1] * 2), device="cuda", dtype=torch.float16
        ).chunk(2, dim=-1)[1]
        assert value.stride(-2) == 16384
    else:
        value = torch.randn(shape, device="cuda", dtype=torch.float16)

    actual = try_mul_reduce_max_f32_exact(gate, value)
    expected_product = gate.mul(value.float())
    expected_maximum = expected_product.detach().abs().amax(dim=-1, keepdim=True)

    assert actual is not None
    actual_product, actual_maximum = actual
    assert actual_product.dtype == torch.float32
    assert actual_maximum.shape == (*shape[:-1], 1)
    assert torch.equal(actual_product, expected_product)
    assert torch.equal(actual_maximum, expected_maximum)


@torch.no_grad()
def test_mul_reduce_max_propagates_nonfinite_values():
    gate = torch.ones((3, 16), device="cuda", dtype=torch.float32)
    value = torch.ones_like(gate, dtype=torch.float16)
    gate[0, 3] = torch.nan
    gate[1, 5] = torch.inf
    gate[2, 7] = -torch.inf

    actual = try_mul_reduce_max_f32_exact(gate, value)
    expected_product = gate.mul(value.float())
    expected_maximum = expected_product.detach().abs().amax(dim=-1, keepdim=True)

    assert actual is not None
    actual_product, actual_maximum = actual
    assert torch.equal(torch.isnan(actual_product), torch.isnan(expected_product))
    assert torch.equal(torch.isinf(actual_product), torch.isinf(expected_product))
    assert torch.equal(torch.isnan(actual_maximum), torch.isnan(expected_maximum))
    assert torch.equal(torch.isinf(actual_maximum), torch.isinf(expected_maximum))
    finite = torch.isfinite(expected_product)
    assert torch.equal(actual_product[finite], expected_product[finite])


@torch.no_grad()
@pytest.mark.parametrize("scale_value", [1.0, 2.0, 4.0])
def test_scale_cast_matches_eager_at_fp16_rounding_boundaries(scale_value):
    below_one = torch.nextafter(
        torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda")
    )
    above_one = torch.nextafter(
        torch.tensor(1.0, device="cuda"), torch.tensor(2.0, device="cuda")
    )
    half_ulp = torch.tensor(2.0**-11, device="cuda")
    values = torch.stack(
        (
            below_one,
            torch.tensor(1.0, device="cuda"),
            above_one,
            1.0 + half_ulp,
            1.0 + half_ulp + 2.0**-23,
            -(1.0 + half_ulp),
            torch.tensor(65504.0, device="cuda"),
            torch.tensor(2.0**-24, device="cuda"),
        )
    )
    product = values.repeat(3, 1024)
    scale = torch.full(
        (product.shape[0], 1), scale_value, device="cuda", dtype=torch.float32
    )

    actual = try_scale_cast_f16_exact(product, scale)
    expected = product.div(scale).to(torch.float16)

    assert actual is not None
    assert actual.dtype == torch.float16
    assert torch.equal(actual, expected)


@torch.no_grad()
def test_restore_scale_add_bias_matches_eager():
    torch.cuda.manual_seed(1)
    value = torch.randn((2, 17, 8192), device="cuda", dtype=torch.float16)
    scale = torch.tensor(
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0] * 5,
        device="cuda",
        dtype=torch.float32,
    )[: value.shape[0] * value.shape[1]].reshape(*value.shape[:-1], 1)
    bias = torch.randn(value.shape[-1], device="cuda", dtype=torch.float16)

    actual = try_restore_scale_add_bias_f32_exact(value, scale, bias)
    expected = value.float()
    expected.mul_(scale)
    expected.add_(bias.float())

    assert actual is not None
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


@torch.no_grad()
def test_fusions_reject_unsupported_inputs():
    gate = torch.randn((2, 3, 16), device="cuda", dtype=torch.float32)
    value = torch.randn_like(gate, dtype=torch.float16)
    product = torch.randn_like(gate)
    scale = torch.ones((2, 3, 1), device="cuda", dtype=torch.float32)
    bias = torch.randn(16, device="cuda", dtype=torch.float16)
    nonuniform_value = value.transpose(0, 1)
    noncontiguous_product = torch.randn(
        (2, 3, 32), device="cuda", dtype=torch.float32
    )[..., ::2]
    noncontiguous_output = torch.randn(
        (2, 3, 32), device="cuda", dtype=torch.float16
    )[..., ::2]

    assert try_mul_reduce_max_f32_exact(gate.cpu(), value.cpu()) is None
    assert try_mul_reduce_max_f32_exact(gate, value.bfloat16()) is None
    assert (
        try_mul_reduce_max_f32_exact(gate.transpose(0, 1).contiguous(), nonuniform_value)
        is None
    )
    assert try_scale_cast_f16_exact(product.bfloat16(), scale) is None
    assert try_scale_cast_f16_exact(noncontiguous_product, scale) is None
    assert try_restore_scale_add_bias_f32_exact(value, scale, bias.float()) is None
    assert (
        try_restore_scale_add_bias_f32_exact(noncontiguous_output, scale, bias)
        is None
    )


def test_fusions_reject_grad_enabled_execution():
    gate = torch.randn((2, 3, 16), device="cuda", dtype=torch.float32)
    value = torch.randn_like(gate, dtype=torch.float16)
    product = torch.randn_like(gate)
    scale = torch.ones((2, 3, 1), device="cuda", dtype=torch.float32)
    bias = torch.randn(16, device="cuda", dtype=torch.float16)

    assert torch.is_grad_enabled()
    assert try_mul_reduce_max_f32_exact(gate, value) is None
    assert try_scale_cast_f16_exact(product, scale) is None
    assert try_restore_scale_add_bias_f32_exact(value, scale, bias) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
