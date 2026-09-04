# SPDX-License-Identifier: Apache-2.0
"""Bit-exact data-movement fusions for the MiniMax H3 video VAE FFN."""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.triton.numerics import div_rn_f32, mul_rn_f32
from sglang.multimodal_gen.runtime.platforms import current_platform


@triton.jit
def _mul_reduce_max_f32_kernel(
    product_ptr,
    max_ptr,
    gate_ptr,
    value_ptr,
    rows,
    width: tl.constexpr,
    value_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < width
    linear_offsets = row * width + offsets
    gate = tl.load(gate_ptr + linear_offsets, mask=mask, other=0.0)
    value = tl.load(
        value_ptr + row * value_row_stride + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    product = mul_rn_f32(gate, value)
    tl.store(product_ptr + linear_offsets, product, mask=mask)
    maximum = tl.max(tl.where(mask, tl.abs(product), 0.0), axis=0)
    has_nan = tl.max(tl.where(mask, product != product, 0), axis=0) != 0
    maximum = tl.where(has_nan, float("nan"), maximum)
    tl.store(max_ptr + row, maximum, mask=row < rows)


@triton.jit
def _scale_cast_f16_kernel(
    output_ptr,
    product_ptr,
    scale_ptr,
    numel,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    product = tl.load(product_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr + offsets // width, mask=mask)
    output = div_rn_f32(product, scale)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _restore_scale_add_bias_f32_kernel(
    output_ptr,
    value_ptr,
    scale_ptr,
    bias_ptr,
    numel,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    value = tl.load(value_ptr + offsets, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr + offsets // width, mask=mask)
    bias = tl.load(bias_ptr + offsets % width, mask=mask).to(tl.float32)
    restored = mul_rn_f32(value, scale)
    tl.store(output_ptr + offsets, restored + bias, mask=mask)


def _has_uniform_flattened_row_stride(tensor: torch.Tensor) -> bool:
    if tensor.ndim < 2 or tensor.stride(-1) != 1:
        return False
    row_stride = tensor.stride(-2)
    expected = row_stride
    for dim in range(tensor.ndim - 2, -1, -1):
        if tensor.stride(dim) != expected:
            return False
        expected *= tensor.shape[dim]
    return True


def try_mul_reduce_max_f32_exact(
    gate: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse ``gate * value.float()`` and its rowwise absolute maximum."""
    if (
        not current_platform.is_cuda()
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not gate.is_cuda
        or gate.dtype != torch.float32
        or value.dtype != torch.float16
        or gate.device != value.device
        or gate.shape != value.shape
        or not gate.is_contiguous()
        or not _has_uniform_flattened_row_stride(value)
        or gate.numel() == 0
    ):
        return None

    width = gate.shape[-1]
    block_size = triton.next_power_of_2(width)
    if block_size > 65536:
        return None
    rows = gate.numel() // width
    product = torch.empty_like(gate)
    maximum = torch.empty((*gate.shape[:-1], 1), device=gate.device, dtype=torch.float32)
    _mul_reduce_max_f32_kernel[(rows,)](
        product,
        maximum,
        gate,
        value,
        rows,
        width=width,
        value_row_stride=value.stride(-2),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return product, maximum


def try_scale_cast_f16_exact(
    product: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor | None:
    """Fuse rowwise FP32 division and the following FP16 cast."""
    if (
        not current_platform.is_cuda()
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not product.is_cuda
        or product.dtype != torch.float32
        or scale.dtype != torch.float32
        or product.device != scale.device
        or scale.shape != (*product.shape[:-1], 1)
        or not product.is_contiguous()
        or not scale.is_contiguous()
        or product.numel() == 0
    ):
        return None

    output = torch.empty_like(product, dtype=torch.float16)
    block_size = 1024
    _scale_cast_f16_kernel[(triton.cdiv(product.numel(), block_size),)](
        output,
        product,
        scale,
        product.numel(),
        width=product.shape[-1],
        BLOCK_SIZE=block_size,
    )
    return output


def try_restore_scale_add_bias_f32_exact(
    value: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor | None:
    """Fuse FP16-to-FP32 cast, row scale restore, and FP32 bias add."""
    if (
        not current_platform.is_cuda()
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not value.is_cuda
        or value.dtype != torch.float16
        or scale.dtype != torch.float32
        or bias.dtype != torch.float16
        or value.device != scale.device
        or value.device != bias.device
        or scale.shape != (*value.shape[:-1], 1)
        or bias.shape != (value.shape[-1],)
        or not value.is_contiguous()
        or not scale.is_contiguous()
        or not bias.is_contiguous()
        or value.numel() == 0
    ):
        return None

    output = torch.empty_like(value, dtype=torch.float32)
    block_size = 1024
    _restore_scale_add_bias_f32_kernel[(triton.cdiv(value.numel(), block_size),)](
        output,
        value,
        scale,
        bias,
        value.numel(),
        width=value.shape[-1],
        BLOCK_SIZE=block_size,
    )
    return output


__all__ = [
    "try_mul_reduce_max_f32_exact",
    "try_restore_scale_add_bias_f32_exact",
    "try_scale_cast_f16_exact",
]
