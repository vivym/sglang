# SPDX-License-Identifier: Apache-2.0
"""Bit-exact layout and scale/add operations for dynamic BF16 LoRA."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.common.numerics import (
    mul_rn_f32,
    round_bf16_to_fp32,
)
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _round_lora_bf16_boundary(value):
    rounded = round_bf16_to_fp32(value)
    # ``mul.rn.f32`` may emit the all-ones NaN payload. The generic integer
    # rounding helper overflows on that payload, while an eager BF16 store
    # keeps it as NaN. Preserve NaNs before the next arithmetic boundary.
    return tl.where(value != value, value, rounded)


@triton.jit
def _lora_scale_add_kernel(
    output_ptr,
    delta_ptr,
    scale,
    post_scale,
    numel,
    APPLY_SCALE: tl.constexpr,
    APPLY_POST_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    output = tl.load(output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    delta = tl.load(delta_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if APPLY_SCALE:
        delta = _round_lora_bf16_boundary(mul_rn_f32(delta, scale))
    if APPLY_POST_SCALE:
        delta = _round_lora_bf16_boundary(mul_rn_f32(delta, post_scale))

    # The BF16 store reproduces the final rounding boundary of output.add_(delta).
    tl.store(output_ptr + offsets, output + delta, mask=mask)


def can_use_fused_lora_scale_add(
    output: torch.Tensor,
    delta: torch.Tensor,
) -> bool:
    """Return whether the exact in-place Triton path supports these tensors."""
    return (
        not torch.is_grad_enabled()
        and output.is_cuda
        and delta.is_cuda
        and output.dtype is torch.bfloat16
        and delta.dtype is torch.bfloat16
        and output.device == delta.device
        and output.shape == delta.shape
        and output.is_contiguous()
        and delta.is_contiguous()
        and output.numel() > 0
    )


def can_use_direct_stacked_lora_bmm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Return whether grouped LoRA B can write directly to token-major output."""
    return (
        not torch.is_grad_enabled()
        and hidden.is_cuda
        and weight.is_cuda
        and hidden.dtype is torch.bfloat16
        and weight.dtype is torch.bfloat16
        and hidden.device == weight.device
        and hidden.dim() == 3
        and weight.dim() == 3
        and hidden.shape[0] > 0
        and hidden.shape[1] > 1
        and hidden.shape[2] > 0
        and weight.shape[1] > 0
        and hidden.shape[1] == weight.shape[0]
        and hidden.shape[2] == weight.shape[2]
        and hidden.is_contiguous()
        and weight.is_contiguous()
    )


def can_use_contracted_lora_addmm(
    output: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Return whether a regular LoRA B GEMM can accumulate into ``output``."""
    return (
        not torch.is_grad_enabled()
        and output.is_cuda
        and hidden.is_cuda
        and weight.is_cuda
        and output.dtype is torch.bfloat16
        and hidden.dtype is torch.bfloat16
        and weight.dtype is torch.bfloat16
        and output.device == hidden.device == weight.device
        and output.dim() == hidden.dim() == weight.dim() == 2
        and output.shape[0] > 0
        and output.shape[1] > 0
        and output.shape[0] == hidden.shape[0]
        and output.shape[1] == weight.shape[0]
        and hidden.shape[1] == weight.shape[1]
        and output.is_contiguous()
        and hidden.is_contiguous()
        and weight.is_contiguous()
    )


def can_use_contracted_stacked_lora_baddbmm(
    output: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Return whether stacked LoRA B GEMMs can accumulate into ``output``."""
    return (
        can_use_direct_stacked_lora_bmm(hidden, weight)
        and output.is_cuda
        and output.dtype is torch.bfloat16
        and output.device == hidden.device
        and output.dim() == 2
        and output.shape[0] == hidden.shape[0]
        and output.shape[1] == hidden.shape[1] * weight.shape[1]
        and output.is_contiguous()
    )


def _fake_direct_stacked_lora_bmm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return hidden.new_empty((hidden.shape[0], hidden.shape[1] * weight.shape[1]))


@register_custom_op(
    op_name="direct_stacked_lora_bmm",
    mutates_args=[],
    fake_impl=_fake_direct_stacked_lora_bmm,
)
def direct_stacked_lora_bmm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute grouped LoRA B directly in contiguous ``[tokens, groups*out]``."""
    rows, groups, _ = hidden.shape
    output = hidden.new_empty((rows, groups, weight.shape[1]))
    torch.bmm(
        hidden.permute(1, 0, 2),
        weight.transpose(1, 2),
        out=output.permute(1, 0, 2),
    )
    return output.flatten(start_dim=-2)


@register_custom_op(
    op_name="contracted_lora_addmm_",
    mutates_args=["output"],
)
def contracted_lora_addmm_(
    output: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    post_scale: float,
) -> None:
    """Contract a regular LoRA B GEMM, scales, and residual add into one op."""
    torch.addmm(
        output,
        hidden,
        weight.T,
        beta=1.0,
        alpha=scale * post_scale,
        out=output,
    )


@register_custom_op(
    op_name="contracted_stacked_lora_baddbmm_",
    mutates_args=["output"],
)
def contracted_stacked_lora_baddbmm_(
    output: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    post_scale: float,
) -> None:
    """Contract stacked LoRA B GEMMs, scales, and residual add into one op."""
    groups = hidden.shape[1]
    output_per_group = weight.shape[1]
    output_grouped = output.view(output.shape[0], groups, output_per_group).permute(
        1, 0, 2
    )
    torch.baddbmm(
        output_grouped,
        hidden.permute(1, 0, 2),
        weight.transpose(1, 2),
        beta=1.0,
        alpha=scale * post_scale,
        out=output_grouped,
    )


@register_custom_op(
    op_name="triton_fused_lora_scale_add_",
    mutates_args=["output"],
)
def fused_lora_scale_add_(
    output: torch.Tensor,
    delta: torch.Tensor,
    scale: float,
    post_scale: float,
) -> None:
    """Apply eager-equivalent ``mul_``, optional ``mul_``, and ``add_`` in one pass."""
    numel = output.numel()
    with torch.cuda.device(output.device):
        _lora_scale_add_kernel[(triton.cdiv(numel, 1024),)](
            output,
            delta,
            scale,
            post_scale,
            numel,
            APPLY_SCALE=scale != 1.0,
            APPLY_POST_SCALE=post_scale != 1.0,
            BLOCK=1024,
        )


__all__ = [
    "can_use_contracted_lora_addmm",
    "can_use_contracted_stacked_lora_baddbmm",
    "can_use_direct_stacked_lora_bmm",
    "can_use_fused_lora_scale_add",
    "contracted_lora_addmm_",
    "contracted_stacked_lora_baddbmm_",
    "direct_stacked_lora_bmm",
    "fused_lora_scale_add_",
]
