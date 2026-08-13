# SPDX-License-Identifier: Apache-2.0
# Transformer building blocks for the MiniMax H3 visual VAE ViT decoder.
import math
from typing import Optional

import torch
import torch.nn as nn
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph

from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding,
)
from sglang.kernels.ops.diffusion.triton.scale_shift import (
    try_fused_scaled_residual_add_exact,
)

from .attention import Attention
from .vit_utils import _env_flag, _vit_torch_compile_kwargs

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _vit_norm_input(module, hidden_states):
    if _env_flag("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1"):
        return hidden_states.float()
    return hidden_states.to(getattr(module.weight, "dtype", hidden_states.dtype))


def _scaled_residual_add(residual, x, scale):
    fused = try_fused_scaled_residual_add_exact(residual, x, scale)
    return residual + x * scale if fused is None else fused


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        activation_fn: str = "silu",
        bias: bool = True,
        use_gated: bool = True,
        glu_balanced: bool = False,
    ):
        super().__init__()
        ratio = 2 / 3 if (use_gated and glu_balanced) else 1
        inner_dim = round(dim * mult * ratio)
        dim_out = dim_out if dim_out is not None else dim
        self.use_gated = use_gated

        if use_gated:
            self.w1 = nn.Linear(dim, inner_dim * 2, bias=bias)
        else:
            self.w1 = nn.Linear(dim, inner_dim, bias=bias)

        if activation_fn == "silu":
            self.act_fn = nn.SiLU()
        elif activation_fn == "gelu":
            self.act_fn = nn.GELU()
        elif activation_fn == "gelu-approximate":
            self.act_fn = nn.GELU(approximate="tanh")
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.w2 = nn.Linear(inner_dim, dim_out, bias=bias)
        self._compile_forward_enabled = _env_flag(
            "MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE", "0"
        )
        self._compile_forward_fatal = _env_flag(
            "MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE_FATAL", "0"
        )
        self._compiled_forward = None

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.w1(hidden_states)
        # DEBUG(fast-h3 INT8 排障, 临时)
        n_nan, n_inf = int(torch.isnan(hidden_states).sum()), int(torch.isinf(hidden_states).sum())
        if n_nan or n_inf:
            with open("/tmp/h3-nan-debug.log", "a") as f:
                f.write(f"[ffn-w1] NaN={n_nan} inf={n_inf} max_abs={float(hidden_states.abs().max()):.4f}\n")

        if self.use_gated:
            if (
                isinstance(self.act_fn, nn.SiLU)
                and hidden_states.is_cuda
                and hidden_states.dtype in (torch.float16, torch.bfloat16)
                and hidden_states.is_contiguous()
                and hidden_states.shape[-1] % 32 == 0
                and not _env_flag("MINIMAX_H3_VAE_FFN_FP32_ACT", "1")
            ):
                hidden_states = silu_and_mul_with_activation_rounding(hidden_states)
            else:
                gate, hidden_states = hidden_states.chunk(2, dim=-1)
                # fast-h3 修复: fp16/bf16 下 silu 的 exp 溢出（|x|>11/88）产生 NaN
                # （INT8 latents 偶发临界值触发）→ 激活在 fp32 计算
                if _env_flag("MINIMAX_H3_VAE_FFN_FP32_ACT", "1"):
                    gate = self.act_fn(gate.float()).to(hidden_states.dtype)
                else:
                    gate = self.act_fn(gate)
                hidden_states = gate.mul_(hidden_states)
        else:
            hidden_states = self.act_fn(hidden_states)
        # DEBUG(fast-h3 INT8 排障, 临时)
        n_nan, n_inf = int(torch.isnan(hidden_states).sum()), int(torch.isinf(hidden_states).sum())
        if n_nan or n_inf:
            with open("/tmp/h3-nan-debug.log", "a") as f:
                f.write(f"[ffn-act] NaN={n_nan} inf={n_inf} max_abs={float(hidden_states.abs().max()):.4f}\n")

        hidden_states = self.w2(hidden_states)
        # DEBUG(fast-h3 INT8 排障, 临时)
        n_nan, n_inf = int(torch.isnan(hidden_states).sum()), int(torch.isinf(hidden_states).sum())
        if n_nan or n_inf:
            with open("/tmp/h3-nan-debug.log", "a") as f:
                f.write(f"[ffn-w2] NaN={n_nan} inf={n_inf} max_abs={float(hidden_states.abs().max()):.4f}\n")
        return hidden_states

    def _get_forward_impl(self):
        if not self._compile_forward_enabled:
            return self._forward_impl
        if self._compiled_forward is not None:
            return self._compiled_forward
        if not hasattr(torch, "compile"):
            message = (
                "torch.compile is unavailable; falling back to eager ViT FeedForward"
            )
            if self._compile_forward_fatal:
                raise RuntimeError(message)
            logger.warning(f"[ViTFeedForward] {message}")
            self._compile_forward_enabled = False
            return self._forward_impl

        kwargs = _vit_torch_compile_kwargs(
            "MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE"
        )
        try:
            self._compiled_forward = torch.compile(self._forward_impl, **kwargs)
            logger.info(f"[ViTFeedForward] torch.compile enabled kwargs={kwargs}")
        except Exception as exc:
            if self._compile_forward_fatal:
                raise
            logger.warning(
                f"[ViTFeedForward] torch.compile setup failed: {type(exc).__name__}: {exc}; "
                "falling back to eager"
            )
            self._compile_forward_enabled = False
            self._compiled_forward = None
            return self._forward_impl
        return self._compiled_forward

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forward_impl = self._get_forward_impl()
        try:
            return forward_impl(hidden_states)
        except Exception as exc:
            if (
                self._compile_forward_enabled
                and self._compiled_forward is not None
                and forward_impl is self._compiled_forward
                and not self._compile_forward_fatal
            ):
                logger.warning(
                    f"[ViTFeedForward] compiled forward failed: {type(exc).__name__}: {exc}; "
                    "disabling compile and retrying eager"
                )
                self._compile_forward_enabled = False
                self._compiled_forward = None
                return self._forward_impl(hidden_states)
            raise


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=10000, n_dim=3, use_angle=False):
        super().__init__()
        self.dim = dim
        self.n_dim = n_dim

        if dim % (2 * n_dim) != 0:
            raise ValueError(
                f"head_dim {dim} must be divisible by 2 * n_dim {2 * n_dim}"
            )

        if use_angle:
            self.angle_scale = 2.0 * math.pi
        else:
            self.angle_scale = 1.0

        inv_freq = 1 / rotary_base ** torch.arange(
            0, 1, 2 * n_dim / dim, dtype=torch.float32
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, img_ids):
        B, N, D = img_ids.shape
        if D != self.n_dim:
            raise ValueError(f"Expected {self.n_dim} dimensions, got {D}")

        with torch.autocast("cuda", enabled=False):
            angles = (
                self.angle_scale
                * img_ids[:, :, :, None]
                * self.inv_freq.to(img_ids.device)[None, None, None, :]
            )
            angles = angles.flatten(2, 3)
            angles = angles.tile(2)
            angles = angles.unsqueeze(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        return cos.to(dtype=img_ids.dtype), sin.to(dtype=img_ids.dtype)


@maybe_allow_in_graph
class TransformerBlock(nn.Module):
    def __init__(
        self,
        heads: int,
        dim_head: int,
        embed_dim: Optional[int] = None,
        ffn_glu_balanced: bool = False,
        norm_type: str = "layer_norm",
        norm_affine: bool = True,
        qk_norm_type: str = "rms_norm",
        qk_norm_affine: bool = False,
        ffn_activation_fn: str = "silu",
        ffn_use_gated: bool = True,
        use_scale: bool = True,
        bias: bool = True,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        dim = embed_dim if embed_dim is not None else dim_head * heads
        self.use_scale = use_scale

        if norm_type == "layer_norm":
            norm_class = nn.LayerNorm
        elif norm_type == "rms_norm":
            norm_class = nn.RMSNorm
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.norm1 = norm_class(
            dim,
            elementwise_affine=norm_affine,
            eps=eps,
        )
        self.attn = Attention(
            heads=heads,
            dim_head=dim_head,
            embed_dim=dim,
            qk_norm_type=qk_norm_type,
            qk_norm_affine=qk_norm_affine,
            bias=bias,
            eps=eps,
            **kwargs,
        )
        if use_scale:
            self.scale1 = nn.Parameter(torch.zeros(dim))

        self.norm2 = norm_class(
            dim,
            elementwise_affine=norm_affine,
            eps=eps,
        )
        self.ff = FeedForward(
            dim=dim,
            activation_fn=ffn_activation_fn,
            bias=bias,
            use_gated=ffn_use_gated,
            glu_balanced=ffn_glu_balanced,
        )
        if use_scale:
            self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        rotary_pos_emb: Optional[torch.FloatTensor] = None,
        pack_info: dict = {},
    ):
        norm_hidden_states = self.norm1(_vit_norm_input(self.norm1, hidden_states)).to(
            hidden_states.dtype
        )
        attn_output = self.attn(norm_hidden_states, rotary_pos_emb, pack_info)
        # DEBUG(fast-h3 INT8 排障, 临时): attention/FFN 输出 NaN 检查
        if torch.isnan(attn_output).any():
            with open("/tmp/h3-nan-debug.log", "a") as f:
                f.write(f"[block-attn-out] NaN={int(torch.isnan(attn_output).sum())}\n")
        if self.use_scale:
            hidden_states = _scaled_residual_add(
                hidden_states, attn_output, self.scale1
            )
        else:
            hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(_vit_norm_input(self.norm2, hidden_states)).to(
            hidden_states.dtype
        )
        ff_output = self.ff(norm_hidden_states)
        if torch.isnan(ff_output).any():
            with open("/tmp/h3-nan-debug.log", "a") as f:
                f.write(f"[block-ff-out] NaN={int(torch.isnan(ff_output).sum())}\n")
        if self.use_scale:
            hidden_states = _scaled_residual_add(hidden_states, ff_output, self.scale2)
        else:
            hidden_states = hidden_states + ff_output

        return hidden_states
