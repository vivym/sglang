# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0


import hashlib
import importlib.util
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import sageattention
from sageattention import sageattn

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (  # FlashAttentionMetadata,
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

SAGE_SM80_VARIANT_ENV = "SGLANG_H3_SAGE_SM80_VARIANT"
SAGE_SM80_ROOT_ENV = "SGLANG_H3_SAGE_SM80_ROOT"
SAGE_SM80_QATTN_SHA256_ENV = "SGLANG_H3_SAGE_SM80_QATTN_SHA256"
SAGE_SM80_RESEARCH_VARIANTS = frozenset(
    {
        "cuda_fp32",
        "cuda_per_warp_fp32",
        "cuda_per_warp_fp32_precombined_skip_noop",
        "cuda_fp16",
        "cuda_fp16_row_pipeline_precombined_scale",
    }
)


@dataclass(frozen=True, slots=True)
class SageSM80ResearchConfig:
    variant: str
    root: Path
    package_path: Path
    binding_path: Path
    qattn_extension: Path
    qattn_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _persistent_root(raw: str) -> Path:
    if not raw:
        raise RuntimeError(f"{SAGE_SM80_ROOT_ENV} is required")
    root = Path(raw).expanduser()
    if not root.is_absolute():
        raise RuntimeError(f"{SAGE_SM80_ROOT_ENV} must be absolute")
    root = root.resolve()
    if root == Path("/tmp") or Path("/tmp") in root.parents:
        raise RuntimeError(f"{SAGE_SM80_ROOT_ENV} must use persistent storage")
    if not (root / "sageattention/sm80.py").is_file():
        raise RuntimeError(f"{SAGE_SM80_ROOT_ENV} is not a SageAttention checkout")
    return root


def resolve_sage_sm80_research_config(
    environment: Mapping[str, str],
) -> SageSM80ResearchConfig | None:
    variant = environment.get(SAGE_SM80_VARIANT_ENV, "").strip()
    companion_values = {
        SAGE_SM80_ROOT_ENV: environment.get(SAGE_SM80_ROOT_ENV, "").strip(),
        SAGE_SM80_QATTN_SHA256_ENV: environment.get(
            SAGE_SM80_QATTN_SHA256_ENV, ""
        ).strip(),
    }
    if not variant:
        stale = [name for name, value in companion_values.items() if value]
        if stale:
            raise RuntimeError(
                f"{SAGE_SM80_VARIANT_ENV} is required when setting " + ", ".join(stale)
            )
        return None
    if variant not in SAGE_SM80_RESEARCH_VARIANTS:
        allowed = ", ".join(sorted(SAGE_SM80_RESEARCH_VARIANTS))
        raise RuntimeError(
            f"unsupported {SAGE_SM80_VARIANT_ENV}={variant!r}; expected one of {allowed}"
        )

    root = _persistent_root(companion_values[SAGE_SM80_ROOT_ENV])
    expected_sha256 = companion_values[SAGE_SM80_QATTN_SHA256_ENV]
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise RuntimeError(
            f"{SAGE_SM80_QATTN_SHA256_ENV} must be a lowercase SHA256 digest"
        )

    from sageattention import sm80 as sm80_module

    package_path = Path(sageattention.__file__).resolve()
    binding_path = Path(sm80_module.__file__).resolve()
    for label, path in (("package", package_path), ("binding", binding_path)):
        if root not in path.parents:
            raise RuntimeError(
                f"SageAttention {label} imported outside the required root: {path}"
            )
    if variant not in sm80_module.VARIANTS:
        raise RuntimeError(f"SageAttention binding is missing variant {variant!r}")

    extension_spec = importlib.util.find_spec("sageattention._qattn_sm80")
    if extension_spec is None or extension_spec.origin is None:
        raise RuntimeError("SageAttention SM80 qattn extension is unavailable")
    extension_path = Path(extension_spec.origin).resolve()
    if root not in extension_path.parents:
        raise RuntimeError(
            "SageAttention qattn extension imported outside the required root: "
            f"{extension_path}"
        )
    actual_sha256 = _sha256(extension_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "SageAttention qattn SHA256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    return SageSM80ResearchConfig(
        variant=variant,
        root=root,
        package_path=package_path,
        binding_path=binding_path,
        qattn_extension=extension_path,
        qattn_sha256=actual_sha256,
    )


@lru_cache(maxsize=1)
def current_sage_sm80_research_config() -> SageSM80ResearchConfig | None:
    config = resolve_sage_sm80_research_config(os.environ)
    if config is not None:
        logger.warning(
            "Enabled fail-closed H3 SageAttention SM80 research variant %s "
            "with qattn SHA256 %s",
            config.variant,
            config.qattn_sha256,
        )
    return config


def _trailing_padding_used_len(
    *,
    total_tokens: int,
    max_seqlen: int,
    bounds: tuple[int, ...],
) -> int | None:
    """Return live token count for H3-style [0, used, total] trailing padding."""
    if len(bounds) != 3:
        return None
    start, used, total = bounds
    if start != 0 or used >= total or total != total_tokens or used != max_seqlen:
        return None
    return used


class SageAttentionBackend(AttentionBackend):

    @classmethod
    def supports_ring_rotation(cls) -> bool:
        return True

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_ATTN

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        self._sm80_research = current_sage_sm80_research_config()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        if self._sm80_research is None:
            output = sageattn(
                query,
                key,
                value,
                # since input is (batch_size, seq_len, head_num, head_dim)
                tensor_layout="NHD",
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
                return_lse=return_softmax_lse,
            )
        else:
            if self.causal:
                raise RuntimeError("H3 SageAttention SM80 research mode is non-causal only")
            if return_softmax_lse:
                raise RuntimeError(
                    "H3 SageAttention SM80 research mode does not expose softmax LSE"
                )
            from sageattention.sm80 import sm80_attention

            output, trace = sm80_attention(
                query,
                key,
                value,
                variant=self._sm80_research.variant,
                sm_scale=self.softmax_scale,
                return_trace=True,
                # Per-call finite reductions synchronize every layer. The paired
                # harness checks complete outputs outside the measured interval.
                check_finite=False,
            )
            if (
                trace["requested_variant"] != self._sm80_research.variant
                or trace["fallback_allowed"] is not False
            ):
                raise RuntimeError(f"unexpected SageAttention dispatch trace: {trace}")
        if return_softmax_lse:
            output, softmax_lse = output
            return output, softmax_lse
        return output

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(x) for x in cu_seqlens.tolist())
        )
        return self._sage_packed(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            bounds=bounds,
            max_seqlen=max_seqlen,
        )

    def _sage_packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        bounds: tuple[int, ...],
        max_seqlen: int,
    ) -> torch.Tensor:
        # MiniMax-H3 packs one live document as bounds=(0, used, total):
        # [0, used) are real tokens; [used, total) is 64-aligned tail padding.
        used = _trailing_padding_used_len(
            total_tokens=query.shape[0],
            max_seqlen=max_seqlen,
            bounds=bounds,
        )
        if used is not None:
            live_out = self.forward(
                query[:used].unsqueeze(0),
                key[:used].unsqueeze(0),
                value[:used].unsqueeze(0),
                None,
            )[0]
            if used == query.shape[0]:
                return live_out
            # Keep padded tail at zero so downstream masked rows stay inactive.
            output = torch.zeros_like(query)
            output[:used] = live_out
            return output

        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            output[start:stop] = self.forward(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
                None,
            )[0]
        return output
