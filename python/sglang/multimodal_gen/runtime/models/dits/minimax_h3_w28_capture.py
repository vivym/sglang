# SPDX-License-Identifier: Apache-2.0
"""Opt-in, compact W28 block-gating capture for real MiniMax H3 activations."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import math
import os
import re
import shutil
import socket
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


ENABLE_ENV = "MINIMAX_H3_W28_CAPTURE"
ANALYZE_ENV = "MINIMAX_H3_W28_CAPTURE_ANALYZE"
PATH_ENV = "MINIMAX_H3_W28_CAPTURE_PATH"
LAYERS_ENV = "MINIMAX_H3_W28_CAPTURE_LAYERS"
STEPS_ENV = "MINIMAX_H3_W28_CAPTURE_STEPS"
HEADS_ENV = "MINIMAX_H3_W28_CAPTURE_HEADS"
Q_BLOCKS_ENV = "MINIMAX_H3_W28_CAPTURE_Q_BLOCKS"
MAX_SAMPLES_ENV = "MINIMAX_H3_W28_CAPTURE_MAX_SAMPLES"
RUN_ID_ENV = "MINIMAX_H3_W28_CAPTURE_RUN_ID"
SAGEATTENTION_ROOT_ENV = "MINIMAX_H3_W28_SAGEATTENTION_ROOT"
QKV_SNAPSHOT_PATH_ENV = "MINIMAX_H3_W28_QKV_SNAPSHOT_PATH"
QKV_SNAPSHOT_DIR_ENV = "MINIMAX_H3_W28_QKV_SNAPSHOT_DIR"
QKV_SNAPSHOT_MAX_BYTES_ENV = "MINIMAX_H3_W28_QKV_SNAPSHOT_MAX_BYTES"
FULL_QKV_SNAPSHOT_PATH_ENV = "MINIMAX_H3_W29_FULL_QKV_SNAPSHOT_PATH"

BLOCK_Q = 128
WARP_Q = 32
BLOCK_K = 64
GATE_K = 16
CUMULATIVE_BUDGET = 2.0e-4
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_MAIN_LAYER_PATTERN = re.compile(r"^blocks\.(\d+)\.attn$")
_WRITE_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()


@dataclass(frozen=True)
class CaptureConfig:
    output_path: Path
    layers: frozenset[int]
    steps: frozenset[int] | None
    heads: frozenset[int]
    q_blocks: tuple[str | int, ...]
    max_samples: int
    run_id: str
    sageattention_root: Path
    qkv_snapshot_path: Path | None
    qkv_snapshot_dir: Path | None
    full_qkv_snapshot_path: Path | None
    qkv_snapshot_max_bytes: int
    analyze_samples: bool


@dataclass
class _CaptureState:
    config: CaptureConfig
    sampled: set[tuple[int, int, int, int]]
    snapshot_bytes: int = 0
    full_snapshot_written: bool = False


_CAPTURE_STATE: _CaptureState | None = None


def capture_enabled(environment: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environment is None else environment
    return env.get(ENABLE_ENV, "").strip().lower() in _TRUE_VALUES


def _analysis_enabled(environment: Mapping[str, str]) -> bool:
    return environment.get(ANALYZE_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def is_main_dit_attention_prefix(attention_prefix: str) -> bool:
    return _MAIN_LAYER_PATTERN.fullmatch(attention_prefix) is not None


def _parse_int_set(value: str, *, name: str, allow_all: bool = False):
    value = value.strip()
    if allow_all and value == "*":
        return None
    try:
        parsed = frozenset(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not parsed or min(parsed) < 0:
        raise ValueError(f"{name} must contain non-negative integers")
    return parsed


def _parse_q_blocks(value: str) -> tuple[str | int, ...]:
    parsed: list[str | int] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if item in {"first", "middle", "last"}:
            parsed.append(item)
            continue
        try:
            block = int(item)
        except ValueError as exc:
            raise ValueError(
                f"{Q_BLOCKS_ENV} entries must be first, middle, last, or block indices"
            ) from exc
        if block < 0:
            raise ValueError(f"{Q_BLOCKS_ENV} block indices must be non-negative")
        parsed.append(block)
    if not parsed:
        raise ValueError(f"{Q_BLOCKS_ENV} must not be empty")
    return tuple(dict.fromkeys(parsed))


def parse_capture_config(
    environment: Mapping[str, str] | None = None,
) -> CaptureConfig | None:
    env = os.environ if environment is None else environment
    if not capture_enabled(env):
        return None
    raw_path = env.get(PATH_ENV, "").strip()
    if not raw_path:
        raise ValueError(f"{PATH_ENV} is required when {ENABLE_ENV}=1")
    output_path = Path(raw_path).expanduser()
    if not output_path.is_absolute():
        raise ValueError(f"{PATH_ENV} must be an absolute path")
    output_path = output_path.resolve()
    if output_path == Path("/tmp") or Path("/tmp") in output_path.parents:
        raise ValueError(f"{PATH_ENV} must use persistent storage, not /tmp")
    if output_path.suffix != ".jsonl":
        raise ValueError(f"{PATH_ENV} must end in .jsonl")

    layers = _parse_int_set(env.get(LAYERS_ENV, "0,24,49"), name=LAYERS_ENV)
    steps = _parse_int_set(env.get(STEPS_ENV, "0"), name=STEPS_ENV, allow_all=True)
    heads = _parse_int_set(env.get(HEADS_ENV, "0,27,55"), name=HEADS_ENV)
    try:
        max_samples = int(env.get(MAX_SAMPLES_ENV, "256"))
    except ValueError as exc:
        raise ValueError(f"{MAX_SAMPLES_ENV} must be an integer") from exc
    if not 1 <= max_samples <= 4096:
        raise ValueError(f"{MAX_SAMPLES_ENV} must be between 1 and 4096")
    qkv_snapshot_path = _qkv_snapshot_path(env, max_samples=max_samples)
    qkv_snapshot_dir = _qkv_snapshot_dir(env)
    full_qkv_snapshot_path = _full_qkv_snapshot_path(env)
    if qkv_snapshot_path is not None and qkv_snapshot_dir is not None:
        raise ValueError(
            f"{QKV_SNAPSHOT_PATH_ENV} and {QKV_SNAPSHOT_DIR_ENV} are mutually exclusive"
        )
    try:
        qkv_snapshot_max_bytes = int(
            env.get(QKV_SNAPSHOT_MAX_BYTES_ENV, str(4 * 1024**3))
        )
    except ValueError as exc:
        raise ValueError(f"{QKV_SNAPSHOT_MAX_BYTES_ENV} must be an integer") from exc
    if not 64 * 1024**2 <= qkv_snapshot_max_bytes <= 64 * 1024**3:
        raise ValueError(
            f"{QKV_SNAPSHOT_MAX_BYTES_ENV} must be between 64 MiB and 64 GiB"
        )
    return CaptureConfig(
        output_path=output_path,
        layers=layers,
        steps=steps,
        heads=heads,
        q_blocks=_parse_q_blocks(env.get(Q_BLOCKS_ENV, "first,middle,last")),
        max_samples=max_samples,
        run_id=env.get(RUN_ID_ENV, "").strip() or str(uuid.uuid4()),
        sageattention_root=_persistent_sageattention_root(env),
        qkv_snapshot_path=qkv_snapshot_path,
        qkv_snapshot_dir=qkv_snapshot_dir,
        full_qkv_snapshot_path=full_qkv_snapshot_path,
        qkv_snapshot_max_bytes=qkv_snapshot_max_bytes,
        analyze_samples=_analysis_enabled(env),
    )


def _persistent_sageattention_root(environment: Mapping[str, str]) -> Path:
    raw_root = environment.get(SAGEATTENTION_ROOT_ENV, "").strip()
    if not raw_root:
        raise ValueError(f"{SAGEATTENTION_ROOT_ENV} is required when {ENABLE_ENV}=1")
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        raise ValueError(f"{SAGEATTENTION_ROOT_ENV} must be an absolute path")
    root = root.resolve()
    if root == Path("/tmp") or Path("/tmp") in root.parents:
        raise ValueError(f"{SAGEATTENTION_ROOT_ENV} must not use /tmp")
    if not (root / "sageattention/sm80_compile.py").is_file():
        raise ValueError(f"{SAGEATTENTION_ROOT_ENV} is not a SageAttention checkout")
    return root


def _qkv_snapshot_path(
    environment: Mapping[str, str], *, max_samples: int
) -> Path | None:
    raw_path = environment.get(QKV_SNAPSHOT_PATH_ENV, "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{QKV_SNAPSHOT_PATH_ENV} must be an absolute path")
    path = path.resolve()
    if path == Path("/tmp") or Path("/tmp") in path.parents:
        raise ValueError(f"{QKV_SNAPSHOT_PATH_ENV} must use persistent storage, not /tmp")
    if path.suffix != ".safetensors":
        raise ValueError(f"{QKV_SNAPSHOT_PATH_ENV} must end in .safetensors")
    if max_samples != 1:
        raise ValueError(f"{QKV_SNAPSHOT_PATH_ENV} requires max_samples=1")
    return path


def _qkv_snapshot_dir(environment: Mapping[str, str]) -> Path | None:
    raw_path = environment.get(QKV_SNAPSHOT_DIR_ENV, "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{QKV_SNAPSHOT_DIR_ENV} must be an absolute path")
    path = path.resolve()
    if path == Path("/tmp") or Path("/tmp") in path.parents:
        raise ValueError(f"{QKV_SNAPSHOT_DIR_ENV} must use persistent storage, not /tmp")
    if path.exists() and not path.is_dir():
        raise ValueError(f"{QKV_SNAPSHOT_DIR_ENV} must be a directory")
    return path


def _full_qkv_snapshot_path(environment: Mapping[str, str]) -> Path | None:
    raw_path = environment.get(FULL_QKV_SNAPSHOT_PATH_ENV, "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{FULL_QKV_SNAPSHOT_PATH_ENV} must be an absolute path")
    path = path.resolve()
    if path == Path("/tmp") or Path("/tmp") in path.parents:
        raise ValueError(
            f"{FULL_QKV_SNAPSHOT_PATH_ENV} must use persistent storage, not /tmp"
        )
    if path.suffix != ".safetensors":
        raise ValueError(f"{FULL_QKV_SNAPSHOT_PATH_ENV} must end in .safetensors")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def research_package_identity(expected_root: Path) -> dict[str, Any]:
    import sageattention
    from sageattention import core

    package_path = Path(sageattention.__file__).resolve()
    binding_path = Path(core.sm80_compile.__file__).resolve()
    for path in (package_path, binding_path):
        if expected_root not in path.parents:
            raise RuntimeError(
                f"W28 capture imported {path} outside expected root {expected_root}"
            )
    operators = (
        "qk_int8_sv_f16_accum_f16_attn_row_pipeline",
        "qk_int8_sv_f16_accum_f16_attn_block_gating",
    )
    missing = [name for name in operators if not hasattr(core.sm80_compile, name)]
    if missing:
        raise RuntimeError(
            "W28 SageAttention binding is missing: " + ", ".join(missing)
        )
    extension_spec = importlib.util.find_spec("sageattention._qattn_sm80")
    if extension_spec is None or extension_spec.origin is None:
        raise RuntimeError("W28 SM80 qattn extension is unavailable")
    extension_path = Path(extension_spec.origin).resolve()
    if expected_root not in extension_path.parents:
        raise RuntimeError(
            f"W28 capture imported {extension_path} outside expected root {expected_root}"
        )
    return {
        "package": str(package_path),
        "binding": str(binding_path),
        "qattn_extension": str(extension_path),
        "qattn_extension_bytes": extension_path.stat().st_size,
        "qattn_extension_sha256": _sha256(extension_path),
        "required_operators": list(operators),
    }


def q_block_starts(real_sequence: int, selectors: tuple[str | int, ...]) -> list[int]:
    if real_sequence < BLOCK_Q:
        return []
    full_blocks = real_sequence // BLOCK_Q
    selected: list[int] = []
    for selector in selectors:
        if selector == "first":
            block = 0
        elif selector == "middle":
            block = full_blocks // 2
        elif selector == "last":
            block = full_blocks - 1
        else:
            block = int(selector)
        if block < full_blocks:
            selected.append(block * BLOCK_Q)
    return list(dict.fromkeys(selected))


def _block_mass_and_denominator(
    probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if probabilities.ndim != 2 or probabilities.shape[0] != BLOCK_Q:
        raise ValueError(f"probabilities must have shape [{BLOCK_Q}, K]")
    keys = probabilities.shape[1]
    padded_keys = math.ceil(keys / BLOCK_K) * BLOCK_K
    if padded_keys != keys:
        probabilities = torch.nn.functional.pad(probabilities, (0, padded_keys - keys))
    block_mass = probabilities.view(BLOCK_Q, -1, GATE_K).sum(dim=-1)
    tile_mass = block_mass.view(BLOCK_Q, -1, BLOCK_K // GATE_K).sum(dim=-1)
    tile_denominator = tile_mass.cumsum(dim=-1)
    denominator = tile_denominator.repeat_interleave(BLOCK_K // GATE_K, dim=-1)
    return block_mass[:, : math.ceil(keys / GATE_K)], denominator[
        :, : math.ceil(keys / GATE_K)
    ]


def cumulative_gate_from_block_mass(
    block_mass: torch.Tensor,
    denominator: torch.Tensor,
    *,
    budget: float = CUMULATIVE_BUDGET,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return q16/K16 skip decisions and final per-row skipped probability mass."""
    if block_mass.ndim != 2 or block_mass.shape[0] != BLOCK_Q:
        raise ValueError(f"block_mass must have shape [{BLOCK_Q}, key_blocks]")
    if denominator.shape != block_mass.shape:
        raise ValueError("denominator must have the same shape as block_mass")
    if budget != CUMULATIVE_BUDGET:
        raise ValueError("the W28 capture budget is fixed at 2e-4")

    masses = block_mass.detach().to(device="cpu", dtype=torch.float64)
    denominators = denominator.detach().to(device="cpu", dtype=torch.float64)
    q_groups = BLOCK_Q // GATE_K
    decisions = torch.zeros((q_groups, masses.shape[1]), dtype=torch.bool)
    skipped_mass = torch.zeros(BLOCK_Q, dtype=torch.float64)
    for group in range(q_groups):
        row_start = group * GATE_K
        row_stop = row_start + GATE_K
        group_skipped = skipped_mass[row_start:row_stop]
        for key_block in range(masses.shape[1]):
            candidate = group_skipped + masses[row_start:row_stop, key_block]
            limit = budget * denominators[row_start:row_stop, key_block]
            skip = bool(torch.isfinite(candidate).all() and (candidate <= limit).all())
            if skip:
                decisions[group, key_block] = True
                group_skipped.copy_(candidate)
    return decisions, skipped_mass


def _quantiles(values: torch.Tensor) -> dict[str, float | None]:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    points = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    if values.numel() == 0:
        return {f"p{round(point * 100):02d}": None for point in points}
    result = torch.quantile(
        values, torch.tensor(points, device=values.device, dtype=torch.float32)
    )
    return {
        f"p{round(point * 100):02d}": float(value)
        for point, value in zip(points, result.tolist(), strict=True)
    }


def _comparison(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate_f32 = candidate.float()
    reference_f32 = reference.float()
    candidate_finite = torch.isfinite(candidate_f32)
    reference_finite = torch.isfinite(reference_f32)
    joint_finite = candidate_finite & reference_finite
    candidate_nonfinite = int((~candidate_finite).sum().item())
    reference_nonfinite = int((~reference_finite).sum().item())
    result: dict[str, Any] = {
        "finite": candidate_nonfinite == 0 and reference_nonfinite == 0,
        "candidate_nonfinite": candidate_nonfinite,
        "reference_nonfinite": reference_nonfinite,
        "joint_nonfinite": int((~candidate_finite & ~reference_finite).sum().item()),
        "candidate_only_nonfinite": int(
            (~candidate_finite & reference_finite).sum().item()
        ),
        "reference_only_nonfinite": int(
            (candidate_finite & ~reference_finite).sum().item()
        ),
        "bitwise_equal": bool(torch.equal(candidate, reference)),
    }
    if not result["finite"]:
        result.update(
            mae=None,
            rmse=None,
            max_abs=None,
            relative_l2=None,
            joint_finite_fraction=float(joint_finite.float().mean().item()),
        )
        return result
    difference = candidate_f32 - reference_f32
    reference_norm = reference_f32.norm()
    relative_l2 = (
        difference.norm() / reference_norm if reference_norm > 0 else difference.norm()
    )
    result.update(
        mae=float(difference.abs().mean().item()),
        rmse=float(difference.square().mean().sqrt().item()),
        max_abs=float(difference.abs().max().item()),
        relative_l2=float(relative_l2.item()),
        joint_finite_fraction=1.0,
    )
    return result


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    finite_values = values[finite]
    result = {
        "elements": values.numel(),
        "nonfinite": int((~finite).sum().item()),
    }
    if finite_values.numel() == 0:
        return result | {
            "minimum": None,
            "maximum": None,
            "max_abs": None,
            "rms": None,
        }
    return result | {
        "minimum": float(finite_values.min().item()),
        "maximum": float(finite_values.max().item()),
        "max_abs": float(finite_values.abs().max().item()),
        "rms": float(finite_values.square().mean().sqrt().item()),
    }


def _dequantize_per_warp(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_indices = torch.arange(BLOCK_Q, device=q_int8.device) // WARP_Q
    k_indices = torch.arange(k_int8.shape[1], device=k_int8.device) // BLOCK_K
    q_rows = q_scale.index_select(2, q_indices).permute(0, 2, 1).unsqueeze(-1)
    k_rows = k_scale.index_select(2, k_indices).permute(0, 2, 1).unsqueeze(-1)
    return q_int8.float() * q_rows, k_int8.float() * k_rows


@torch.no_grad()
def analyze_activation_sample(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
) -> dict[str, Any]:
    """Analyze one aligned Q128 block, one head, and the complete live K/V."""
    if query.shape != (1, BLOCK_Q, 1, 128):
        raise ValueError("query sample must have shape [1, 128, 1, 128]")
    if key.ndim != 4 or key.shape[0] != 1 or key.shape[2:] != (1, 128):
        raise ValueError("key sample must have shape [1, K, 1, 128]")
    if value.shape != key.shape:
        raise ValueError("value sample must match key shape")
    if not query.is_cuda or query.device != key.device or query.device != value.device:
        raise ValueError("W28 activation analysis requires Q/K/V on one CUDA device")
    if (
        query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        raise ValueError("W28 activation analysis requires BF16 Q/K/V")

    from sageattention import core

    key_mean = key.mean(dim=1, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = core.per_warp_int8_cuda(
        query,
        key,
        key_mean,
        tensor_layout="NHD",
        BLKQ=BLOCK_Q,
        WARPQ=WARP_Q,
        BLKK=BLOCK_K,
    )
    q_dequant, k_dequant = _dequantize_per_warp(q_int8, k_int8, q_scale, k_scale)
    logits = torch.matmul(
        q_dequant[0, :, 0], k_dequant[0, :, 0].transpose(0, 1)
    ) * float(softmax_scale)
    probabilities = torch.softmax(logits, dim=-1)
    block_mass, denominator = _block_mass_and_denominator(probabilities)
    decisions, skipped_mass = cumulative_gate_from_block_mass(block_mass, denominator)

    skip_blocks_by_row = decisions.repeat_interleave(GATE_K, dim=0)
    skip_mask = skip_blocks_by_row.repeat_interleave(GATE_K, dim=1)[
        :, : key.shape[1]
    ].to(device=probabilities.device)
    value_f32 = value[0, :, 0].float()
    dense_reference = torch.matmul(probabilities, value_f32)
    gated_reference = torch.matmul(probabilities.masked_fill(skip_mask, 0.0), value_f32)

    value_half = value.to(torch.float16)
    release = torch.empty_like(query)
    release_repeat = torch.empty_like(query)
    candidate = torch.empty_like(query)
    candidate_repeat = torch.empty_like(query)
    operator_args = (
        q_int8,
        k_int8,
        value_half,
        q_scale,
        k_scale,
        0,
        0,
        2,
        float(softmax_scale),
        0,
    )
    for output in (release, release_repeat):
        core.sm80_compile.qk_int8_sv_f16_accum_f16_attn_row_pipeline(
            operator_args[0],
            operator_args[1],
            operator_args[2],
            output,
            *operator_args[3:],
        )
    for output in (candidate, candidate_repeat):
        core.sm80_compile.qk_int8_sv_f16_accum_f16_attn_block_gating(
            operator_args[0],
            operator_args[1],
            operator_args[2],
            output,
            *operator_args[3:],
        )

    q16_decisions = decisions.numel()
    q16_skips = int(decisions.sum().item())
    warp_decisions = decisions.view(BLOCK_Q // WARP_Q, 2, -1).all(dim=1)
    row_entropy = -(probabilities * probabilities.clamp_min(1.0e-30).log()).sum(dim=-1)
    return {
        "contract": {
            "q_block": BLOCK_Q,
            "q_scale_rows": WARP_Q,
            "k_scale_rows": BLOCK_K,
            "gate_k": GATE_K,
            "cumulative_budget": CUMULATIVE_BUDGET,
            "smooth_k_full_sequence_mean": True,
            "complete_live_k": True,
        },
        "quantization": {
            "q_scale_shape": list(q_scale.shape),
            "k_scale_shape": list(k_scale.shape),
            "key_mean_l2": float(key_mean.float().norm().item()),
        },
        "input_stats": {
            "query": _tensor_stats(query),
            "key": _tensor_stats(key),
            "centered_dequantized_key": _tensor_stats(k_dequant),
            "value": _tensor_stats(value),
            "logits": _tensor_stats(logits),
            "dense_reference": _tensor_stats(dense_reference),
            "release": _tensor_stats(release),
            "candidate": _tensor_stats(candidate),
        },
        "distribution": {
            "row_entropy": _quantiles(row_entropy),
            "row_max_probability": _quantiles(probabilities.max(dim=-1).values),
            "k16_probability_mass": _quantiles(block_mass),
        },
        "work_model": {
            "q16_decisions": q16_decisions,
            "q16_skips": q16_skips,
            "pv_hmma_skip_fraction": q16_skips / q16_decisions,
            "warp_key_blocks": warp_decisions.numel(),
            "ldsm_skips": int(warp_decisions.sum().item()),
            "pv_ldsm_skip_fraction": float(warp_decisions.float().mean().item()),
            "skipped_mass_fraction": _quantiles(skipped_mass),
            "budget_utilization": _quantiles(skipped_mass / CUMULATIVE_BUDGET),
            "maximum_skipped_mass_fraction": float(skipped_mass.max().item()),
        },
        "errors": {
            "oracle_gated_vs_dense": _comparison(gated_reference, dense_reference),
            "candidate_vs_release": _comparison(candidate, release),
            "release_vs_dense_quantized_reference": _comparison(
                release[0, :, 0], dense_reference
            ),
            "candidate_vs_gated_quantized_reference": _comparison(
                candidate[0, :, 0], gated_reference
            ),
            "release_repeatability": _comparison(release_repeat, release),
            "candidate_repeatability": _comparison(candidate_repeat, candidate),
        },
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_qkv_snapshot(
    path: Path,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: Mapping[str, Any],
    softmax_scale: float,
) -> dict[str, Any]:
    """Atomically persist one exact replay sample outside Git."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".partial")
    if path.exists() or partial_path.exists():
        raise FileExistsError(f"refusing to overwrite W28 Q/K/V snapshot: {path}")
    tensors = {
        "query": query.detach().to(device="cpu").contiguous(),
        "key": key.detach().to(device="cpu").contiguous(),
        "value": value.detach().to(device="cpu").contiguous(),
    }
    snapshot_metadata = {
        "schema_version": "1.0.0",
        "softmax_scale": repr(float(softmax_scale)),
        **{name: str(value) for name, value in metadata.items()},
    }
    try:
        save_file(tensors, str(partial_path), metadata=snapshot_metadata)
        with partial_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "format": "safetensors",
        "git_eligible": False,
        "tensor_shapes": {
            name: list(tensor.shape) for name, tensor in tensors.items()
        },
        "tensor_dtypes": {
            name: str(tensor.dtype) for name, tensor in tensors.items()
        },
    }


def _write_qkv_bundle(
    path: Path,
    *,
    queries: Mapping[int, torch.Tensor],
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: Mapping[str, Any],
    softmax_scale: float,
    remaining_bytes: int,
) -> dict[str, Any]:
    """Persist one deduplicated K/V stratum with all selected Q128 blocks."""
    from safetensors.torch import save_file

    if not queries:
        raise ValueError("Q/K/V bundle requires at least one query block")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".partial")
    if path.exists() or partial_path.exists():
        raise FileExistsError(f"refusing to overwrite W28 Q/K/V bundle: {path}")
    tensors = {
        "key": key.detach().to(device="cpu").contiguous(),
        "value": value.detach().to(device="cpu").contiguous(),
        **{
            f"query_{start}": query.detach().to(device="cpu").contiguous()
            for start, query in sorted(queries.items())
        },
    }
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    if tensor_bytes + 1024**2 > remaining_bytes:
        raise RuntimeError(
            "Q/K/V snapshot byte budget exceeded: "
            f"need {tensor_bytes}, remaining {remaining_bytes}"
        )
    free_bytes = shutil.disk_usage(path.parent).free
    if free_bytes - tensor_bytes < 2 * 1024**3:
        raise RuntimeError(
            "Q/K/V snapshot would leave less than 2 GiB free: "
            f"need {tensor_bytes}, free {free_bytes}"
        )
    snapshot_metadata = {
        "schema_version": "2.0.0",
        "softmax_scale": repr(float(softmax_scale)),
        "q_block_starts": ",".join(str(start) for start in sorted(queries)),
        **{name: str(value) for name, value in metadata.items()},
    }
    try:
        save_file(tensors, str(partial_path), metadata=snapshot_metadata)
        with partial_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "tensor_bytes": tensor_bytes,
        "sha256": _sha256(path),
        "format": "safetensors",
        "schema_version": "2.0.0",
        "git_eligible": False,
        "q_block_starts": sorted(queries),
        "tensor_shapes": {
            name: list(tensor.shape) for name, tensor in tensors.items()
        },
        "tensor_dtypes": {
            name: str(tensor.dtype) for name, tensor in tensors.items()
        },
    }


def _write_full_qkv_snapshot(
    path: Path,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: Mapping[str, Any],
    softmax_scale: float,
    remaining_bytes: int,
) -> dict[str, Any]:
    """Atomically persist one complete live 56-head attention input."""
    from safetensors.torch import save_file

    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("full Q/K/V snapshot requires matching four-dimensional tensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".partial")
    if path.exists() or partial_path.exists():
        raise FileExistsError(f"refusing to overwrite full H3 Q/K/V snapshot: {path}")
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in (query, key, value)
    )
    if tensor_bytes + 1024**2 > remaining_bytes:
        raise RuntimeError(
            "full Q/K/V snapshot byte budget exceeded: "
            f"need {tensor_bytes}, remaining {remaining_bytes}"
        )
    free_bytes = shutil.disk_usage(path.parent).free
    if free_bytes - tensor_bytes < 2 * 1024**3:
        raise RuntimeError(
            "full Q/K/V snapshot would leave less than 2 GiB free: "
            f"need {tensor_bytes}, free {free_bytes}"
        )
    tensors = {
        "query": query.detach().to(device="cpu").contiguous(),
        "key": key.detach().to(device="cpu").contiguous(),
        "value": value.detach().to(device="cpu").contiguous(),
    }
    snapshot_metadata = {
        "schema_version": "3.0.0",
        "softmax_scale": repr(float(softmax_scale)),
        **{name: str(value) for name, value in metadata.items()},
    }
    try:
        save_file(tensors, str(partial_path), metadata=snapshot_metadata)
        with partial_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "tensor_bytes": tensor_bytes,
        "sha256": _sha256(path),
        "format": "safetensors",
        "schema_version": "3.0.0",
        "git_eligible": False,
        "tensor_shapes": {
            name: list(tensor.shape) for name, tensor in tensors.items()
        },
        "tensor_dtypes": {
            name: str(tensor.dtype) for name, tensor in tensors.items()
        },
    }


def _state() -> _CaptureState:
    global _CAPTURE_STATE
    with _STATE_LOCK:
        if _CAPTURE_STATE is not None:
            return _CAPTURE_STATE
        config = parse_capture_config()
        if config is None:
            raise RuntimeError("W28 capture state requested while capture is disabled")
        package_identity = research_package_identity(config.sageattention_root)
        _CAPTURE_STATE = _CaptureState(config=config, sampled=set())
        _append_jsonl(
            config.output_path,
            {
                "schema_version": "1.0.0",
                "record_type": "run_start",
                "run_id": config.run_id,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "config": {
                    "layers": sorted(config.layers),
                    "steps": "*" if config.steps is None else sorted(config.steps),
                    "heads": sorted(config.heads),
                    "q_blocks": list(config.q_blocks),
                    "max_samples": config.max_samples,
                    "cumulative_budget": CUMULATIVE_BUDGET,
                    "sageattention_root": str(config.sageattention_root),
                    "qkv_snapshot_path": (
                        str(config.qkv_snapshot_path)
                        if config.qkv_snapshot_path is not None
                        else None
                    ),
                    "qkv_snapshot_dir": (
                        str(config.qkv_snapshot_dir)
                        if config.qkv_snapshot_dir is not None
                        else None
                    ),
                    "full_qkv_snapshot_path": (
                        str(config.full_qkv_snapshot_path)
                        if config.full_qkv_snapshot_path is not None
                        else None
                    ),
                    "qkv_snapshot_max_bytes": config.qkv_snapshot_max_bytes,
                    "analyze_samples": config.analyze_samples,
                },
                "sageattention": package_identity,
            },
        )
        return _CAPTURE_STATE


def maybe_capture_w28_activation(
    *,
    attention_prefix: str,
    step: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    real_sequence: int,
    softmax_scale: float,
) -> None:
    """Capture configured main-DiT strata; do nothing unless explicitly enabled."""
    if not capture_enabled():
        return
    state = _state()
    if not is_main_dit_attention_prefix(attention_prefix):
        return
    match = _MAIN_LAYER_PATTERN.fullmatch(attention_prefix)
    assert match is not None
    layer = int(match.group(1))
    config = state.config
    if layer not in config.layers or (
        config.steps is not None and step not in config.steps
    ):
        return
    if real_sequence <= 0 or real_sequence > query.shape[0]:
        raise ValueError("real_sequence must identify a non-empty Q/K/V prefix")

    if config.full_qkv_snapshot_path is not None and not state.full_snapshot_written:
        full_snapshot = _write_full_qkv_snapshot(
            config.full_qkv_snapshot_path,
            query=query[:real_sequence].unsqueeze(0).contiguous(),
            key=key[:real_sequence].unsqueeze(0).contiguous(),
            value=value[:real_sequence].unsqueeze(0).contiguous(),
            metadata={
                "run_id": config.run_id,
                "layer": layer,
                "step": step,
                "real_sequence": real_sequence,
                "physical_sequence": query.shape[0],
                "attention_prefix": attention_prefix,
            },
            softmax_scale=softmax_scale,
            remaining_bytes=config.qkv_snapshot_max_bytes - state.snapshot_bytes,
        )
        state.snapshot_bytes += full_snapshot["bytes"]
        state.full_snapshot_written = True
        _append_jsonl(
            config.output_path,
            {
                "schema_version": "1.0.0",
                "record_type": "full_qkv_snapshot",
                "run_id": config.run_id,
                "layer": layer,
                "step": step,
                "real_sequence": real_sequence,
                "physical_sequence": query.shape[0],
                "attention_prefix": attention_prefix,
                "snapshot": full_snapshot,
            },
        )

    starts = q_block_starts(real_sequence, config.q_blocks)
    for head in sorted(config.heads):
        if head >= query.shape[1]:
            raise ValueError(
                f"configured W28 head {head} exceeds local head count {query.shape[1]}"
            )
        pending_starts = [
            q_start
            for q_start in starts
            if (layer, step, head, q_start) not in state.sampled
        ]
        if not pending_starts:
            continue
        if len(state.sampled) + len(pending_starts) > config.max_samples:
            raise RuntimeError(
                f"W28 capture exceeded configured maximum of {config.max_samples} samples"
            )
        key_head = key[:real_sequence, head : head + 1].unsqueeze(0).contiguous()
        value_head = value[:real_sequence, head : head + 1].unsqueeze(0).contiguous()
        query_blocks = {
            q_start: (
                query[q_start : q_start + BLOCK_Q, head : head + 1]
                .unsqueeze(0)
                .contiguous()
            )
            for q_start in pending_starts
        }
        bundle = None
        if config.qkv_snapshot_dir is not None:
            safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.run_id)[:80]
            bundle_path = config.qkv_snapshot_dir / (
                f"{safe_run_id}-l{layer}-s{step}-h{head}.safetensors"
            )
            bundle = _write_qkv_bundle(
                bundle_path,
                queries=query_blocks,
                key=key_head,
                value=value_head,
                metadata={
                    "run_id": config.run_id,
                    "layer": layer,
                    "step": step,
                    "head": head,
                    "real_sequence": real_sequence,
                    "physical_sequence": query.shape[0],
                    "attention_prefix": attention_prefix,
                },
                softmax_scale=softmax_scale,
                remaining_bytes=config.qkv_snapshot_max_bytes - state.snapshot_bytes,
            )
            state.snapshot_bytes += bundle["bytes"]

        for q_start in pending_starts:
            sample_key = (layer, step, head, q_start)
            state.sampled.add(sample_key)
            metadata = {
                "schema_version": "1.0.0",
                "record_type": "activation_sample",
                "run_id": config.run_id,
                "layer": layer,
                "step": step,
                "head": head,
                "q_block_start": q_start,
                "real_sequence": real_sequence,
                "physical_sequence": query.shape[0],
                "dtype": str(query.dtype),
                "device": str(query.device),
                "attention_prefix": attention_prefix,
            }
            try:
                q_block = query_blocks[q_start]
                snapshot = None
                if config.qkv_snapshot_path is not None:
                    snapshot = _write_qkv_snapshot(
                        config.qkv_snapshot_path,
                        query=q_block,
                        key=key_head,
                        value=value_head,
                        metadata={
                            "run_id": config.run_id,
                            "layer": layer,
                            "step": step,
                            "head": head,
                            "q_block_start": q_start,
                            "real_sequence": real_sequence,
                            "physical_sequence": query.shape[0],
                            "attention_prefix": attention_prefix,
                        },
                        softmax_scale=softmax_scale,
                    )
                result = (
                    analyze_activation_sample(
                        q_block,
                        key_head,
                        value_head,
                        softmax_scale=softmax_scale,
                    )
                    if config.analyze_samples
                    else None
                )
                _append_jsonl(
                    config.output_path,
                    metadata
                    | {
                        "qkv_snapshot": snapshot,
                        "qkv_bundle": bundle,
                        "analysis": result,
                    },
                )
            except Exception as exc:
                _append_jsonl(
                    config.output_path,
                    metadata
                    | {
                        "record_type": "capture_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise


def _reset_capture_state_for_tests() -> None:
    global _CAPTURE_STATE
    with _STATE_LOCK:
        _CAPTURE_STATE = None
