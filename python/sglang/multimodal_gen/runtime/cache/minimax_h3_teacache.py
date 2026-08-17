# SPDX-License-Identifier: Apache-2.0
"""Request-local TeaCache state for the MiniMax H3 packed block stack."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

MINIMAX_H3_TEACACHE_NUM_STEPS_EXTRA_KEY = "minimax_h3_teacache_num_steps"


@dataclass
class MiniMaxH3TeaCacheState:
    previous_modulated_input: torch.Tensor | None = None
    previous_residual: torch.Tensor | None = None
    previous_calibration_output: torch.Tensor | None = None
    accumulated_distance: float = 0.0
    computed_steps: list[int] = field(default_factory=list)
    cached_steps: list[int] = field(default_factory=list)
    proxy_rel_l1: dict[int, float] = field(default_factory=dict)
    rescaled_rel_l1: dict[int, float] = field(default_factory=dict)
    output_rel_l1: dict[int, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_calibration_output = None
        self.accumulated_distance = 0.0
        self.computed_steps.clear()
        self.cached_steps.clear()
        self.proxy_rel_l1.clear()
        self.rescaled_rel_l1.clear()
        self.output_rel_l1.clear()

    def summary(self) -> dict[str, object]:
        return {
            "computed_steps": list(self.computed_steps),
            "cached_steps": list(self.cached_steps),
            "proxy_rel_l1": dict(self.proxy_rel_l1),
            "rescaled_rel_l1": dict(self.rescaled_rel_l1),
            "output_rel_l1": dict(self.output_rel_l1),
            "accumulated_distance": self.accumulated_distance,
        }


def _relative_l1(
    current: torch.Tensor,
    previous: torch.Tensor,
    *,
    chunk_rows: int = 4096,
    reduce_sums: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> float:
    if current.shape != previous.shape:
        raise ValueError(
            "MiniMax H3 TeaCache comparison shape mismatch: "
            f"current={tuple(current.shape)}, previous={tuple(previous.shape)}"
        )
    if current.numel() == 0 and reduce_sums is None:
        return 0.0
    if chunk_rows <= 0:
        raise ValueError("MiniMax H3 TeaCache chunk_rows must be positive")

    current_rows = current.reshape(-1, current.shape[-1])
    previous_rows = previous.reshape_as(current_rows)
    numerator = torch.zeros((), dtype=torch.float32, device=current.device)
    denominator = torch.zeros((), dtype=torch.float32, device=current.device)
    for start in range(0, current_rows.shape[0], chunk_rows):
        stop = min(start + chunk_rows, current_rows.shape[0])
        numerator.add_(
            (current_rows[start:stop] - previous_rows[start:stop])
            .abs()
            .sum(dtype=torch.float32)
        )
        denominator.add_(previous_rows[start:stop].abs().sum(dtype=torch.float32))
    sums = torch.stack((numerator, denominator))
    if reduce_sums is not None:
        sums = reduce_sums(sums)
    value = (sums[0] / sums[1].clamp_min(1e-8)).item()
    if not math.isfinite(value):
        raise ValueError("MiniMax H3 TeaCache relative L1 is non-finite")
    return value


def _rescale_distance(value: float, coefficients: list[float]) -> float:
    if not coefficients:
        raise ValueError("MiniMax H3 TeaCache coefficients must not be empty")
    result = 0.0
    for coefficient in coefficients:
        coefficient = float(coefficient)
        if not math.isfinite(coefficient):
            raise ValueError("MiniMax H3 TeaCache coefficient is non-finite")
        result = result * value + coefficient
    if not math.isfinite(result):
        raise ValueError(f"MiniMax H3 TeaCache rescaled relative L1 is invalid: {result}")
    return result


def decide_minimax_h3_teacache(
    state: MiniMaxH3TeaCacheState,
    modulated_input: torch.Tensor,
    *,
    step: int,
    num_steps: int,
    threshold: float,
    start_skipping: int,
    end_skipping: int,
    coefficients: list[float],
    reduce_sums: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> bool:
    """Return True to execute the H3 block stack, False to reuse its residual."""

    if step == 0:
        state.reset()
    if step < 0 or step >= num_steps:
        raise ValueError(f"MiniMax H3 TeaCache step {step} outside [0, {num_steps})")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "MiniMax H3 TeaCache threshold must be finite and non-negative"
        )

    previous = state.previous_modulated_input
    raw_distance = None
    scaled_distance = None
    if previous is not None:
        raw_distance = _relative_l1(
            modulated_input,
            previous,
            reduce_sums=reduce_sums,
        )
        scaled_distance = _rescale_distance(raw_distance, coefficients)
        state.proxy_rel_l1[step] = raw_distance
        state.rescaled_rel_l1[step] = scaled_distance

    boundary = step < start_skipping or step >= end_skipping
    if boundary or previous is None or state.previous_residual is None:
        should_compute = True
        state.accumulated_distance = 0.0
    else:
        assert scaled_distance is not None
        state.accumulated_distance += abs(scaled_distance)
        should_compute = state.accumulated_distance >= threshold
        if should_compute:
            state.accumulated_distance = 0.0

    # The official TeaCache algorithm advances the proxy baseline every step,
    # including cache hits. Ownership transfers to the request-local state.
    state.previous_modulated_input = modulated_input.detach()
    if should_compute:
        state.computed_steps.append(step)
    else:
        state.cached_steps.append(step)
    return should_compute


def update_minimax_h3_teacache_residual(
    state: MiniMaxH3TeaCacheState,
    block_output: torch.Tensor,
    block_input: torch.Tensor,
    *,
    step: int,
    collect_calibration: bool = False,
    valid_rows: int | None = None,
    reduce_sums: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    if block_output.shape != block_input.shape:
        raise ValueError(
            "MiniMax H3 TeaCache block input/output shape mismatch: "
            f"output={tuple(block_output.shape)}, input={tuple(block_input.shape)}"
        )
    if block_output.data_ptr() == block_input.data_ptr():
        raise RuntimeError(
            "MiniMax H3 TeaCache block input was overwritten in place; "
            "residual caching requires an independent input snapshot"
        )

    if valid_rows is not None and (
        valid_rows < 0 or valid_rows > block_output.shape[0]
    ):
        raise ValueError(
            "MiniMax H3 TeaCache valid_rows outside block output shape: "
            f"{valid_rows} not in [0, {block_output.shape[0]}]"
        )

    previous_output = state.previous_calibration_output
    if collect_calibration and previous_output is not None:
        metric_output = block_output
        metric_previous = previous_output
        if valid_rows is not None:
            metric_output = block_output[:valid_rows]
            metric_previous = previous_output[:valid_rows]
        state.output_rel_l1[step] = _relative_l1(
            metric_output,
            metric_previous,
            reduce_sums=reduce_sums,
        )
    state.previous_calibration_output = (
        block_output.detach() if collect_calibration else None
    )

    residual = state.previous_residual
    if (
        residual is None
        or residual.shape != block_output.shape
        or residual.device != block_output.device
        or residual.dtype != block_output.dtype
    ):
        residual = torch.empty_like(block_output)
    torch.sub(block_output, block_input, out=residual)
    state.previous_residual = residual.detach()


__all__ = [
    "MINIMAX_H3_TEACACHE_NUM_STEPS_EXTRA_KEY",
    "MiniMaxH3TeaCacheState",
    "decide_minimax_h3_teacache",
    "update_minimax_h3_teacache_residual",
]
