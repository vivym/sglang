# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Sequence


def minimax_h3_align_frame_count(frame_count: int) -> int:
    """Snap ``frame_count`` up to the MiniMax H3 17n+5 frame boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    return current + (5 - current) % 17


def minimax_h3_video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def minimax_h3_frame_count_from_video_latent_t(out_t: int) -> int:
    if out_t == 1:
        return 1
    if out_t < 2 or (out_t - 2) % 5 != 0:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(out_t) - 2) // 5) + 5


def minimax_h3_audio_latent_t(duration_seconds: float) -> int:
    # Rounding happens at the 40 Hz audio latent boundary.
    return int(round(float(duration_seconds) * 40.0))


def minimax_h3_time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
    base_schedule: Sequence[float] | None = None,
) -> list[float]:
    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")
    if num_steps <= 0:
        raise ValueError("MiniMax H3 num_steps must be > 0")

    import torch

    if base_schedule is None:
        # The default rectified-flow sigma range is uniform over [1.0, 0.0].
        base = torch.linspace(
            1.0,
            0.0,
            int(num_steps),
            device="cpu",
            dtype=torch.float32,
        )
    else:
        values = tuple(base_schedule)
        if len(values) != int(num_steps):
            raise ValueError(
                "MiniMax H3 base_schedule length must equal num_steps: "
                f"{len(values)} != {num_steps}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("MiniMax H3 base_schedule must contain finite numbers")
        normalized = tuple(float(value) for value in values)
        if normalized[0] != 1.0 or normalized[-1] != 0.0:
            raise ValueError(
                "MiniMax H3 base_schedule must start at 1.0 and end at 0.0"
            )
        if any(
            current <= following
            for current, following in zip(normalized, normalized[1:])
        ):
            raise ValueError("MiniMax H3 base_schedule must be strictly decreasing")
        base = torch.tensor(normalized, device="cpu", dtype=torch.float32)
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
    # A one-point request is still exactly one point.  Normal serving uses
    # multiple points, but preserving the requested cardinality keeps
    # ``num_inference_steps`` the sole schedule-size control.
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
    return [float(value) for value in shifted.tolist()]
