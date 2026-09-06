# SPDX-License-Identifier: Apache-2.0
"""Machine-readable receipts for disaggregated diffusion requests."""

from __future__ import annotations

import json
import math
from typing import Any


DISAGG_RECEIPT_SCHEMA = "sglang.diffusion.disagg-receipt/v1"


def log_disagg_receipt(
    logger: Any,
    event: str,
    request_id: str,
    **fields: Any,
) -> dict[str, Any]:
    """Emit one stable JSON object and return it for focused tests."""
    receipt = {
        "schema": DISAGG_RECEIPT_SCHEMA,
        "event": event,
        "request_id": request_id,
        **fields,
    }
    for name, value in receipt.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"disaggregation receipt field {name!r} is not finite")
    logger.info(
        "disagg_receipt=%s",
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )
    return receipt


def cuda_events_elapsed_s(start_event: Any, end_event: Any) -> float | None:
    """Wait for a CUDA copy-completion event and return its device elapsed time."""
    if end_event is None:
        return None
    end_event.synchronize()
    if start_event is None:
        return None
    try:
        return float(start_event.elapsed_time(end_event)) / 1000.0
    except (AttributeError, RuntimeError, TypeError):
        return None
