from __future__ import annotations

import json

import pytest

from sglang.multimodal_gen.runtime.disaggregation.telemetry import (
    DISAGG_RECEIPT_SCHEMA,
    cuda_events_elapsed_s,
    log_disagg_receipt,
)


class _Logger:
    def __init__(self) -> None:
        self.args = None

    def info(self, *args) -> None:
        self.args = args


class _Event:
    def __init__(self) -> None:
        self.synchronized = False

    def synchronize(self) -> None:
        self.synchronized = True

    def elapsed_time(self, other) -> float:
        assert isinstance(other, _Event)
        assert other.synchronized
        return 12.5


class _UntimedEvent(_Event):
    def elapsed_time(self, other) -> float:
        raise RuntimeError("timing is unsupported")


def test_disagg_receipt_is_stable_json():
    logger = _Logger()
    receipt = log_disagg_receipt(
        logger,
        "tensor_wire_push",
        "request-a",
        payload_bytes=4096,
        wall_s=0.125,
        status="ok",
    )

    assert receipt["schema"] == DISAGG_RECEIPT_SCHEMA
    assert logger.args[0] == "disagg_receipt=%s"
    assert json.loads(logger.args[1]) == receipt


def test_disagg_receipt_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="not finite"):
        log_disagg_receipt(_Logger(), "tensor_h2d", "request-a", cuda_s=float("nan"))


def test_cuda_events_elapsed_seconds():
    start = _Event()
    end = _Event()
    assert cuda_events_elapsed_s(start, end) == pytest.approx(0.0125)
    assert end.synchronized

    end_without_start = _Event()
    assert cuda_events_elapsed_s(None, end_without_start) is None
    assert end_without_start.synchronized

    untimed_end = _UntimedEvent()
    assert cuda_events_elapsed_s(_UntimedEvent(), untimed_end) is None
    assert untimed_end.synchronized
