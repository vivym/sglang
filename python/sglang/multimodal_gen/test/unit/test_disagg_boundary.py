from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.disaggregation.boundary import (
    DISAGG_BOUNDARY_FIELD_PREFIX,
    partition_boundary_fields,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
)
from sglang.multimodal_gen.runtime.pipelines_core import Req


_TENSOR_KEY = f"{DISAGG_BOUNDARY_FIELD_PREFIX}test_tensor"
_SCALAR_KEY = f"{DISAGG_BOUNDARY_FIELD_PREFIX}test_schema"


class _BoundaryPipeline:
    def export_disagg_boundary(self, req, *, source_role, destination_role):
        assert source_role is RoleType.ENCODER
        assert destination_role is RoleType.DENOISER
        return {_TENSOR_KEY: req.extra["model_tensor"]}, {_SCALAR_KEY: "v1"}

    def restore_disagg_boundary(
        self,
        req,
        *,
        source_role,
        destination_role,
        tensor_fields,
        scalar_fields,
    ):
        assert source_role is RoleType.ENCODER
        assert destination_role is RoleType.DENOISER
        req.extra["restored_tensor"] = tensor_fields[_TENSOR_KEY]
        req.extra["restored_schema"] = scalar_fields[_SCALAR_KEY]


def _scheduler(pipeline=None):
    return SimpleNamespace(
        worker=SimpleNamespace(pipeline=pipeline),
        _disagg_role=RoleType.DENOISER,
    )


def test_model_boundary_round_trips_without_leaking_wire_fields_to_req():
    pipeline = _BoundaryPipeline()
    scheduler = _scheduler(pipeline)
    req = Req(request_id="boundary", prompt="test")
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    req.extra["model_tensor"] = expected

    tensors, scalars = SchedulerDisaggMixin._extract_disagg_transfer_fields(
        scheduler,
        req,
        source_role=RoleType.ENCODER,
        destination_role=RoleType.DENOISER,
    )
    rebuilt = SchedulerDisaggMixin._build_disagg_req(
        scheduler, dict(scalars), dict(tensors)
    )

    assert torch.equal(rebuilt.extra["restored_tensor"], expected)
    assert rebuilt.extra["restored_schema"] == "v1"
    assert not hasattr(rebuilt, _TENSOR_KEY)
    assert not hasattr(rebuilt, _SCALAR_KEY)


@pytest.mark.parametrize("kind", ["tensor", "scalar"])
def test_model_boundary_export_requires_reserved_namespace(kind):
    class InvalidPipeline(_BoundaryPipeline):
        def export_disagg_boundary(self, req, *, source_role, destination_role):
            del req, source_role, destination_role
            if kind == "tensor":
                return {"unscoped": torch.ones(1)}, {}
            return {}, {"unscoped": "v1"}

    req = Req(request_id="invalid", prompt="test")
    with pytest.raises(ValueError, match="must start with"):
        SchedulerDisaggMixin._extract_disagg_transfer_fields(
            _scheduler(InvalidPipeline()),
            req,
            source_role=RoleType.ENCODER,
            destination_role=RoleType.DENOISER,
        )


def test_model_boundary_export_rejects_tensor_scalar_name_collision():
    class InvalidPipeline(_BoundaryPipeline):
        def export_disagg_boundary(self, req, *, source_role, destination_role):
            del req, source_role, destination_role
            return {_TENSOR_KEY: torch.ones(1)}, {_TENSOR_KEY: "not-a-tensor"}

    req = Req(request_id="collision", prompt="test")
    with pytest.raises(ValueError, match="both tensor and scalar"):
        SchedulerDisaggMixin._extract_disagg_transfer_fields(
            _scheduler(InvalidPipeline()),
            req,
            source_role=RoleType.ENCODER,
            destination_role=RoleType.DENOISER,
        )


def test_build_rejects_boundary_fields_without_restore_hook():
    req = Req(request_id="missing-hook", prompt="test")
    _, scalars = SchedulerDisaggMixin._extract_disagg_transfer_fields(
        SimpleNamespace(worker=None),
        req,
        source_role=RoleType.ENCODER,
        destination_role=RoleType.DENOISER,
    )
    scalars[_SCALAR_KEY] = "v1"

    with pytest.raises(ValueError, match="without a pipeline restore hook"):
        SchedulerDisaggMixin._build_disagg_req(None, scalars, {})


def test_partition_boundary_fields_preserves_both_partitions():
    ordinary, boundary = partition_boundary_fields(
        {"request_id": "r", _SCALAR_KEY: "v1"}
    )
    assert ordinary == {"request_id": "r"}
    assert boundary == {_SCALAR_KEY: "v1"}
