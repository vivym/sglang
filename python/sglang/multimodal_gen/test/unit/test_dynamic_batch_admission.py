# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.managers.dynamic_batch_admission import (
    BatchAdmissionController,
)


class _UnexpectedCostConfig:
    def estimate_request_cost(self, _req):
        raise AssertionError("cost must not be evaluated without a max_cost cap")


def _controller() -> BatchAdmissionController:
    controller = BatchAdmissionController.__new__(BatchAdmissionController)
    controller._mode = "dynamic"
    controller._user_max_batch_size = 2
    controller._model_path = "model"
    controller._offload = False
    controller._device_memory_gb = 64.0
    controller._rules = []
    controller._pipeline_config = _UnexpectedCostConfig()
    return controller


def _req():
    return SimpleNamespace(num_outputs_per_prompt=1, resolution_key="768x1344")


def test_admission_skips_cost_estimation_without_cost_cap():
    controller = _controller()
    current = _req()

    assert controller.reject_reason_for_candidate([current], _req()) is None
    assert controller.batch_is_full([current]) is False
    assert controller.limit_reason_for_batch([current]) is None
