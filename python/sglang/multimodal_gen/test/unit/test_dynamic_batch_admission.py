# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.managers.dynamic_batch_admission import (
    BatchAdmissionController,
    BatchingRule,
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


def test_encoder_token_cap_uses_right_padded_batch_size_without_cost_estimate():
    class EncoderTokenConfig(_UnexpectedCostConfig):
        @staticmethod
        def estimate_encoder_batch_tokens(reqs):
            if any(not hasattr(req, "encoder_tokens") for req in reqs):
                return None
            return len(reqs) * max(req.encoder_tokens for req in reqs)

    controller = _controller()
    controller._pipeline_config = EncoderTokenConfig()
    controller._rules = [
        BatchingRule(
            model="model",
            max_batch_size=2,
            max_encoder_batch_tokens=40960,
        )
    ]
    short = _req()
    short.encoder_tokens = 47
    long = _req()
    long.encoder_tokens = 40960

    assert controller.reject_reason_for_candidate([short], _req()) == (
        "encoder_token_budget:unavailable"
    )
    assert controller.reject_reason_for_candidate([short], short) is None
    assert controller.reject_reason_for_candidate([long], short) == (
        "encoder_token_budget:81920>40960"
    )
    assert controller.batch_is_full([long]) is True
    assert controller.limit_reason_for_batch([long]) == (
        "encoder_token_budget_next:81920>40960"
    )
