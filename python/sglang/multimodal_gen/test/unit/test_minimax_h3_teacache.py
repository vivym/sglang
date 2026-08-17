from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.runtime.cache.minimax_h3_teacache import (
    MINIMAX_H3_TEACACHE_NUM_STEPS_EXTRA_KEY,
    MiniMaxH3TeaCacheState,
    _relative_l1,
    decide_minimax_h3_teacache,
    update_minimax_h3_teacache_residual,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel


def _decide(state, value, step, *, threshold=0.15):
    return decide_minimax_h3_teacache(
        state,
        torch.full((3, 2), value, dtype=torch.bfloat16),
        step=step,
        num_steps=6,
        threshold=threshold,
        start_skipping=1,
        end_skipping=5,
        coefficients=[1.0, 0.0],
    )


def test_relative_l1_uses_full_tensor_with_chunking():
    previous = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    current = previous + torch.tensor([[1.0, -1.0], [0.0, 2.0], [-2.0, 0.0]])

    actual = _relative_l1(current, previous, chunk_rows=1)
    expected = (current - previous).abs().sum() / previous.abs().sum()

    assert actual == pytest.approx(float(expected), rel=1e-6)


def test_relative_l1_reduces_numerator_and_denominator_together():
    reduced = []

    def reduce_sums(value):
        reduced.append(value.clone())
        return value * 2

    actual = _relative_l1(
        torch.tensor([[2.0, 4.0]]),
        torch.tensor([[1.0, 2.0]]),
        reduce_sums=reduce_sums,
    )

    assert actual == pytest.approx(1.0)
    torch.testing.assert_close(reduced[0], torch.tensor([3.0, 3.0]))


def test_empty_sequence_parallel_rank_still_joins_reduction():
    calls = 0

    def reduce_sums(value):
        nonlocal calls
        calls += 1
        torch.testing.assert_close(value, torch.zeros(2))
        return torch.tensor([2.0, 4.0])

    actual = _relative_l1(
        torch.empty(0, 2),
        torch.empty(0, 2),
        reduce_sums=reduce_sums,
    )

    assert calls == 1
    assert actual == pytest.approx(0.5)


def test_teacache_accumulates_deltas_and_protects_boundaries():
    state = MiniMaxH3TeaCacheState()

    assert _decide(state, 1.0, 0)
    update_minimax_h3_teacache_residual(
        state,
        torch.ones(3, 2),
        torch.zeros(3, 2),
        step=0,
    )
    assert not _decide(state, 1.05, 1)
    assert state.accumulated_distance == pytest.approx(0.046875, rel=0.05)
    assert not _decide(state, 1.10, 2)
    assert _decide(state, 1.20, 3)
    update_minimax_h3_teacache_residual(
        state,
        torch.full((3, 2), 1.2),
        torch.zeros(3, 2),
        step=3,
    )
    assert not _decide(state, 1.22, 4)
    assert _decide(state, 1.23, 5)

    assert state.computed_steps == [0, 3, 5]
    assert state.cached_steps == [1, 2, 4]


def test_zero_threshold_forces_dense_calibration_and_records_output_delta():
    state = MiniMaxH3TeaCacheState()

    for step, value in enumerate((1.0, 1.1, 1.2)):
        assert _decide(state, value, step, threshold=0.0)
        update_minimax_h3_teacache_residual(
            state,
            torch.full((3, 2), value * 2),
            torch.full((3, 2), value),
            step=step,
            collect_calibration=True,
        )

    assert state.computed_steps == [0, 1, 2]
    assert state.cached_steps == []
    assert set(state.proxy_rel_l1) == {1, 2}
    assert set(state.output_rel_l1) == {1, 2}
    assert state.output_rel_l1[1] == pytest.approx(0.1, rel=0.05)


def test_output_metric_excludes_packed_padding_rows_and_caches_true_residual():
    previous_output = torch.ones(3, 2)
    previous_residual = torch.empty(3, 2)
    state = MiniMaxH3TeaCacheState(
        previous_residual=previous_residual,
        previous_calibration_output=previous_output,
    )
    block_input = torch.tensor([[0.5, 0.5], [0.5, 0.5], [20.0, 20.0]])
    block_output = torch.tensor([[2.0, 2.0], [2.0, 2.0], [100.0, 100.0]])

    update_minimax_h3_teacache_residual(
        state,
        block_output,
        block_input,
        step=1,
        collect_calibration=True,
        valid_rows=2,
    )

    assert state.output_rel_l1[1] == pytest.approx(1.0)
    torch.testing.assert_close(state.previous_residual, block_output - block_input)
    assert state.previous_residual.data_ptr() == previous_residual.data_ptr()
    assert state.previous_calibration_output.data_ptr() == block_output.data_ptr()


def test_residual_update_rejects_in_place_block_input_alias():
    state = MiniMaxH3TeaCacheState()
    aliased = torch.ones(3, 2)

    with pytest.raises(RuntimeError, match="overwritten in place"):
        update_minimax_h3_teacache_residual(
            state,
            aliased,
            aliased,
            step=0,
        )


def test_step_zero_resets_tensors_and_metrics():
    state = MiniMaxH3TeaCacheState()
    assert _decide(state, 1.0, 0)
    update_minimax_h3_teacache_residual(
        state,
        torch.ones(3, 2),
        torch.zeros(3, 2),
        step=0,
    )
    assert not _decide(state, 1.01, 1)

    assert _decide(state, 2.0, 0)
    assert state.computed_steps == [0]
    assert state.cached_steps == []
    assert state.previous_residual is None
    assert state.previous_calibration_output is None


def test_negative_polynomial_value_contributes_by_magnitude():
    state = MiniMaxH3TeaCacheState()

    assert _decide(state, 1.0, 0, threshold=0.05)
    update_minimax_h3_teacache_residual(
        state,
        torch.ones(3, 2),
        torch.zeros(3, 2),
        step=0,
    )
    should_compute = decide_minimax_h3_teacache(
        state,
        torch.full((3, 2), 1.1, dtype=torch.bfloat16),
        step=1,
        num_steps=6,
        threshold=0.05,
        start_skipping=1,
        end_skipping=5,
        coefficients=[-1.0, 0.0],
    )

    assert should_compute
    assert state.rescaled_rel_l1[1] < 0.0


def test_h3_context_uses_actual_dit_calls_for_negative_boundaries():
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    batch = SimpleNamespace(
        enable_teacache=True,
        teacache_params=TeaCacheParams(
            teacache_thresh=0.17,
            start_skipping=2,
            end_skipping=-2,
            coefficients=[1.0, 0.0],
        ),
        do_classifier_free_guidance=False,
        num_inference_steps=20,
        extra={MINIMAX_H3_TEACACHE_NUM_STEPS_EXTRA_KEY: 19},
    )

    with set_forward_context(
        current_timestep=18,
        attn_metadata=None,
        forward_batch=batch,
    ):
        config = model._minimax_h3_teacache_config()

    assert config == (18, 19, 0.17, 2, 17, [1.0, 0.0])


@pytest.mark.parametrize(
    ("threshold", "coefficients", "error"),
    [
        (-0.1, [1.0, 0.0], "threshold"),
        (0.1, [], "coefficients"),
        (0.1, [float("nan")], "coefficient"),
    ],
)
def test_teacache_rejects_invalid_configuration(threshold, coefficients, error):
    state = MiniMaxH3TeaCacheState(
        previous_modulated_input=torch.ones(2, 2),
        previous_residual=torch.ones(2, 2),
    )

    with pytest.raises(ValueError, match=error):
        decide_minimax_h3_teacache(
            state,
            torch.ones(2, 2),
            step=1,
            num_steps=3,
            threshold=threshold,
            start_skipping=0,
            end_skipping=3,
            coefficients=coefficients,
        )
