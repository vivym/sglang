import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3PDDHead


def _bare_head() -> MiniMaxH3PDDHead:
    head = MiniMaxH3PDDHead.__new__(MiniMaxH3PDDHead)
    torch.nn.Module.__init__(head)
    head.num_steps = 32
    head.block_size = 4
    head.nfe = 8
    head.video_shift = 12.0
    head.audio_shift = 3.0
    head._schedule_validated = False
    return head


def test_pdd_sampling_plan_matches_released_block_formula():
    plans = MiniMaxH3PDDHead._sampling_plans(12.0, 32, 4)
    grid = 1.0 - MiniMaxH3PDDHead._shifted_sigmas(12.0, 32)
    step_sizes = grid.diff()

    expected = torch.zeros((8, 32), dtype=torch.float32)
    for step, start in enumerate(range(0, 32, 4)):
        block = step_sizes[start : start + 4]
        expected[step, start : start + 4] = (block / block.sum()).float()

    assert torch.equal(plans, expected)
    assert torch.equal(plans.count_nonzero(dim=1), torch.full((8,), 4))
    assert torch.allclose(plans.sum(dim=1), torch.ones(8), atol=0, rtol=0)


def test_pdd_projection_matches_released_parallel_head_formula():
    hidden = torch.arange(15, dtype=torch.bfloat16).reshape(3, 5) / 8
    weight = torch.arange(4 * 2 * 5, dtype=torch.bfloat16).reshape(4, 2, 5) / 16
    bias = torch.arange(4 * 2, dtype=torch.bfloat16).reshape(4, 2) / 8
    plan = torch.tensor([0.0, 0.25, 0.75, 0.0], dtype=torch.float32)

    actual = MiniMaxH3PDDHead._project(hidden, plan, weight, bias)
    official_weight = torch.einsum(
        "pn,noi->poi", plan[None].to(torch.bfloat16), weight
    ).flatten(0, 1)
    official_bias = torch.einsum(
        "pn,no->po", plan[None].to(torch.bfloat16), bias
    ).flatten()
    expected = torch.nn.functional.linear(hidden, official_weight, official_bias)

    assert torch.equal(actual, expected)


def test_pdd_materialized_heads_are_bitwise_equal_to_released_einsum():
    head = _bare_head()
    head.nfe = 2
    video_weight = torch.arange(4 * 3 * 5, dtype=torch.bfloat16).reshape(4, 3, 5) / 13
    video_bias = torch.arange(4 * 3, dtype=torch.bfloat16).reshape(4, 3) / 11
    audio_weight = torch.arange(4 * 2 * 5, dtype=torch.bfloat16).reshape(4, 2, 5) / 17
    audio_bias = torch.arange(4 * 2, dtype=torch.bfloat16).reshape(4, 2) / 7
    plans = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32
    )
    head.register_buffer("video_weight", video_weight, persistent=False)
    head.register_buffer("video_bias", video_bias, persistent=False)
    head.register_buffer("audio_weight", audio_weight, persistent=False)
    head.register_buffer("audio_bias", audio_bias, persistent=False)
    head.register_buffer("video_plans", plans, persistent=False)
    head.register_buffer("audio_plans", plans.flip(1), persistent=False)
    head.register_buffer("fused_video_weight", None, persistent=False)
    head.register_buffer("fused_video_bias", None, persistent=False)
    head.register_buffer("fused_audio_weight", None, persistent=False)
    head.register_buffer("fused_audio_bias", None, persistent=False)

    expected_video_weight = torch.einsum(
        "pn,noi->poi", plans.to(torch.bfloat16), video_weight
    )
    expected_video_bias = torch.einsum(
        "pn,no->po", plans.to(torch.bfloat16), video_bias
    )
    expected_audio_weight = torch.einsum(
        "pn,noi->poi", plans.flip(1).to(torch.bfloat16), audio_weight
    )
    expected_audio_bias = torch.einsum(
        "pn,no->po", plans.flip(1).to(torch.bfloat16), audio_bias
    )

    head.materialize_fused_heads()

    assert torch.equal(head.fused_video_weight, expected_video_weight)
    assert torch.equal(head.fused_video_bias, expected_video_bias)
    assert torch.equal(head.fused_audio_weight, expected_audio_weight)
    assert torch.equal(head.fused_audio_bias, expected_audio_bias)
    assert head.video_weight is None
    assert head.video_bias is None
    assert head.audio_weight is None
    assert head.audio_bias is None


def test_pdd_schedule_accepts_only_released_eight_nfe_contract():
    head = _bare_head()
    video = MiniMaxH3PDDHead._coarse_schedule(12.0, 32, 4)
    audio = MiniMaxH3PDDHead._coarse_schedule(3.0, 32, 4)

    head.validate_schedule(video, audio, sampler_mode="euler")

    assert head._schedule_validated
    with pytest.raises(ValueError, match="released video schedule"):
        head.validate_schedule(video[:-1], audio, sampler_mode="euler")
    with pytest.raises(ValueError, match="sampler_mode='euler'"):
        head.validate_schedule(video, audio, sampler_mode="res_multistep")
