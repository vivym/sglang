# SPDX-License-Identifier: Apache-2.0
"""Numerical contract for request-static H3 denoise metadata."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_minimax_h3_euler_ancestral import (
    _minimax_h3_euler_eta0_step,
    _minimax_h3_rf_v_to_x0,
    minimax_h3_res_multistep_coeffs,
    minimax_h3_res_multistep_eta0_step,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    _build_local_embedding_layout,
    _minimax_h3_res_multistep_update_target_rows_,
    _minimax_h3_update_target_rows_,
    minimax_h3_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
    _resolve_debug_latent_dump_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)


def _branch(
    mode: str, token_tags: torch.Tensor | None = None
) -> MiniMaxH3DenoiseBranch:
    common = dict(text_len=3, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    if mode == "t2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=False,
        )
    elif mode == "fl2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=True,
            keyframe_frame_indices=[0, -1],
            frame_count=5,
        )
    else:
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            **common,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
            ],
        )
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(3, 5120),
        token_tags=packed["token_tags"] if token_tags is None else token_tags,
        device=torch.device("cpu"),
    )


def test_precomputed_timestep_plan_matches_full_unique_reference():
    """Preplanning must preserve fp32 collisions and every packed row class."""

    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        assert branch.static_kwargs["skip_mask_out_condition"]
        assert "token_tags" not in branch.static_kwargs
        assert not bool((branch.static_kwargs["block_token_tags"] < 0).any())
        torch.testing.assert_close(
            branch.static_kwargs["img_pos_for_infer_output_info"]["position_ids"],
            branch.img_pos_dev[branch.update_mask_dev],
            rtol=0,
            atol=0,
        )
        video_steps = [0.75, 0.1]
        audio_steps = [0.625, 0.2]
        plan = branch.prepare_timestep_plan(
            video_timesteps=video_steps,
            audio_timesteps=audio_steps,
            imgvid_cond_noise_aug=0.6,
            audio_ref_cond_noise_aug=0.4,
        )

        assert branch.static_kwargs["packed_seq_params"]["cu_seqlens_q_host"] == tuple(
            int(value)
            for value in branch.static_kwargs["packed_seq_params"][
                "cu_seqlens_q"
            ].tolist()
        )
        assert branch.static_kwargs["refiner_packed_seq_params"][
            "cu_seqlens_q_host"
        ] == (0, 3, 3)

        for step, (video_t, audio_t) in enumerate(
            zip(video_steps, audio_steps, strict=True)
        ):
            reference = torch.full((branch.seq_len,), video_t, dtype=torch.float32)
            reference[branch.img_cond_seq_idx] = max(video_t, 0.6)
            reference[branch.audio_target_seq_idx] = audio_t
            reference[branch.audio_ref_seq_idx] = max(audio_t, 0.4)
            expected = torch.unique(reference, sorted=True, return_inverse=True)
            torch.testing.assert_close(plan[step][0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(plan[step][1], expected[1], rtol=0, atol=0)
            torch.testing.assert_close(
                plan[step][2],
                branch.static_kwargs["block_token_tags"]
                + expected[1] * MINIMAX_H3_ADALN_MODALITY_NUM,
                rtol=0,
                atol=0,
            )

        repeated_plan = branch.prepare_timestep_plan(
            video_timesteps=[0.0, 0.1, 0.2],
            audio_timesteps=[0.0, 0.2, 0.4],
            imgvid_cond_noise_aug=0.999,
            audio_ref_cond_noise_aug=1.0,
        )
        assert repeated_plan[1][1] is repeated_plan[2][1]
        assert repeated_plan[1][2] is repeated_plan[2][2]


def test_inplace_target_update_matches_scheduler_math():
    generator = torch.Generator().manual_seed(7)
    for sigma_curr, sigma_next in ((1.0, 0.7), (0.2, 0.0), (0.0, 0.0)):
        state = torch.randn(11, 32, generator=generator)
        velocity = torch.randn(11, 32, generator=generator)
        timestep = torch.tensor(1.0 - sigma_curr)
        ratio = torch.tensor(0.0 if sigma_curr == 0.0 else sigma_next / sigma_curr)
        denoised = _minimax_h3_rf_v_to_x0(state, velocity, timestep)
        expected = _minimax_h3_euler_eta0_step(
            state,
            denoised,
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
            sigma_ratio=ratio,
        )

        actual = state.clone()
        velocity_scratch = velocity.clone()
        _minimax_h3_update_target_rows_(
            actual,
            velocity_scratch,
            sigma_t=1.0 - timestep,
            sigma_curr=sigma_curr,
            sigma_ratio=ratio,
            one_minus_sigma_ratio=1.0 - ratio,
            denoised_scratch=torch.empty_like(actual),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_res_multistep_coefficients_preserve_constant_x0_euler_weight():
    sigmas = [1.0, 0.8, 0.5, 0.2, 0.0]
    coeffs = minimax_h3_res_multistep_coeffs(sigmas)

    assert coeffs[0] is None
    assert coeffs[-1] is None
    for step in (1, 2):
        hb1, hb2 = coeffs[step]
        assert hb1 + hb2 == pytest.approx(
            1.0 - sigmas[step + 1] / sigmas[step], abs=1e-15
        )


def test_res_multistep_schedule_matches_pinned_runninghub_golden():
    """Guard the full schedule port from RunningHub commit d6c5f7b."""

    payload = {}
    for points in (13, 15, 17, 21, 50):
        for shift in (3.0, 12.0):
            sigmas = minimax_h3_time_shift_sigmas(
                num_steps=points,
                shift_scale=shift,
            )
            payload[f"{points}:{shift:g}"] = {
                "sigmas": sigmas,
                "coeffs": minimax_h3_res_multistep_coeffs(sigmas),
            }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()

    assert hashlib.sha256(canonical).hexdigest() == (
        "756da5593fde7a7142060ff3e931dc1721512a37dc497b31ee694d2c15749562"
    )


def test_inplace_res_multistep_update_matches_reference_math():
    generator = torch.Generator().manual_seed(17)
    state = torch.randn(11, 32, generator=generator)
    velocity = torch.randn(11, 32, generator=generator)
    previous_denoised = torch.randn(11, 32, generator=generator)
    sigmas = [1.0, 0.75, 0.3, 0.0]
    hb1, hb2 = minimax_h3_res_multistep_coeffs(sigmas)[1]
    denoised = _minimax_h3_rf_v_to_x0(
        state,
        velocity,
        torch.tensor(1.0 - sigmas[1]),
    )
    sigma_ratio = torch.tensor(sigmas[2] / sigmas[1])
    # This is deliberately left-associative to match the pinned upstream
    # res_multistep expression, including its fp32 rounding points.
    expected = (
        sigma_ratio * state + hb1 * denoised + hb2 * previous_denoised
    )

    actual = state.clone()
    denoised_scratch = torch.empty_like(actual)
    _minimax_h3_res_multistep_update_target_rows_(
        actual,
        velocity.clone(),
        previous_denoised,
        sigma_t=torch.tensor(sigmas[1]),
        sigma_ratio=sigma_ratio,
        hb1=hb1,
        hb2=hb2,
        denoised_scratch=denoised_scratch,
    )

    torch.testing.assert_close(denoised_scratch, denoised, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _run_tiny_loop(*, sampler_mode: str | None, video_sigmas, audio_sigmas):
    branch = _branch("t2va")
    video = torch.linspace(-1.0, 1.0, branch.img_pos.numel() * 96).reshape(-1, 96)
    audio = torch.linspace(-0.5, 0.5, branch.audio_pos.numel() * 32).reshape(-1, 32)

    def model(**kwargs):
        video_state = kwargs["x"][0].index_select(0, branch.img_target_seq_idx)
        audio_state = kwargs["audio_x"][0].index_select(0, branch.audio_pos_dev)
        return video_state.square().mul(0.025).add(0.01), audio_state.mul(-0.15)

    kwargs = dict(
        model=model,
        positive=branch,
        initial_video_rows=video,
        initial_audio_rows=audio,
        keyframe_cond_rows=None,
        sigmas_video=video_sigmas,
        sigmas_audio=audio_sigmas,
        device=torch.device("cpu"),
    )
    if sampler_mode is not None:
        kwargs["sampler_mode"] = sampler_mode
    return minimax_h3_denoise_loop(**kwargs)


def test_default_sampler_is_bitwise_euler_and_res_history_is_request_local():
    video_sigmas = [1.0, 0.8, 0.5, 0.2, 0.0]
    audio_sigmas = [1.0, 0.65, 0.35, 0.1, 0.0]

    default = _run_tiny_loop(
        sampler_mode=None,
        video_sigmas=video_sigmas,
        audio_sigmas=audio_sigmas,
    )
    explicit = _run_tiny_loop(
        sampler_mode="euler",
        video_sigmas=video_sigmas,
        audio_sigmas=audio_sigmas,
    )
    first_res = _run_tiny_loop(
        sampler_mode="res_multistep",
        video_sigmas=video_sigmas,
        audio_sigmas=audio_sigmas,
    )
    second_res = _run_tiny_loop(
        sampler_mode="res_multistep",
        video_sigmas=video_sigmas,
        audio_sigmas=audio_sigmas,
    )

    for left, right in zip(default, explicit, strict=True):
        assert torch.equal(left, right)
    for left, right in zip(first_res, second_res, strict=True):
        assert torch.equal(left, right)
    assert not torch.equal(first_res[0], default[0])
    assert not torch.equal(first_res[1], default[1])


def test_res_multistep_full_loop_matches_independent_modality_recurrences():
    video_sigmas = [1.0, 0.82, 0.51, 0.23, 0.0]
    audio_sigmas = [1.0, 0.61, 0.29, 0.08, 0.0]
    branch = _branch("t2va")
    initial_video = torch.linspace(
        -1.0, 1.0, branch.img_pos.numel() * 96
    ).reshape(-1, 96)
    initial_audio = torch.linspace(
        -0.5, 0.5, branch.audio_pos.numel() * 32
    ).reshape(-1, 32)

    def velocity(state: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "video":
            return state.square().mul(0.025).add(0.01)
        return state.mul(-0.15)

    def reference(initial, sigmas, modality):
        state = initial.clone()
        previous_denoised = None
        coeffs = minimax_h3_res_multistep_coeffs(sigmas)
        for step, (sigma_curr, sigma_next) in enumerate(
            zip(sigmas[:-1], sigmas[1:], strict=True)
        ):
            current_velocity = velocity(state, modality)
            denoised = _minimax_h3_rf_v_to_x0(
                state,
                current_velocity,
                torch.tensor(1.0 - sigma_curr),
            )
            if coeffs[step] is None:
                state = _minimax_h3_euler_eta0_step(
                    state,
                    denoised,
                    sigma_curr=sigma_curr,
                    sigma_next=sigma_next,
                    sigma_ratio=torch.tensor(
                        0.0 if sigma_curr == 0.0 else sigma_next / sigma_curr
                    ),
                )
            else:
                assert previous_denoised is not None
                hb1, hb2 = coeffs[step]
                state = minimax_h3_res_multistep_eta0_step(
                    state,
                    denoised,
                    previous_denoised,
                    sigma_curr=sigma_curr,
                    sigma_next=sigma_next,
                    hb1=hb1,
                    hb2=hb2,
                )
            previous_denoised = denoised
        return state

    expected_video = reference(initial_video, video_sigmas, "video")
    expected_audio = reference(initial_audio, audio_sigmas, "audio")
    actual_video, actual_audio = _run_tiny_loop(
        sampler_mode="res_multistep",
        video_sigmas=video_sigmas,
        audio_sigmas=audio_sigmas,
    )

    torch.testing.assert_close(actual_video, expected_video, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_audio, expected_audio, rtol=1e-6, atol=1e-6)


def test_local_text_layout_is_a_contiguous_prefix_per_ulysses_rank():
    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        text_len = int(branch.static_kwargs["prompt_embeds"].shape[0])
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                layout = _build_local_embedding_layout(
                    seq_len=branch.seq_len,
                    text_pos=torch.arange(text_len),
                    img_pos=branch.img_pos,
                    audio_pos=branch.audio_pos,
                    world_size=world_size,
                    rank=rank,
                    device=torch.device("cpu"),
                )
                start = int(layout["text_source_start"])
                stop = int(layout["text_source_stop"])
                row_start = rank * (branch.seq_len // world_size)
                expected = torch.nonzero(
                    (torch.arange(text_len) >= row_start)
                    & (
                        torch.arange(text_len)
                        < row_start + branch.seq_len // world_size
                    )
                ).view(-1)
                assert expected.tolist() == list(range(start, stop))


def test_rank_local_token_tags_match_reference_slice():
    for mode in ("t2va", "fl2va", "ref2va"):
        seq_len = _branch(mode).seq_len
        token_tags = torch.arange(seq_len, dtype=torch.long) - seq_len // 2
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                with patch(
                    "sglang.multimodal_gen.runtime.pipelines_core.stages."
                    "model_specific_stages.minimax_h3.denoise_loop.get_ulysses_ctx",
                    return_value=(world_size, rank),
                ):
                    branch = _branch(mode, token_tags=token_tags)
                local_rows = branch.seq_len // world_size
                expected = token_tags[
                    rank * local_rows : (rank + 1) * local_rows
                ].clamp(min=0)
                torch.testing.assert_close(
                    branch.static_kwargs["block_token_tags"], expected, rtol=0, atol=0
                )


def test_debug_latent_dump_path_can_distinguish_grouped_requests():
    assert (
        _resolve_debug_latent_dump_path(
            "/tmp/h3-{seed}-{request_id}-{output_stem}.pt",
            seed=43,
            request_id="request-1",
            output_file_name="/tmp/cache-candidate.mp4",
        )
        == "/tmp/h3-43-request-1-cache-candidate.pt"
    )
    assert (
        _resolve_debug_latent_dump_path("/tmp/h3-{seed}.pt", seed=None, request_id=None)
        == "/tmp/h3-unknown.pt"
    )


def test_teacache_request_unmounts_active_cache_dit_hook():
    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
    stage.transformer = object()
    stage._cache_dit_enabled = True
    stage._cached_num_steps = 19
    stage._minimax_h3_cache_mode = "generic"
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(enable_teacache=True),
    )
    restored = object()

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.minimax_h3.stages.denoising."
        "disable_cache_on_transformer",
        return_value=restored,
    ) as disable:
        stage._maybe_enable_cache_dit(19, batch)

    disable.assert_called_once()
    assert stage.transformer is restored
    assert not stage._cache_dit_enabled
    assert stage._cached_num_steps is None
    assert stage._minimax_h3_cache_mode is None


def test_teacache_summary_is_copied_into_request_metrics():
    recorded = {}
    summary = {"computed_steps": [0, 2], "cached_steps": [1]}
    model = SimpleNamespace(minimax_h3_teacache_summary=lambda: summary)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(enable_teacache=True),
        metrics=SimpleNamespace(
            record_metadata=lambda name, value: recorded.update({name: value})
        ),
    )

    MiniMaxH3DenoisingStage._record_minimax_h3_teacache_metrics(model, batch)

    assert recorded == {"minimax_h3_teacache": summary}
