# SPDX-License-Identifier: Apache-2.0
"""High-value task, partition, and public request admission contracts."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    LAYERWISE_OFFLOAD,
    RESIDENT,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    partition_for_task,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.server_args.server_args import Backend

TARGET = {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 5.0,
}


def test_teacache_api_params_are_typed_and_request_scoped(monkeypatch):
    for name in (
        "SGLANG_H3_TEACACHE_THRESHOLD",
        "SGLANG_H3_TEACACHE_START",
        "SGLANG_H3_TEACACHE_END",
        "SGLANG_H3_TEACACHE_COEFFICIENTS",
    ):
        monkeypatch.delenv(name, raising=False)

    dense_before = MiniMaxH3SamplingParams()
    canary = MiniMaxH3SamplingParams(
        enable_teacache=True,
        teacache_params={
            "teacache_thresh": 0.08,
            "start_skipping": 15,
            "end_skipping": 16,
            "coefficients": [1.0, 0.0],
        },
    )
    dense_after = MiniMaxH3SamplingParams()

    assert dense_before.enable_teacache is False
    assert dense_after.enable_teacache is False
    assert dense_before.teacache_params.teacache_thresh == 0.0
    assert dense_after.teacache_params.teacache_thresh == 0.0
    assert isinstance(canary.teacache_params, TeaCacheParams)
    assert canary.teacache_params.teacache_thresh == pytest.approx(0.08)
    assert canary.teacache_params.get_skip_boundaries(19, False) == (15, 16)


@pytest.mark.parametrize(
    ("task", "conditions", "partition", "visual", "audio", "chains"),
    [
        ("t2va", [], "fl2va", [], [], []),
        (
            "fl2va",
            [
                {
                    "type": "image",
                    "uri": "file:///first.png",
                    "role": "keyframe",
                    "frame_index": 0,
                },
                {
                    "type": "image",
                    "uri": "file:///last.png",
                    "role": "keyframe",
                    "frame_index": -1,
                },
            ],
            "fl2va",
            [0, 1],
            [],
            ["image.target_canvas", "image.target_canvas"],
        ),
        (
            "ref2va",
            [
                {
                    "type": "image",
                    "uri": "file:///image.png",
                    "role": "reference",
                },
                {
                    "type": "video",
                    "uri": "file:///video.mp4",
                    "role": "reference",
                    "start_time_seconds": 12.5,
                },
                {
                    "type": "audio",
                    "uri": "file:///audio.wav",
                    "role": "reference",
                },
                {
                    "type": "video_audio",
                    "uri": "file:///av.mp4",
                    "role": "reference",
                },
            ],
            "ref2va",
            [0, 1, 3],
            [1, 2, 3],
            [
                "image.reference_preserve",
                "video.reference_preserve",
                "audio",
                "video_audio.reference_preserve",
            ],
        ),
        (
            "ref2va",
            [
                {
                    "type": "image",
                    "uri": "file:///first.png",
                    "role": "keyframe",
                    "frame_index": 0,
                },
                {
                    "type": "image",
                    "uri": "file:///subject.png",
                    "role": "reference",
                },
            ],
            "ref2va",
            [0, 1],
            [],
            ["image.target_canvas", "image.reference_preserve"],
        ),
    ],
)
def test_public_tasks_resolve_to_exact_partition_and_encoder_plan(
    task, conditions, partition, visual, audio, chains
):
    canonical = minimax_h3_validate_canonical_request(
        task=task,
        prompt="contract",
        conditions=conditions,
        target=TARGET,
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)

    assert partition_for_task(task) == partition
    assert plan.task == task
    assert plan.encoders["visual"] == visual
    assert plan.encoders["audio"] == audio
    assert [material.material_chain for material in plan.materials] == chains
    if task == "ref2va":
        assert plan.encoders["qwen"]["ordered_condition_indices"] == [
            index
            for index, condition in enumerate(conditions)
            if condition["role"] == "reference"
        ]
    for index, condition in enumerate(conditions):
        if condition.get("start_time_seconds") is not None:
            assert plan.materials[index].start_time_seconds == 12.5
    assert plan.shape["frame_count"] == 124
    assert plan.shape["video_latent_t"] == 37


def test_batch_admission_cost_uses_h3_packed_target_rows():
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="admission cost",
        conditions=[],
        target=TARGET,
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    batch = SimpleNamespace(
        extra={"minimax_h3_resolved_plan": plan},
        num_outputs_per_prompt=2,
    )

    # 37 * (768 / 32) * (1344 / 32) video rows + 207 * 2 audio rows.
    assert MiniMaxH3PipelineConfig().estimate_request_cost(batch) == 75420.0


def test_encoder_batch_admission_counts_right_padding_not_raw_token_sum():
    config = MiniMaxH3PipelineConfig()
    batches = [
        SimpleNamespace(
            extra={
                "minimax_h3_precomputed_presentation": {
                    "presentation_token_count": count
                }
            },
            num_outputs_per_prompt=1,
        )
        for count in (30000, 10000)
    ]

    assert config.estimate_encoder_batch_tokens(batches) == 60000
    batches[1].extra.clear()
    assert config.estimate_encoder_batch_tokens(batches) is None


def test_mixed_duration_requests_share_encoder_batch_signature():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig())

    def request(duration: float, frames: int):
        params = MiniMaxH3SamplingParams(
            prompt=f"duration {duration}",
            task="t2va",
            conditions=[],
            target={**TARGET, "duration_seconds": duration},
            num_inference_steps=2,
            flow_shift=12.0,
            audio_flow_shift=3.0,
        )
        params.width = 1344
        params.height = 768
        params.num_frames = frames
        return SimpleNamespace(
            sampling_params=params,
            profile=False,
            profile_all_stages=False,
            num_profiled_timesteps=5,
            extra={},
        )

    five_seconds = request(5.0, 124)
    fifteen_seconds = request(15.0, 362)

    assert scheduler._build_dynamic_batch_signature(
        five_seconds
    ) == scheduler._build_dynamic_batch_signature(fifteen_seconds)

    fifteen_seconds.sampling_params.num_inference_steps = 20
    assert scheduler._build_dynamic_batch_signature(
        five_seconds
    ) != scheduler._build_dynamic_batch_signature(fifteen_seconds)


def test_ref2va_rejects_keyframes_without_a_reference():
    with pytest.raises(ValueError, match="at least one reference"):
        minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="contract",
            conditions=[
                {
                    "type": "image",
                    "uri": "file:///first.png",
                    "role": "keyframe",
                    "frame_index": 0,
                }
            ],
            target=TARGET,
        )


@pytest.mark.parametrize(
    ("partition", "tasks"),
    [("fl2va", ["t2va", "fl2va"]), ("ref2va", ["ref2va"])],
)
def test_loaded_weight_partition_admits_only_its_declared_tasks(partition, tasks):
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": partition,
                "tasks": tasks,
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )

    assert [metadata.canonical_task(task) for task in tasks] == tasks
    rejected = "ref2va" if partition == "fl2va" else "t2va"
    with pytest.raises(ValueError):
        metadata.canonical_task(rejected)


def test_synthetic_warmup_target_honors_warmup_flags():
    def target(num_frames=None, resolution=None):
        width, height = map(int, (resolution or "896x512").split("x"))
        req = SimpleNamespace(num_frames=17, width=width, height=height)
        server_args = SimpleNamespace(
            warmup_num_frames=num_frames,
            warmup_resolutions=None if resolution is None else [resolution],
        )
        return MiniMaxH3SamplingParams._synthetic_warmup_target(req, server_args)

    assert target() == TARGET
    assert target(num_frames=345) == {**TARGET, "duration_seconds": 345 / 24.0}
    assert target(resolution="768x1344") == {**TARGET, "aspect_ratio": "9:16"}
    assert target(resolution="832x464") == TARGET


def test_duration_admission_accepts_released_4_to_15_second_range():
    for duration in (4.0, 15.0):
        target = {**TARGET, "duration_seconds": duration}
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="duration contract",
            conditions=[],
            target=target,
            seed=0,
        )
        assert canonical["target"]["duration_seconds"] == duration

    for duration in (3.9, 15.1):
        target = {**TARGET, "duration_seconds": duration}
        with pytest.raises(ValueError, match=r"\[4, 15\]"):
            minimax_h3_validate_canonical_request(
                task="t2va",
                prompt="duration contract",
                conditions=[],
                target=target,
                seed=0,
            )


def test_video_adapter_lowers_only_native_fields_and_rejects_cfg():
    request = VideoGenerationsRequest(
        prompt="contract",
        task="t2va",
        conditions=[],
        target=TARGET,
        flow_shift=8.0,
        audio_flow_shift=2.0,
        quality="high",
        imgvid_cond_noise_aug_for_inference=0.75,
        audio_cond_noise_aug_for_inference=0.5,
    )
    generic = {
        "prompt": request.prompt,
        "seed": request.seed,
        "flow_shift": request.flow_shift,
    }

    lowered = MiniMaxH3SamplingParams.lower_video_request_kwargs(request, generic)
    assert lowered == {
        "prompt": "contract",
        "seed": request.seed,
        "task": "t2va",
        "conditions": [],
        "target": TARGET,
        "flow_shift": 8.0,
        "audio_flow_shift": 2.0,
        "quality": "high",
        "sampler_mode": "euler",
        "imgvid_cond_noise_aug_for_inference": 0.75,
        "audio_cond_noise_aug_for_inference": 0.5,
    }

    with pytest.raises(ValueError):
        MiniMaxH3SamplingParams.lower_video_request_kwargs(
            request, {**generic, "guidance_scale": 7.5}
        )


@pytest.mark.parametrize("bad_quality", ["ultra", "draft", "", 1])
def test_video_adapter_rejects_invalid_quality(bad_quality):
    request = VideoGenerationsRequest(
        prompt="contract",
        task="t2va",
        conditions=[],
        target=TARGET,
        quality=bad_quality,
    )
    with pytest.raises(ValueError, match="quality must be one of"):
        MiniMaxH3SamplingParams.lower_video_request_kwargs(
            request, {"prompt": request.prompt, "seed": request.seed}
        )


class _HopperCapability:
    def to_int(self) -> int:
        return 90


def _quality_server_args():
    return SimpleNamespace(
        attention_backend=None,
        model_variant="fl2va",
        num_gpus=4,
        backend=Backend.AUTO,
        component_attention_backends={},
        enable_breakable_cuda_graph=False,
        enable_torch_compile=False,
        is_dit_layerwise_offload_selected=False,
        minimax_h3_adaln_online=False,
        performance_mode="speed",
        quantization=None,
        transformer_weights_path=None,
        regional_compile=False,
        ring_degree=1,
        sp_degree=4,
        tp_size=1,
        ulysses_degree=4,
        use_fsdp_inference=False,
        lora_path=None,
        lora_scale=1.0,
        lora_merge_mode="auto",
    )


def test_high_quality_deployment_rejects_transformer_weight_override():
    config = MiniMaxH3PipelineConfig()
    server_args = _quality_server_args()
    server_args.transformer_weights_path = "model.gguf"

    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "get_device_name", return_value="NVIDIA H200"),
        patch.object(
            current_platform,
            "get_device_capability",
            return_value=_HopperCapability(),
        ),
        pytest.raises(ValueError, match="transformer_weights_path"),
    ):
        config.validate_quality_deployment(server_args)


def test_high_quality_request_warns_when_bcg_suppresses_cache_dit():
    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
    stage.server_args = SimpleNamespace(enable_breakable_cuda_graph=True)
    stage._cache_dit_enabled = False
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(
            quality="high",
            _explicit_fields={"quality"},
            enable_cache_dit=None,
            cache_dit_params=None,
        )
    )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
        "logger.warning_once"
    ) as warning_once:
        stage._maybe_enable_cache_dit(50, batch)

    warning_once.assert_called_once_with(
        "Cache-DiT was requested but is disabled because breakable CUDA graphs "
        "are enabled."
    )


def test_admission_rejects_steps_exceeding_online_adaln_gpu_plans():
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": "fl2va",
                "tasks": ["t2va", "fl2va"],
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )
    stage = MiniMaxH3PartitionAdmissionStage(metadata)
    server_args = _quality_server_args()
    server_args.minimax_h3_adaln_online = True
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(task="t2va", quality="lossless"),
        num_inference_steps=50,
        is_warmup=False,
    )
    with patch.dict(os.environ, {"SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS": "8"}):
        with pytest.raises(
            ValueError, match="SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS"
        ):
            stage.forward(batch, server_args)

        batch.num_inference_steps = 9
        assert stage.forward(batch, server_args) is batch


def test_extra_high_quality_does_not_enable_h3_cache_dit():
    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
    stage.server_args = SimpleNamespace(enable_breakable_cuda_graph=False)
    stage._cache_dit_enabled = False
    stage._minimax_h3_cache_mode = None
    stage._minimax_h3_quality = "lossless"
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(
            quality="extra-high",
            _explicit_fields={"quality"},
            enable_cache_dit=None,
            cache_dit_params=None,
        )
    )

    # Even a server-wide generic Cache-DiT default must not turn an explicit
    # fusion-only quality tier into an approximate H3 request.
    with patch.object(DenoisingStage, "_cache_dit_requested", return_value=True):
        stage._maybe_enable_cache_dit(50, batch)

    assert stage._minimax_h3_quality == "extra-high"
    assert stage._minimax_h3_cache_mode is None
    assert not stage._cache_dit_enabled


def _res_cache_stage(*, cache_active: bool = False):
    preservation = []
    transformer = SimpleNamespace(set_cache_dit_input_preservation=preservation.append)
    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
    stage.server_args = SimpleNamespace(enable_breakable_cuda_graph=False)
    stage.transformer = transformer
    stage._cache_dit_enabled = cache_active
    stage._minimax_h3_cache_mode = "generic" if cache_active else None
    return stage, preservation


def _res_cache_batch(*, teacache=False, cache_override=None):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(
            quality="fast",
            sampler_mode="res_multistep",
            enable_teacache=teacache,
            enable_cache_dit=cache_override,
        )
    )


def test_res_multistep_rejects_teacache_and_request_cache_dit():
    stage, _ = _res_cache_stage()
    with pytest.raises(ValueError, match="TeaCache"):
        stage._maybe_enable_cache_dit(
            12,
            _res_cache_batch(teacache=True, cache_override=False),
        )
    with pytest.raises(ValueError, match="Cache-DiT"):
        stage._maybe_enable_cache_dit(
            12,
            _res_cache_batch(cache_override=True),
        )


def test_res_multistep_explicit_cache_opt_out_unmounts_previous_request():
    stage, preservation = _res_cache_stage(cache_active=True)

    def unmount():
        stage._cache_dit_enabled = False

    with (
        patch.object(stage, "_unmount_cache_dit", side_effect=unmount) as unmount_mock,
        patch.object(DenoisingStage, "_cache_dit_requested", return_value=True),
    ):
        stage._maybe_enable_cache_dit(
            12,
            _res_cache_batch(cache_override=False),
        )

    unmount_mock.assert_called_once_with()
    assert stage._minimax_h3_cache_mode is None
    assert preservation == [False]


def test_res_multistep_rejects_server_default_cache_dit():
    stage, _ = _res_cache_stage()
    with (
        patch.object(DenoisingStage, "_cache_dit_requested", return_value=True),
        pytest.raises(ValueError, match="Cache-DiT"),
    ):
        stage._maybe_enable_cache_dit(
            12,
            _res_cache_batch(cache_override=None),
        )


def test_quality_admission_fails_closed_outside_validated_request():
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": "fl2va",
                "tasks": ["t2va", "fl2va"],
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="quality",
        conditions=[],
        target=TARGET,
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(task="t2va", quality="high"),
        num_inference_steps=50,
        is_warmup=False,
    )
    stage = MiniMaxH3PartitionAdmissionStage(metadata)
    config = MiniMaxH3PipelineConfig()
    server_args = _quality_server_args()
    server_args.pipeline_config = config

    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "get_device_name", return_value="NVIDIA H200"),
        patch.object(
            current_platform,
            "get_device_capability",
            return_value=_HopperCapability(),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            "minimax_h3.release_metadata.minimax_h3_plan_from_batch",
            return_value=plan,
        ),
    ):
        assert stage.forward(batch, server_args) is batch
        batch.num_inference_steps = 40
        with pytest.raises(ValueError, match="validated only"):
            stage.forward(batch, server_args)

    batch.sampling_params.quality = "lossless"
    batch.num_inference_steps = 50
    server_args.attention_backend = "sage_attn"
    assert stage.forward(batch, server_args) is batch

    batch.sampling_params.quality = "extra-high"
    assert stage.forward(batch, server_args) is batch

    batch.sampling_params.quality = "ultra"
    server_args.attention_backend = None
    with pytest.raises(ValueError, match="quality must be one of"):
        stage.forward(batch, server_args)


def test_fast_quality_binds_startup_lora_and_canary_workload():
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": "fl2va",
                "tasks": ["t2va", "fl2va"],
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="fast canary",
        conditions=[],
        target=TARGET,
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(task="t2va", quality="fast"),
        num_inference_steps=7,
        is_warmup=False,
    )
    stage = MiniMaxH3PartitionAdmissionStage(metadata)
    server_args = _quality_server_args()
    server_args.pipeline_config = MiniMaxH3PipelineConfig()

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "minimax_h3.release_metadata.minimax_h3_plan_from_batch",
        return_value=plan,
    ):
        with pytest.raises(ValueError, match="requires an explicitly configured"):
            stage.forward(batch, server_args)

        server_args.lora_path = "/adapter.safetensors"
        server_args.lora_merge_mode = "dynamic"
        assert stage.forward(batch, server_args) is batch

        batch.sampling_params.quality = "lossless"
        with pytest.raises(ValueError, match='requires quality="fast"'):
            stage.forward(batch, server_args)

        batch.sampling_params.quality = "fast"
        batch.num_inference_steps = 8
        with pytest.raises(ValueError, match="canary is validated only"):
            stage.forward(batch, server_args)

        batch.num_inference_steps = 9
        server_args.lora_scale = 0.5
        with pytest.raises(ValueError, match="requires lora_scale=1.0"):
            stage.forward(batch, server_args)


def test_res_multistep_is_explicit_request_scoped_fast_canary(monkeypatch):
    with pytest.raises(ValueError, match="experimental canary"):
        MiniMaxH3SamplingParams(
            sampler_mode="res_multistep",
            quality="fast",
        )

    monkeypatch.setenv("SGLANG_H3_EXPERIMENTAL_RES_MULTISTEP", "1")
    params = MiniMaxH3SamplingParams(
        sampler_mode="res_multistep",
        quality="fast",
    )
    assert params.sampler_mode == "res_multistep"

    with pytest.raises(ValueError, match='requires quality="fast"'):
        MiniMaxH3SamplingParams(sampler_mode="res_multistep")
    with pytest.raises(ValueError, match="cannot be combined"):
        MiniMaxH3SamplingParams(
            sampler_mode="res_multistep",
            quality="fast",
            enable_teacache=True,
        )


def test_res_multistep_admission_binds_candidate_grid_and_excludes_lora(monkeypatch):
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": "fl2va",
                "tasks": ["t2va", "fl2va"],
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="solver canary",
        conditions=[],
        target={**TARGET, "duration_seconds": 15.0},
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(
            task="t2va",
            quality="fast",
            sampler_mode="res_multistep",
        ),
        num_inference_steps=13,
        is_warmup=False,
    )
    stage = MiniMaxH3PartitionAdmissionStage(metadata)
    server_args = _quality_server_args()
    server_args.pipeline_config = MiniMaxH3PipelineConfig()

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "minimax_h3.release_metadata.minimax_h3_plan_from_batch",
        return_value=plan,
    ):
        with pytest.raises(ValueError, match="canary is disabled"):
            stage.forward(batch, server_args)

        monkeypatch.setenv("SGLANG_H3_EXPERIMENTAL_RES_MULTISTEP", "1")
        assert stage.forward(batch, server_args) is batch

        batch.num_inference_steps = 14
        with pytest.raises(ValueError, match="admitted only"):
            stage.forward(batch, server_args)

        batch.num_inference_steps = 15
        server_args.lora_path = "/adapter.safetensors"
        with pytest.raises(ValueError, match="cannot be combined"):
            stage.forward(batch, server_args)


def test_validate_server_args_requires_packed_varlen_backend():
    config = MiniMaxH3PipelineConfig()
    server_args = SimpleNamespace(
        component_attention_backends={},
        attention_backend="sage_attn",
        ring_degree=1,
        resolve_component_attention_backend=lambda *_names: (None, None),
    )
    with patch(
        "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3.get_attn_backend"
    ) as get_attn_backend:
        MiniMaxH3PipelineConfig.validate_server_args(config, server_args)
    get_attn_backend.assert_called_once_with(
        128,
        torch.bfloat16,
        selected_attention_backend=AttentionBackendEnum.SAGE_ATTN,
        attention_requirements=AttentionRequirements(packed_varlen=True),
    )
    with patch(
        "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3.get_attn_backend",
        side_effect=ValueError("does not implement packed varlen attention"),
    ):
        with pytest.raises(ValueError, match="does not implement packed varlen"):
            MiniMaxH3PipelineConfig.validate_server_args(config, server_args)

    server_args.component_attention_backends = {"transformer": "cube_sparse_attn"}
    server_args.resolve_component_attention_backend = lambda *_names: (
        AttentionBackendEnum.CUBE_SPARSE_ATTN,
        "transformer",
    )
    server_args.ring_degree = 2
    with pytest.raises(ValueError, match="ring parallelism requires"):
        MiniMaxH3PipelineConfig.validate_server_args(config, server_args)


def test_validate_server_args_accepts_transformer_backend_override():
    config = MiniMaxH3PipelineConfig()
    server_args = SimpleNamespace(
        component_attention_backends={"transformer": "subblock_sparse_attn"},
        attention_backend="fa",
        ring_degree=1,
        resolve_component_attention_backend=lambda *_names: (
            AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN,
            "transformer",
        ),
    )

    with patch(
        "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3.get_attn_backend"
    ) as get_attn_backend:
        MiniMaxH3PipelineConfig.validate_server_args(config, server_args)
    get_attn_backend.assert_called_once_with(
        128,
        torch.bfloat16,
        selected_attention_backend=AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN,
        attention_requirements=AttentionRequirements(packed_varlen=True),
    )


def test_resolve_transformer_attention_backend_uses_selector_precedence():
    config = MiniMaxH3PipelineConfig()
    subblock = AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN
    fa = AttentionBackendEnum.FA
    sdpa = AttentionBackendEnum.TORCH_SDPA
    cases = (
        ("fa", subblock, None, subblock),
        ("subblock_sparse_attn", fa, None, fa),
        (subblock, None, None, subblock),
        ("fa", subblock, sdpa, sdpa),
    )
    for global_backend, component_backend, forced_backend, expected in cases:
        server_args = SimpleNamespace(
            attention_backend=global_backend,
            resolve_component_attention_backend=lambda *_names: (
                component_backend,
                "transformer" if component_backend is not None else None,
            ),
        )
        with patch(
            "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3."
            "get_global_forced_attn_backend",
            return_value=forced_backend,
        ):
            resolved = config.resolve_transformer_attention_backend(server_args)
            assert resolved is expected
            assert config.uses_subblock_attention(server_args) is (
                expected is AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN
            )


def test_mps_admission_requires_layerwise_residency_for_every_h3_component():
    config = MiniMaxH3PipelineConfig()
    modes = {
        "transformer": LAYERWISE_OFFLOAD,
        "text_encoder": LAYERWISE_OFFLOAD,
        "video_vae": LAYERWISE_OFFLOAD,
        "audio_vae": LAYERWISE_OFFLOAD,
    }
    server_args = SimpleNamespace(
        component_attention_backends={},
        attention_backend=None,
        enable_torch_compile=False,
        ring_degree=1,
        residency_mode=modes.get,
        resolve_component_attention_backend=lambda *_names: (None, None),
    )

    with patch.object(current_platform, "is_mps", return_value=True):
        MiniMaxH3PipelineConfig.validate_server_args(config, server_args)

        modes["audio_vae"] = RESIDENT
        with pytest.raises(ValueError, match="audio_vae"):
            MiniMaxH3PipelineConfig.validate_server_args(config, server_args)
