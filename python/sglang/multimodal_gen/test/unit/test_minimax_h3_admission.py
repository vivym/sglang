# SPDX-License-Identifier: Apache-2.0
"""High-value task, partition, and public request admission contracts."""

from __future__ import annotations

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
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
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
        assert plan.materials[1].start_time_seconds == 12.5
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
        performance_mode="speed",
        quantization=None,
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


def test_validate_server_args_requires_packed_varlen_backend():
    config = SimpleNamespace(
        vae_config=SimpleNamespace(resolved_parallel_decode_mode=lambda: None),
        dit_config=SimpleNamespace(arch_config=SimpleNamespace(attention_head_dim=128)),
        _server_arg_value=MiniMaxH3PipelineConfig._server_arg_value,
        _force_vae_resident=lambda _server_args: None,
    )
    server_args = SimpleNamespace(
        component_attention_backends={}, attention_backend="sage_attn"
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
