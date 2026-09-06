from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import (
    RoleType,
    filter_modules_for_role,
)
from sglang.multimodal_gen.runtime.disaggregation import (
    scheduler_mixin as scheduler_mixin_module,
)
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import unpack_tensors
from sglang.multimodal_gen.runtime.pipelines import minimax_h3_pipeline as h3_pipeline
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.disagg_boundary import (
    MINIMAX_H3_DISAGG_RELEASE_IDENTITY_FIELD,
    MINIMAX_H3_DISAGG_SCHEMA_FIELD,
    MINIMAX_H3_DISAGG_TEXT_TAGS_FIELD,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
    MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
    minimax_h3_plan_from_batch,
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.latent_preparation import (
    MiniMaxH3LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)


_TEST_RELEASE_IDENTITY = {
    "schema": "minimax-h3.disagg-release/v1",
    "manifest_content_sha256": "sha256:" + "1" * 64,
    "partition": "fl2va",
}


def _pipeline(release_identity=None):
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.disagg_release_identity = dict(release_identity or _TEST_RELEASE_IDENTITY)
    return pipeline


def _scheduler(role: RoleType, release_identity=None):
    return SimpleNamespace(
        worker=SimpleNamespace(pipeline=_pipeline(release_identity)),
        _disagg_role=role,
    )


def _t2va_req(*, text_len: int = 3) -> Req:
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="a production test",
        conditions=[],
        target={
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        },
        seed=42,
    )
    plan = minimax_h3_resolve_plan(canonical)
    sampling = MiniMaxH3SamplingParams(
        prompt=canonical["prompt"],
        task="t2va",
        conditions=[],
        target=dict(canonical["target"]),
        seed=42,
        num_inference_steps=20,
        output_path="/tmp/h3-disagg-test",
        output_file_name="test.mp4",
    )
    sampling._explicit_fields = {"task", "target", "num_inference_steps"}
    req = Req(request_id="h3-boundary", sampling_params=sampling)
    req.width = int(plan.shape["width"])
    req.height = int(plan.shape["height"])
    req.fps = int(plan.shape["fps"])
    req.num_frames = int(plan.shape["frame_count"])
    req.generator = torch.Generator(device="cpu").manual_seed(42)
    req.extra[MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY] = canonical
    req.extra[MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = plan
    hidden = (
        torch.arange(text_len * 5120, dtype=torch.float32)
        .reshape(text_len, 5120)
        .to(torch.bfloat16)
    )
    tags = torch.ones(text_len, dtype=torch.int64)
    req.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY] = {
        "positive": {
            "hidden_states": hidden,
            "text_len": text_len,
            "text_token_tags": tags,
        }
    }
    req.prompt_embeds = [hidden]
    req.prompt_seq_lens = [[text_len]]
    return req


def _round_trip(req: Req, *, source: RoleType, destination: RoleType) -> Req:
    sender = _scheduler(source)
    tensors, scalars = SchedulerDisaggMixin._extract_disagg_transfer_fields(
        sender,
        req,
        source_role=source,
        destination_role=destination,
    )
    receiver = _scheduler(destination)
    return SchedulerDisaggMixin._build_disagg_req(
        receiver, dict(scalars), dict(tensors)
    )


def test_h3_decoder_without_media_service_fails_before_vae_compute():
    sent = []

    class _Socket:
        def send_multipart(self, frames, **_kwargs):
            sent.append(frames)

    class _Worker:
        pipeline = _pipeline()

        def execute_forward(self, _reqs):
            raise AssertionError("VAE compute must not run without media service")

    scheduler = SimpleNamespace(
        worker=_Worker(),
        _media_encode_queue=None,
        _pool_result_push=_Socket(),
        _disagg_metrics=None,
    )
    SchedulerDisaggMixin._disagg_decoder_compute(
        scheduler,
        _t2va_req(),
        "h3-no-media",
        "decoder-attempt",
    )

    tensors, scalars = unpack_tensors(sent[0])
    assert tensors == {}
    assert scalars["_transfer_id"] == "decoder-attempt"
    assert "raw frame return is forbidden" in scalars["error"]


def test_h3_encoder_to_denoiser_boundary_round_trip():
    req = _t2va_req()
    rebuilt = _round_trip(req, source=RoleType.ENCODER, destination=RoleType.DENOISER)

    expected = req.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]["positive"]
    actual = rebuilt.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]["positive"]
    assert isinstance(rebuilt.sampling_params, MiniMaxH3SamplingParams)
    assert rebuilt.sampling_params.sampler_mode == "euler"
    assert rebuilt.sampling_params._explicit_fields == {
        "task",
        "target",
        "num_inference_steps",
    }
    assert torch.equal(actual["hidden_states"], expected["hidden_states"])
    assert torch.equal(actual["text_token_tags"], expected["text_token_tags"])
    assert actual["text_len"] == expected["text_len"]
    assert rebuilt.prompt_embeds[0] is actual["hidden_states"]
    assert minimax_h3_plan_from_batch(rebuilt) == minimax_h3_plan_from_batch(req)
    assert all(not key.startswith("_extra_") for key in rebuilt.__dict__)


def test_h3_denoiser_to_decoder_boundary_only_keeps_final_latents():
    req = _round_trip(
        _t2va_req(), source=RoleType.ENCODER, destination=RoleType.DENOISER
    )
    plan = minimax_h3_plan_from_batch(req)
    req.latents = torch.zeros(
        1,
        24,
        int(plan.shape["video_latent_t"]),
        int(plan.shape["height"]) // 16,
        int(plan.shape["width"]) // 16,
        dtype=torch.float32,
    )
    req.audio_latents = torch.zeros(
        2, 32, int(plan.shape["audio_latent_t"]), dtype=torch.float32
    )
    req.timesteps = torch.arange(19, dtype=torch.float32)

    sender = _scheduler(RoleType.DENOISER)
    tensors, scalars = SchedulerDisaggMixin._extract_disagg_transfer_fields(
        sender,
        req,
        source_role=RoleType.DENOISER,
        destination_role=RoleType.DECODER,
    )
    assert set(tensors) == {"latents", "audio_latents"}
    assert all(not key.startswith("_extra_") for key in scalars)

    rebuilt = SchedulerDisaggMixin._build_disagg_req(
        _scheduler(RoleType.DECODER), dict(scalars), dict(tensors)
    )
    assert torch.equal(rebuilt.latents, req.latents)
    assert torch.equal(rebuilt.audio_latents, req.audio_latents)
    assert rebuilt.timesteps is None
    assert minimax_h3_plan_from_batch(rebuilt) == plan


def test_h3_boundary_rejects_non_finite_text_hidden_states():
    req = _t2va_req()
    req.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]["positive"]["hidden_states"][
        0, 0
    ] = float("nan")

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        SchedulerDisaggMixin._extract_disagg_transfer_fields(
            _scheduler(RoleType.ENCODER),
            req,
            source_role=RoleType.ENCODER,
            destination_role=RoleType.DENOISER,
        )


def test_h3_boundary_rejects_missing_text_tags():
    req = _t2va_req()
    req.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]["positive"].pop("text_token_tags")

    with pytest.raises(ValueError, match="text token tags must be a tensor"):
        SchedulerDisaggMixin._extract_disagg_transfer_fields(
            _scheduler(RoleType.ENCODER),
            req,
            source_role=RoleType.ENCODER,
            destination_role=RoleType.DENOISER,
        )


def test_h3_boundary_rejects_unknown_schema_on_restore():
    req = _t2va_req()
    tensors, scalars = SchedulerDisaggMixin._extract_disagg_transfer_fields(
        _scheduler(RoleType.ENCODER),
        req,
        source_role=RoleType.ENCODER,
        destination_role=RoleType.DENOISER,
    )
    scalars[MINIMAX_H3_DISAGG_SCHEMA_FIELD] = "unknown/v99"

    with pytest.raises(ValueError, match="unsupported.*boundary schema"):
        SchedulerDisaggMixin._build_disagg_req(
            _scheduler(RoleType.DENOISER), scalars, tensors
        )


def test_h3_boundary_rejects_release_identity_mismatch():
    req = _t2va_req()
    tensors, scalars = SchedulerDisaggMixin._extract_disagg_transfer_fields(
        _scheduler(RoleType.ENCODER),
        req,
        source_role=RoleType.ENCODER,
        destination_role=RoleType.DENOISER,
    )
    assert scalars[MINIMAX_H3_DISAGG_RELEASE_IDENTITY_FIELD] == (_TEST_RELEASE_IDENTITY)
    incompatible = {
        **_TEST_RELEASE_IDENTITY,
        "manifest_content_sha256": "sha256:" + "2" * 64,
    }

    with pytest.raises(ValueError, match="release identity mismatch"):
        SchedulerDisaggMixin._build_disagg_req(
            _scheduler(RoleType.DENOISER, incompatible), scalars, tensors
        )


def test_h3_boundary_rejects_incomplete_tensor_contract_on_restore():
    req = _t2va_req()
    tensors, scalars = SchedulerDisaggMixin._extract_disagg_transfer_fields(
        _scheduler(RoleType.ENCODER),
        req,
        source_role=RoleType.ENCODER,
        destination_role=RoleType.DENOISER,
    )
    tensors.pop(MINIMAX_H3_DISAGG_TEXT_TAGS_FIELD)

    with pytest.raises(ValueError, match="incomplete or unknown"):
        SchedulerDisaggMixin._build_disagg_req(
            _scheduler(RoleType.DENOISER), scalars, tensors
        )


def test_h3_boundary_v1_rejects_reference_media():
    req = _t2va_req()
    req.extra[MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY]["conditions"] = [
        {"type": "image", "role": "reference", "uri": "/tmp/reference.png"}
    ]

    with pytest.raises(ValueError, match="text-only t2va"):
        SchedulerDisaggMixin._extract_disagg_transfer_fields(
            _scheduler(RoleType.ENCODER),
            req,
            source_role=RoleType.ENCODER,
            destination_role=RoleType.DENOISER,
        )


def test_h3_pipeline_enables_only_compute_roles():
    assert MiniMaxH3PipelineConfig.supports_disaggregation(None)
    for role in (
        RoleType.MONOLITHIC,
        RoleType.ENCODER,
        RoleType.DENOISER,
        RoleType.DECODER,
    ):
        MiniMaxH3Pipeline.validate_disagg_role(None, role)
    with pytest.raises(ValueError, match="not supported"):
        MiniMaxH3Pipeline.validate_disagg_role(None, RoleType.SERVER)


def test_h3_preparation_stages_belong_to_denoiser_role():
    assert MiniMaxH3LatentPreparationStage.role_affinity.fget(None) is RoleType.DENOISER
    assert (
        MiniMaxH3TimestepPreparationStage.role_affinity.fget(None) is RoleType.DENOISER
    )


@pytest.mark.parametrize(
    ("role", "expected_modules"),
    [
        (RoleType.ENCODER, ["processor", "text_encoder", "tokenizer"]),
        (RoleType.DENOISER, ["transformer"]),
        (RoleType.DECODER, ["video_vae", "audio_vae"]),
    ],
)
def test_h3_role_loads_only_owned_components(role, expected_modules):
    assert (
        filter_modules_for_role(MiniMaxH3Pipeline._required_config_modules, role)
        == expected_modules
    )


@pytest.mark.parametrize(
    ("role", "expected_stages"),
    [
        (
            RoleType.ENCODER,
            [
                "InputValidationStage",
                "MiniMaxH3PartitionAdmissionStage",
                "MiniMaxH3TextEncodingStage",
                "MiniMaxH3VisualEncodingStage",
                "MiniMaxH3AudioEncodingStage",
            ],
        ),
        (
            RoleType.DENOISER,
            [
                "MiniMaxH3LatentPreparationStage",
                "MiniMaxH3TimestepPreparationStage",
                "MiniMaxH3DenoisingStage",
            ],
        ),
        (RoleType.DECODER, ["MiniMaxH3DecodingStage"]),
    ],
)
def test_h3_role_constructs_only_owned_stages(monkeypatch, role, expected_stages):
    stage_roles = {
        "InputValidationStage": RoleType.ENCODER,
        "MiniMaxH3PartitionAdmissionStage": RoleType.ENCODER,
        "MiniMaxH3TextEncodingStage": RoleType.ENCODER,
        "MiniMaxH3VisualEncodingStage": RoleType.ENCODER,
        "MiniMaxH3AudioEncodingStage": RoleType.ENCODER,
        "MiniMaxH3LatentPreparationStage": RoleType.DENOISER,
        "MiniMaxH3TimestepPreparationStage": RoleType.DENOISER,
        "MiniMaxH3DenoisingStage": RoleType.DENOISER,
        "MiniMaxH3DecodingStage": RoleType.DECODER,
    }

    class FakeStage:
        def __init__(self, affinity):
            self.role_affinity = affinity

        def set_registered_stage_name(self, stage_name):
            self._registered_stage_name = stage_name

        def set_profile_stage_name(self, stage_name):
            self._profile_stage_name = stage_name

    for stage_name, affinity in stage_roles.items():

        def factory(*args, _affinity=affinity, **kwargs):
            del args, kwargs
            return FakeStage(_affinity)

        factory.__name__ = stage_name
        monkeypatch.setattr(h3_pipeline, stage_name, factory)

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline._disagg_role = role
    pipeline.modules = {
        name: object()
        for name in filter_modules_for_role(
            MiniMaxH3Pipeline._required_config_modules, role
        )
    }
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline.release_metadata = SimpleNamespace(sigma_shift_scales=None)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(arch_config=None),
            audio_vae_config=SimpleNamespace(arch_config=None),
        )
    )

    pipeline.create_pipeline_stages(server_args)

    assert list(pipeline._stage_name_mapping) == expected_stages


def test_h3_disagg_v1_rejects_reference_task_during_admission():
    metadata = MiniMaxH3ReleaseMetadata(
        schema_version=1,
        partition="ref2va",
        tasks=("ref2va",),
        task_aliases={},
        video_sigma_shift=12.0,
        audio_sigma_shift=3.0,
    )
    stage = SimpleNamespace(metadata=metadata)
    batch = SimpleNamespace(sampling_params=SimpleNamespace(task="ref2va"))
    server_args = SimpleNamespace(disagg_role=RoleType.ENCODER)

    with pytest.raises(ValueError, match="boundary v1 supports text-only t2va"):
        MiniMaxH3PartitionAdmissionStage.forward(stage, batch, server_args)


@pytest.mark.parametrize("role", [RoleType.DENOISER, RoleType.DECODER])
def test_h3_multi_rank_broadcast_preserves_boundary_state(monkeypatch, role):
    class BroadcastChannel:
        scalar_fields = None
        tensor_fields = None

    class BroadcastScheduler(SchedulerDisaggMixin):
        def __init__(self, gpu_id):
            self.gpu_id = gpu_id
            self._disagg_role = role
            self.server_args = SimpleNamespace(
                sp_degree=2,
                tp_size=1,
                enable_cfg_parallel=False,
            )
            self.worker = SimpleNamespace(
                pipeline=_pipeline(),
                local_rank=gpu_id,
            )

        def _broadcast_to_all_ranks(self, data):
            if self.gpu_id == 0:
                BroadcastChannel.scalar_fields = data
                return data
            return BroadcastChannel.scalar_fields

        def _broadcast_tensor_dict_to_all_ranks(self, data):
            if self.gpu_id == 0:
                BroadcastChannel.tensor_fields = data
                return data
            return BroadcastChannel.tensor_fields

    req = _round_trip(
        _t2va_req(), source=RoleType.ENCODER, destination=RoleType.DENOISER
    )
    plan = minimax_h3_plan_from_batch(req)
    if role is RoleType.DECODER:
        req.latents = torch.zeros(
            1,
            24,
            int(plan.shape["video_latent_t"]),
            int(plan.shape["height"]) // 16,
            int(plan.shape["width"]) // 16,
            dtype=torch.float32,
        )
        req.audio_latents = torch.zeros(
            2, 32, int(plan.shape["audio_latent_t"]), dtype=torch.float32
        )
        req = _round_trip(req, source=RoleType.DENOISER, destination=RoleType.DECODER)

    monkeypatch.setattr(
        scheduler_mixin_module,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    root = BroadcastScheduler(gpu_id=0)
    receiver = BroadcastScheduler(gpu_id=1)

    assert root._broadcast_req_to_all_ranks(req) is req
    rebuilt = receiver._broadcast_req_to_all_ranks(None)

    assert minimax_h3_plan_from_batch(rebuilt) == minimax_h3_plan_from_batch(req)
    if role is RoleType.DENOISER:
        expected = req.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]["positive"]
        actual = rebuilt.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]["positive"]
        assert torch.equal(actual["hidden_states"], expected["hidden_states"])
        assert torch.equal(actual["text_token_tags"], expected["text_token_tags"])
    else:
        assert torch.equal(rebuilt.latents, req.latents)
        assert torch.equal(rebuilt.audio_latents, req.audio_latents)
