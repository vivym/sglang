import asyncio
import inspect
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.configs.sample.ltx_2 import LTX23SamplingParams
from sglang.multimodal_gen.configs.sample.ltx_2_5 import LTX25SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
    VideoGenerationsRequest,
    VideoResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    RealtimeChunkInputs,
    build_realtime_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _build_video_sampling_params,
    _dispatch_job_async,
    _video_request_model_kwargs,
    create_video,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    add_common_data_to_response,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.media_encoder.protocol import MediaEncodeManifest
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics


def test_multipart_video_declares_perf_dump_path_form_field():
    assert "perf_dump_path" in inspect.signature(create_video).parameters


def test_video_api_forwards_profiling_options():
    request = VideoGenerationsRequest(
        prompt="profile this request",
        task="t2va",
        conditions=[],
        target={
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        },
        profile=True,
        num_profiled_timesteps=3,
        profile_all_stages=False,
        quality="high",
    )
    server_args = SimpleNamespace(
        backend="auto",
        model_id=None,
        model_path="MiniMaxAI/MiniMax-H3",
        pipeline_class_name="MiniMaxH3Pipeline",
        pipeline_config=object(),
    )

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api."
            "get_global_server_args",
            return_value=server_args,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api."
            "build_sampling_params",
            side_effect=lambda request_id, **kwargs: kwargs,
        ),
    ):
        kwargs = _build_video_sampling_params("profile-request", request)

    assert kwargs["profile"] is True
    assert kwargs["num_profiled_timesteps"] == 3
    assert kwargs["profile_all_stages"] is False
    assert kwargs["quality"] == "high"


def test_video_api_forwards_request_scoped_teacache_params():
    teacache_params = {
        "teacache_thresh": 0.08,
        "start_skipping": 15,
        "end_skipping": 16,
        "coefficients": [1.0, 0.0],
    }
    request = VideoGenerationsRequest(
        prompt="preview this request",
        task="t2va",
        conditions=[],
        target={
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 15.0,
        },
        enable_teacache=True,
        teacache_params=teacache_params,
    )
    server_args = SimpleNamespace(
        backend="auto",
        model_id=None,
        model_path="MiniMaxAI/MiniMax-H3",
        pipeline_class_name="MiniMaxH3Pipeline",
        pipeline_config=object(),
    )

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api."
            "get_global_server_args",
            return_value=server_args,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api."
            "build_sampling_params",
            side_effect=lambda request_id, **kwargs: kwargs,
        ),
    ):
        kwargs = _build_video_sampling_params("preview-request", request)

    assert kwargs["enable_teacache"] is True
    assert kwargs["teacache_params"] == teacache_params


def test_video_response_exposes_request_metrics_metadata():
    metrics = RequestMetrics("preview-request")
    metrics.record_metadata(
        "minimax_h3_teacache",
        {"computed_steps": [0, 1], "cached_steps": [15]},
    )
    result = SimpleNamespace(
        peak_memory_mb=0.0,
        metrics=metrics,
        usage=None,
        action_pred=None,
    )

    payload = add_common_data_to_response(
        {"status": "completed"},
        request_id="preview-request",
        result=result,
    )
    response = VideoResponse(**payload)

    assert response.metrics_metadata == {
        "minimax_h3_teacache": {
            "computed_steps": [0, 1],
            "cached_steps": [15],
        }
    }


def test_openai_video_job_completes_from_media_manifest():
    job_id = "media-manifest-job"
    manifest = MediaEncodeManifest(
        request_id=job_id,
        attempt_id="decoder-attempt",
        uri="https://objects.example/media-manifest-job.mp4",
        object_key="media-manifest-job.mp4",
        storage="s3",
        byte_size=1234,
        sha256="1" * 64,
        container="mp4",
        video_codec="h264",
        pixel_format="yuv420p",
        audio_codec="aac",
        width=1344,
        height=768,
        frame_count=362,
        fps=24,
        duration_seconds=362 / 24,
        audio_sample_rate=32_000,
        audio_channels=2,
        source_payload_sha256="2" * 64,
        model_identity={"partition": "fl2va"},
    )
    result = OutputBatch(media_manifest=manifest.to_dict())

    async def scenario():
        await VIDEO_STORE.upsert(job_id, {"id": job_id, "status": "in_progress"})
        try:
            with patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.video_api."
                "process_generation_batch",
                new=AsyncMock(return_value=([], result)),
            ):
                await _dispatch_job_async(job_id, SimpleNamespace())
            stored = await VIDEO_STORE.get(job_id)
            assert stored["status"] == "completed"
            assert stored["url"] == manifest.uri
            assert stored["file_path"] is None
            assert stored["media_manifest"] == manifest.to_dict()
        finally:
            await VIDEO_STORE.pop(job_id)

    asyncio.run(scenario())


def test_legacy_http_returns_media_manifest_without_local_encoding():
    manifest = {
        "schema": "sglang.minimax-h3.media/v1",
        "request_id": "legacy-media-job",
        "attempt_id": "decoder-attempt",
        "uri": "https://objects.example/legacy-media-job.mp4",
    }
    response = OutputBatch(media_manifest=manifest)
    scheduler_client = SimpleNamespace(forward=AsyncMock(return_value=response))

    with patch.object(http_server, "async_scheduler_client", scheduler_client):
        returned = asyncio.run(
            http_server.forward_to_scheduler(SimpleNamespace(), SimpleNamespace())
        )

    assert returned["media_manifest"] == manifest
    assert returned["output"] is None
    assert response.media_manifest == manifest


def test_ltx25_video_extensions_remain_model_specific():
    field_values = {
        "use_diffusion_decoder": True,
        "auto_duration": True,
        "auto_duration_min_seconds": 2.0,
        "auto_duration_max_seconds": 8.0,
    }
    request = VideoGenerationsRequest(
        prompt="a fox in snow",
        extra_body=field_values,
    )
    base_fields = {field.name for field in fields(SamplingParams)}
    ltx25_fields = {field.name for field in fields(LTX25SamplingParams)}

    for field_name in field_values:
        assert field_name not in VideoGenerationsRequest.model_fields
        assert field_name not in base_fields
        assert field_name in ltx25_fields

    assert _video_request_model_kwargs(request, LTX25SamplingParams) == field_values
    assert _video_request_model_kwargs(request, SamplingParams) == {}


def test_ltx23_request_defaults_to_vae_decoder():
    request = Req(sampling_params=LTX23SamplingParams())

    assert request.use_diffusion_decoder is False


def test_realtime_video_api_forwards_sampling_quality():
    request = RealtimeVideoGenerationsRequest(
        type="init",
        prompt="profile this realtime request",
        first_frame="cat.png",
        quality="high",
    )
    chunk_inputs = RealtimeChunkInputs(prompt=request.prompt)

    with patch(
        "sglang.multimodal_gen.runtime.entrypoints.openai.realtime."
        "realtime_adapter.build_sampling_params",
        side_effect=lambda request_id, **kwargs: kwargs,
    ):
        kwargs = build_realtime_sampling_params(
            "realtime-profile-request",
            request=request,
            chunk_inputs=chunk_inputs,
            num_frames=9,
            num_inference_steps=4,
            chunk_size=9,
        )

    assert kwargs["quality"] == "high"
