# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    prompt_admission as admission,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
    MINIMAX_H3_PROBE_FACTS_EXTRA_KEY,
    MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.text_encoding import (
    MiniMaxH3TextEncodingStage,
)
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api


class _Tokenizer:
    model_max_length = 262144

    def __call__(self, text, *, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False
        assert truncation is False
        if text.startswith("tokens:"):
            count = int(text.split(":", 1)[1])
            return {"input_ids": list(range(count))}
        return {"input_ids": [100 + len(text)]}

    @staticmethod
    def convert_tokens_to_ids(token):
        return {
            "<|vision_start|>": 900,
            "<|vision_end|>": 901,
            "<|image_pad|>": 902,
            "<|video_pad|>": 903,
        }[token]


def _runtime():
    image = admission._ProcessorGridConfig(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=65536,
        max_pixels=16777216,
    )
    video = admission._ProcessorGridConfig(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=4096,
        max_pixels=25165824,
    )
    return admission._PromptRuntime(
        tokenizer=_Tokenizer(),
        tokenizer_path="/model/FL2VA/tokenizer",
        processor_path="/model/FL2VA/processor",
        tokenizer_sha256="sha256:tokenizer",
        processor_config_sha256="sha256:processor",
        image=image,
        video=video,
    )


def _material(chain, index):
    return SimpleNamespace(material_chain=chain, condition_index=index)


def _prepare(monkeypatch, plan, batch, *, tasks="t2va", limit=40960):
    monkeypatch.setenv("SGLANG_H3_PROMPT_ADMISSION_MAX_TOKENS", str(limit))
    monkeypatch.setenv("SGLANG_H3_PROMPT_ADMISSION_TASKS", tasks)
    monkeypatch.setenv("SGLANG_H3_PROMPT_ADMISSION_EVIDENCE", "experiment:sha256")
    with (
        patch.object(admission, "_runtime_for_server", return_value=_runtime()),
        patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=SimpleNamespace(),
        ),
    ):
        return admission.minimax_h3_prepare_prompt_admission(batch, plan)


def test_measured_t2va_limit_accepts_40960_and_rejects_next_token(monkeypatch):
    accepted_plan = SimpleNamespace(
        task="t2va", prompt="tokens:40960", materials=[], shape={}
    )
    accepted_batch = SimpleNamespace(extra={})
    payload = _prepare(monkeypatch, accepted_plan, accepted_batch)

    assert payload["prompt_token_count"] == 40960
    assert payload["presentation_token_count"] == 40960
    assert payload["vision_token_count"] == 0
    assert payload["admission"] == {
        "applied": True,
        "max_presentation_tokens": 40960,
        "evidence": "experiment:sha256",
    }

    rejected_plan = SimpleNamespace(
        task="t2va", prompt="tokens:40961", materials=[], shape={}
    )
    with pytest.raises(ValueError, match=r"40961 > 40960"):
        _prepare(monkeypatch, rejected_plan, SimpleNamespace(extra={}))


def test_video_api_returns_400_before_queue_for_admission_rejection(
    monkeypatch, tmp_path
):
    class Sampling:
        def prepare_video_request_for_queue(self, _batch):
            raise ValueError("40961 > 40960 tokens")

        def cleanup_video_request(self, _batch):
            return None

    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            task_type=SimpleNamespace(requires_image_input=lambda: False)
        ),
        input_save_path=str(tmp_path / "inputs"),
        output_path=str(tmp_path / "outputs"),
        served_model_name="minimax-h3",
    )
    batch = SimpleNamespace(extra={})
    monkeypatch.setattr(video_api, "get_global_server_args", lambda: server_args)
    monkeypatch.setattr(
        video_api, "_build_video_sampling_params", lambda *_: Sampling()
    )
    monkeypatch.setattr(video_api, "prepare_request", lambda **_kwargs: batch)

    app = FastAPI()
    app.include_router(video_api.router)
    response = TestClient(app).post(
        "/v1/videos",
        json={
            "prompt": "tokens:40961",
            "task": "t2va",
            "conditions": [],
            "target": {
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 15,
            },
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "40961 > 40960 tokens"}


def test_fl2va_counts_full_presentation_without_claiming_t2va_evidence(monkeypatch):
    plan = SimpleNamespace(
        task="fl2va",
        prompt="tokens:3",
        materials=[_material("image.target_canvas", 0)],
        shape={"frame_count": 362},
    )
    batch = SimpleNamespace(
        extra={
            MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY: {
                0: {"width": 1344, "height": 768}
            },
            MINIMAX_H3_PROBE_FACTS_EXTRA_KEY: {0: {"has_audio": False}},
        }
    )

    payload = _prepare(monkeypatch, plan, batch)

    assert payload["image_token_counts"] == [1008]
    assert payload["image_grids"] == [
        {
            "width": 1344,
            "height": 768,
            "resized_width": 1344,
            "resized_height": 768,
            "grid_t": 1,
            "grid_h": 48,
            "grid_w": 84,
        }
    ]
    assert payload["prompt_token_count"] == 3
    assert payload["vision_token_count"] == 1010
    assert payload["presentation_token_count"] == 1014
    assert payload["admission"]["applied"] is False

    restored = pickle.loads(pickle.dumps(batch))
    ids, tags = admission.minimax_h3_precomputed_presentation(
        restored,
        plan,
        image_token_counts=[1008],
    )
    assert ids.shape == tags.shape == (1014,)
    with pytest.raises(ValueError, match="worker preprocessing"):
        admission.minimax_h3_precomputed_presentation(
            restored,
            plan,
            image_token_counts=[1007],
        )


def test_worker_reuses_http_t2va_ids_without_tokenizing_again(monkeypatch):
    plan = SimpleNamespace(task="t2va", prompt="tokens:3", materials=[], shape={})
    batch = SimpleNamespace(extra={})
    payload = _prepare(monkeypatch, plan, batch)

    class Encoder:
        hidden_dim = 4

        def __init__(self):
            self.seen_ids = None

        def encode_ids(self, input_ids):
            self.seen_ids = input_ids.clone()
            return torch.zeros((int(input_ids.shape[0]), self.hidden_dim))

    class RejectingTokenizer:
        name_or_path = "/model/FL2VA/tokenizer"

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("worker tokenizer must not run")

    encoder = Encoder()
    stage = MiniMaxH3TextEncodingStage(
        text_encoder=encoder,
        tokenizer=RejectingTokenizer(),
        processor=object(),
    )
    stage._manage_text_encoder_use = lambda _index: None

    stage._encode_from_plan(batch, plan)

    assert encoder.seen_ids.tolist() == payload["input_ids"] == [0, 1, 2]
    encoded = batch.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]["positive"]
    assert encoded["text_len"] == 3
    assert encoded["text_token_tags"].tolist() == [1, 1, 1]


def test_ref2va_video_grid_and_audio_label_accounting_match_worker(monkeypatch):
    materials = [
        _material("image.reference_preserve", 0),
        _material("video.reference_preserve", 1),
        _material("audio", 2),
    ]
    plan = SimpleNamespace(
        task="ref2va",
        prompt="tokens:5",
        materials=materials,
        shape={"frame_count": 362},
    )
    batch = SimpleNamespace(
        extra={
            MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY: {
                0: {"width": 3584, "height": 2048},
                1: {"width": 1344, "height": 768},
            },
            MINIMAX_H3_PROBE_FACTS_EXTRA_KEY: {
                0: {"has_audio": False},
                1: {"has_audio": True},
                2: {"has_audio": True},
            },
        }
    )

    payload = _prepare(monkeypatch, plan, batch)

    assert payload["condition_labels"] == [
        ["image", 1],
        ["audio", 1],
        ["video", 1],
        ["audio", 2],
    ]
    assert payload["image_token_counts"] == [7168]
    assert payload["video_block_token_counts"] == [[777] * 16]
    assert payload["video_block_timestamps"][0][0] == pytest.approx(0.25)
    assert payload["video_block_timestamps"][0][-1] == pytest.approx(15.0)
    assert payload["video_grids"][0] == {
        "width": 1344,
        "height": 768,
        "frame_count": 362,
        "sampled_frames": 31,
        "resized_width": 1184,
        "resized_height": 672,
        "grid_t": 16,
        "grid_h": 42,
        "grid_w": 74,
    }
    assert payload["admission"]["applied"] is False

    ids, tags = admission.minimax_h3_precomputed_presentation(
        batch,
        plan,
        image_token_counts=[7168],
        condition_labels=[
            ("image", 1),
            ("audio", 1),
            ("video", 1),
            ("audio", 2),
        ],
        video_block_token_counts=[[777] * 16],
        video_block_timestamps=payload["video_block_timestamps"],
    )
    assert ids.shape == tags.shape
    assert int(tags.eq(0).sum()) == payload["vision_token_count"]


def test_enabled_admission_configuration_fails_closed_without_scope(monkeypatch):
    monkeypatch.setenv("SGLANG_H3_PROMPT_ADMISSION_MAX_TOKENS", "40960")
    monkeypatch.delenv("SGLANG_H3_PROMPT_ADMISSION_TASKS", raising=False)
    monkeypatch.delenv("SGLANG_H3_PROMPT_ADMISSION_EVIDENCE", raising=False)
    with pytest.raises(ValueError, match="TASKS.*required"):
        admission.minimax_h3_prompt_admission_config()

    monkeypatch.setenv("SGLANG_H3_PROMPT_ADMISSION_TASKS", "t2va")
    with pytest.raises(ValueError, match="EVIDENCE.*required"):
        admission.minimax_h3_prompt_admission_config()


def test_zero_limit_disables_admission_without_stale_scope(monkeypatch):
    monkeypatch.setenv("SGLANG_H3_PROMPT_ADMISSION_MAX_TOKENS", "0")
    monkeypatch.setenv("SGLANG_H3_PROMPT_ADMISSION_TASKS", "ref2va")
    monkeypatch.setenv("SGLANG_H3_PROMPT_ADMISSION_EVIDENCE", "stale")

    config = admission.minimax_h3_prompt_admission_config()

    assert config.enabled is False
    assert config.max_presentation_tokens is None
    assert config.calibrated_tasks == ()
    assert config.evidence is None
