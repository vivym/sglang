# SPDX-License-Identifier: Apache-2.0
"""HTTP-side MiniMax H3 presentation preparation and measured admission."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
    minimax_h3_multi_image_presentation,
    minimax_h3_ref2va_condition_labels,
    minimax_h3_ref2va_presentation,
    minimax_h3_ref2va_video_presentation,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
    MINIMAX_H3_QWEN_TEMPORAL_PATCH,
    MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS,
    minimax_h3_reference_video_block_timestamps,
)

MINIMAX_H3_PRESENTATION_EXTRA_KEY = "minimax_h3_precomputed_presentation"

_SCHEMA = "minimax-h3-precomputed-presentation-v1"
_ADMISSION_ENV = "SGLANG_H3_PROMPT_ADMISSION_MAX_TOKENS"
_TASKS_ENV = "SGLANG_H3_PROMPT_ADMISSION_TASKS"
_EVIDENCE_ENV = "SGLANG_H3_PROMPT_ADMISSION_EVIDENCE"
_SUPPORTED_TASKS = frozenset({"t2va", "fl2va", "ref2va"})


@dataclass(frozen=True)
class MiniMaxH3PromptAdmissionConfig:
    max_presentation_tokens: int | None
    calibrated_tasks: tuple[str, ...]
    evidence: str | None

    @property
    def enabled(self) -> bool:
        return self.max_presentation_tokens is not None


@dataclass(frozen=True)
class _ProcessorGridConfig:
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int


@dataclass
class _PromptRuntime:
    tokenizer: Any
    tokenizer_path: str
    processor_path: str
    tokenizer_sha256: str
    processor_config_sha256: str
    image: _ProcessorGridConfig
    video: _ProcessorGridConfig


_RUNTIME_LOCK = threading.Lock()
_RUNTIMES: dict[tuple[str, str], _PromptRuntime] = {}


def minimax_h3_prompt_admission_config() -> MiniMaxH3PromptAdmissionConfig:
    rendered_limit = os.environ.get(_ADMISSION_ENV, "0").strip()
    try:
        limit = int(rendered_limit)
    except ValueError as exc:
        raise ValueError(f"{_ADMISSION_ENV} must be a nonnegative integer") from exc
    if limit < 0:
        raise ValueError(f"{_ADMISSION_ENV} must be a nonnegative integer")
    if limit == 0:
        return MiniMaxH3PromptAdmissionConfig(None, (), None)

    rendered_tasks = os.environ.get(_TASKS_ENV, "").strip()
    tasks = tuple(
        dict.fromkeys(
            value.strip().lower()
            for value in rendered_tasks.split(",")
            if value.strip()
        )
    )
    if not tasks:
        raise ValueError(f"{_TASKS_ENV} is required when {_ADMISSION_ENV} is enabled")
    unsupported = sorted(set(tasks) - _SUPPORTED_TASKS)
    if unsupported:
        raise ValueError(f"{_TASKS_ENV} contains unsupported tasks: {unsupported}")

    evidence = os.environ.get(_EVIDENCE_ENV, "").strip()
    if not evidence:
        raise ValueError(
            f"{_EVIDENCE_ENV} is required when {_ADMISSION_ENV} is enabled"
        )
    return MiniMaxH3PromptAdmissionConfig(limit, tasks, evidence)


def _variant_subfolder(server_args: Any) -> str:
    explicit = getattr(server_args, "model_subfolder", None)
    if explicit:
        return str(explicit)
    variant = str(getattr(server_args, "model_variant", None) or "fl2va").lower()
    try:
        return {"fl2va": "FL2VA", "ref2va": "Ref2VA"}[variant]
    except KeyError as exc:
        raise ValueError(f"unsupported MiniMax H3 model variant {variant!r}") from exc


def _component_path(server_args: Any, name: str) -> str:
    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        maybe_download_model,
    )

    overrides = getattr(server_args, "component_paths", None) or {}
    override = overrides.get(name)
    if override:
        resolved = maybe_download_model(str(override))
        return str(Path(resolved).expanduser().resolve())
    subfolder = _variant_subfolder(server_args)
    model_root = maybe_download_model(
        str(server_args.model_path),
        allow_patterns=[f"{subfolder}/{name}/**"],
    )
    return str((Path(model_root) / subfolder / name).expanduser().resolve())


def _file_set_sha256(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    found = False
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    if not found:
        raise FileNotFoundError(f"no identity files found below {root}")
    return f"sha256:{digest.hexdigest()}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _grid_config(raw: dict[str, Any], *, video: bool) -> _ProcessorGridConfig:
    size = raw.get("size")
    if not isinstance(size, dict):
        raise ValueError("MiniMax H3 processor config requires a size object")
    patch = _positive_int(raw.get("patch_size"), name="processor.patch_size")
    temporal = _positive_int(
        raw.get("temporal_patch_size"),
        name="processor.temporal_patch_size",
    )
    merge = _positive_int(raw.get("merge_size"), name="processor.merge_size")
    expected_temporal = MINIMAX_H3_QWEN_TEMPORAL_PATCH if video else 2
    if temporal != expected_temporal:
        raise ValueError(
            "MiniMax H3 processor temporal patch changed: "
            f"expected {expected_temporal}, got {temporal}"
        )
    return _ProcessorGridConfig(
        patch_size=patch,
        temporal_patch_size=temporal,
        merge_size=merge,
        min_pixels=_positive_int(size.get("shortest_edge"), name="size.shortest_edge"),
        max_pixels=_positive_int(size.get("longest_edge"), name="size.longest_edge"),
    )


def _load_runtime(tokenizer_path: str, processor_path: str) -> _PromptRuntime:
    from transformers import AutoTokenizer

    tokenizer_root = Path(tokenizer_path)
    processor_root = Path(processor_path)
    image_config_path = processor_root / "preprocessor_config.json"
    video_config_path = processor_root / "video_preprocessor_config.json"
    image_raw = _load_json(image_config_path)
    video_raw = _load_json(video_config_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="right",
        use_fast=True,
        local_files_only=True,
    )
    return _PromptRuntime(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        processor_path=processor_path,
        tokenizer_sha256=_file_set_sha256(
            tokenizer_root,
            ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"),
        ),
        processor_config_sha256=_file_set_sha256(
            processor_root,
            ("preprocessor_config.json", "video_preprocessor_config.json"),
        ),
        image=_grid_config(image_raw, video=False),
        video=_grid_config(video_raw, video=True),
    )


def _runtime_for_server(server_args: Any) -> _PromptRuntime:
    tokenizer_path = _component_path(server_args, "tokenizer")
    processor_path = _component_path(server_args, "processor")
    key = (tokenizer_path, processor_path)
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(key)
        if runtime is None:
            runtime = _load_runtime(tokenizer_path, processor_path)
            _RUNTIMES[key] = runtime
    return runtime


def minimax_h3_initialize_prompt_admission(server_args: Any) -> None:
    """Load and validate HTTP-side tokenizer state before readiness."""

    config = minimax_h3_prompt_admission_config()
    runtime = _runtime_for_server(server_args)
    if config.enabled:
        model_limit = getattr(runtime.tokenizer, "model_max_length", None)
        if (
            isinstance(model_limit, int)
            and config.max_presentation_tokens > model_limit
        ):
            raise ValueError(
                f"{_ADMISSION_ENV}={config.max_presentation_tokens} exceeds "
                f"tokenizer model_max_length={model_limit}"
            )


def minimax_h3_prompt_admission_runtime_info(server_args: Any) -> dict[str, Any]:
    config = minimax_h3_prompt_admission_config()
    tokenizer_path = _component_path(server_args, "tokenizer")
    processor_path = _component_path(server_args, "processor")
    runtime = _RUNTIMES.get((tokenizer_path, processor_path))
    return {
        "schema": "minimax-h3-prompt-admission-v1",
        "presentation_precompute": True,
        "enabled": config.enabled,
        "max_presentation_tokens": config.max_presentation_tokens,
        "calibrated_tasks": list(config.calibrated_tasks),
        "evidence": config.evidence,
        "evidence_scope": (
            "per-task measured presentation-token ceiling; unlisted tasks are "
            "counted and precomputed but are not memory-admitted by this ceiling"
        ),
        "tokenizer_path": tokenizer_path,
        "processor_path": processor_path,
        "initialized": runtime is not None,
        "tokenizer_sha256": runtime.tokenizer_sha256 if runtime else None,
        "processor_config_sha256": (
            runtime.processor_config_sha256 if runtime else None
        ),
    }


def _image_token_count(
    runtime: _PromptRuntime,
    *,
    width: int,
    height: int,
) -> tuple[int, dict[str, int]]:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    config = runtime.image
    resized_h, resized_w = smart_resize(
        int(height),
        int(width),
        factor=config.patch_size * config.merge_size,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    grid_h = resized_h // config.patch_size
    grid_w = resized_w // config.patch_size
    merged = grid_h * grid_w
    divisor = config.merge_size**2
    if merged % divisor:
        raise ValueError("MiniMax H3 image grid is not divisible by merge_size**2")
    return merged // divisor, {
        "width": int(width),
        "height": int(height),
        "resized_width": resized_w,
        "resized_height": resized_h,
        "grid_t": 1,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }


def _video_token_geometry(
    runtime: _PromptRuntime,
    *,
    width: int,
    height: int,
    frame_count: int,
) -> tuple[list[int], list[float], dict[str, int]]:
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize

    sample_stride = int(24 / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS)
    sampled_frames = (int(frame_count) + sample_stride - 1) // sample_stride
    config = runtime.video
    resized_h, resized_w = smart_resize(
        num_frames=sampled_frames,
        height=int(height),
        width=int(width),
        temporal_factor=config.temporal_patch_size,
        factor=config.patch_size * config.merge_size,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    grid_t = math.ceil(sampled_frames / config.temporal_patch_size)
    grid_h = resized_h // config.patch_size
    grid_w = resized_w // config.patch_size
    merged = grid_h * grid_w
    divisor = config.merge_size**2
    if merged % divisor:
        raise ValueError("MiniMax H3 video grid is not divisible by merge_size**2")
    per_block = merged // divisor
    timestamps = minimax_h3_reference_video_block_timestamps(int(frame_count))
    if len(timestamps) != grid_t:
        raise ValueError(
            "MiniMax H3 reference-video timestamp and processor grids disagree"
        )
    return (
        [per_block] * grid_t,
        timestamps,
        {
            "width": int(width),
            "height": int(height),
            "frame_count": int(frame_count),
            "sampled_frames": sampled_frames,
            "resized_width": resized_w,
            "resized_height": resized_h,
            "grid_t": grid_t,
            "grid_h": grid_h,
            "grid_w": grid_w,
        },
    )


def _require_material_shape(shapes: Any, condition_index: int) -> dict[str, Any]:
    shape = shapes.get(condition_index) if isinstance(shapes, dict) else None
    if not isinstance(shape, dict):
        raise ValueError(
            "prompt presentation requires a resolved material shape for "
            f"conditions[{condition_index}]"
        )
    return shape


def _build_presentation(
    batch: Any, plan: Any, runtime: _PromptRuntime
) -> dict[str, Any]:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
        MINIMAX_H3_PROBE_FACTS_EXTRA_KEY,
        MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY,
    )

    prompt_ids = list(
        runtime.tokenizer(
            plan.prompt,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )
    if not prompt_ids:
        raise ValueError("MiniMax H3 prompt tokenization produced no tokens")
    material_shapes = batch.extra.get(MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY, {})
    probe_facts = batch.extra.get(MINIMAX_H3_PROBE_FACTS_EXTRA_KEY, {})
    image_counts: list[int] = []
    image_grids: list[dict[str, int]] = []
    video_counts: list[list[int]] = []
    video_timestamps: list[list[float]] = []
    video_grids: list[dict[str, int]] = []
    condition_labels: list[tuple[str, int]] = []

    if plan.task == "t2va":
        ids = torch.tensor(prompt_ids, dtype=torch.long)
        tags = torch.ones(int(ids.shape[0]), dtype=torch.long)
    elif plan.task == "fl2va":
        for material in plan.materials:
            if material.material_chain != "image.target_canvas":
                continue
            shape = _require_material_shape(
                material_shapes, int(material.condition_index)
            )
            count, grid = _image_token_count(
                runtime,
                width=int(shape["width"]),
                height=int(shape["height"]),
            )
            image_counts.append(count)
            image_grids.append(grid)
        ids, tags = minimax_h3_multi_image_presentation(
            runtime.tokenizer,
            prompt=plan.prompt,
            prompt_ids=prompt_ids,
            image_token_counts=image_counts,
        )
    elif plan.task == "ref2va":
        video_has_audio: dict[int, bool] = {}
        for material in plan.materials:
            condition_index = int(material.condition_index)
            if material.material_chain == "image.reference_preserve":
                shape = _require_material_shape(material_shapes, condition_index)
                count, grid = _image_token_count(
                    runtime,
                    width=int(shape["width"]),
                    height=int(shape["height"]),
                )
                image_counts.append(count)
                image_grids.append(grid)
            elif material.material_chain in (
                "video.reference_preserve",
                "video_audio.reference_preserve",
            ):
                shape = _require_material_shape(material_shapes, condition_index)
                counts, timestamps, grid = _video_token_geometry(
                    runtime,
                    width=int(shape["width"]),
                    height=int(shape["height"]),
                    frame_count=int(plan.shape["frame_count"]),
                )
                video_counts.append(counts)
                video_timestamps.append(timestamps)
                video_grids.append(grid)
                facts = probe_facts.get(condition_index)
                if not isinstance(facts, dict) or "has_audio" not in facts:
                    raise ValueError(
                        "prompt presentation requires the audio probe for "
                        f"conditions[{condition_index}]"
                    )
                video_has_audio[condition_index] = bool(facts["has_audio"])
        condition_labels = minimax_h3_ref2va_condition_labels(
            plan,
            video_has_audio=video_has_audio,
        )
        image_arg: int | list[int] | None
        image_arg = image_counts[0] if len(image_counts) == 1 else image_counts or None
        if video_counts:
            ids, tags = minimax_h3_ref2va_video_presentation(
                runtime.tokenizer,
                prompt=plan.prompt,
                prompt_ids=prompt_ids,
                condition_labels=condition_labels,
                image_token_count=image_arg,
                video_block_token_counts=video_counts,
                video_block_timestamps=video_timestamps,
            )
        else:
            ids, tags = minimax_h3_ref2va_presentation(
                runtime.tokenizer,
                prompt=plan.prompt,
                prompt_ids=prompt_ids,
                condition_labels=condition_labels,
                image_token_count=image_arg,
            )
    else:
        raise ValueError(f"unsupported MiniMax H3 task {plan.task!r}")

    input_ids = [int(value) for value in ids.tolist()]
    token_tags = [int(value) for value in tags.tolist()]
    if len(input_ids) != len(token_tags):
        raise ValueError("MiniMax H3 presentation ids and tags do not align")
    return {
        "schema": _SCHEMA,
        "task": plan.task,
        "prompt_sha256": f"sha256:{hashlib.sha256(plan.prompt.encode('utf-8')).hexdigest()}",
        "prompt_token_count": len(prompt_ids),
        "presentation_token_count": len(input_ids),
        "text_token_count": sum(tag == 1 for tag in token_tags),
        "vision_token_count": sum(tag == 0 for tag in token_tags),
        "input_ids": input_ids,
        "token_tags": token_tags,
        "condition_labels": [list(value) for value in condition_labels],
        "image_token_counts": image_counts,
        "image_grids": image_grids,
        "video_block_token_counts": video_counts,
        "video_block_timestamps": video_timestamps,
        "video_grids": video_grids,
        "tokenizer_path": runtime.tokenizer_path,
        "tokenizer_sha256": runtime.tokenizer_sha256,
        "processor_path": runtime.processor_path,
        "processor_config_sha256": runtime.processor_config_sha256,
    }


def minimax_h3_prepare_prompt_admission(batch: Any, plan: Any) -> dict[str, Any]:
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    config = minimax_h3_prompt_admission_config()
    runtime = _runtime_for_server(get_global_server_args())
    payload = _build_presentation(batch, plan, runtime)
    applied = plan.task in config.calibrated_tasks
    payload["admission"] = {
        "applied": applied,
        "max_presentation_tokens": (
            config.max_presentation_tokens if applied else None
        ),
        "evidence": config.evidence if applied else None,
    }
    batch.extra[MINIMAX_H3_PRESENTATION_EXTRA_KEY] = payload
    if applied and payload["presentation_token_count"] > config.max_presentation_tokens:
        raise ValueError(
            "MiniMax H3 prompt presentation exceeds the measured serving "
            f"limit for task {plan.task!r}: "
            f"{payload['presentation_token_count']} > "
            f"{config.max_presentation_tokens} tokens"
        )
    return payload


def minimax_h3_precomputed_presentation(
    batch: Any,
    plan: Any,
    *,
    tokenizer: Any | None = None,
    image_token_counts: list[int] | None = None,
    condition_labels: list[tuple[str, int]] | None = None,
    video_block_token_counts: list[list[int]] | None = None,
    video_block_timestamps: list[list[float]] | None = None,
    return_video_mask: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | None
):
    payload = batch.extra.get(MINIMAX_H3_PRESENTATION_EXTRA_KEY)
    if payload is None:
        return None
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("invalid MiniMax H3 precomputed presentation schema")
    prompt_sha256 = f"sha256:{hashlib.sha256(plan.prompt.encode('utf-8')).hexdigest()}"
    if (
        payload.get("task") != plan.task
        or payload.get("prompt_sha256") != prompt_sha256
    ):
        raise ValueError("MiniMax H3 precomputed presentation does not match the plan")
    if tokenizer is not None:
        worker_path = getattr(tokenizer, "name_or_path", None)
        if not isinstance(worker_path, str) or not worker_path:
            raise ValueError(
                "MiniMax H3 worker tokenizer does not expose its source path"
            )
        if (
            Path(worker_path).expanduser().resolve()
            != Path(payload.get("tokenizer_path", "")).expanduser().resolve()
        ):
            raise ValueError(
                "MiniMax H3 HTTP and worker tokenizer paths do not match: "
                f"http={payload.get('tokenizer_path')!r}, worker={worker_path!r}"
            )

    expected = {
        "image_token_counts": list(image_token_counts or []),
        "condition_labels": [list(value) for value in (condition_labels or [])],
        "video_block_token_counts": list(video_block_token_counts or []),
        "video_block_timestamps": list(video_block_timestamps or []),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"MiniMax H3 precomputed {name} disagrees with worker preprocessing"
            )

    input_ids = payload.get("input_ids")
    token_tags = payload.get("token_tags")
    if not isinstance(input_ids, list) or not isinstance(token_tags, list):
        raise ValueError("MiniMax H3 precomputed presentation requires list payloads")
    if len(input_ids) != len(token_tags) or len(input_ids) != payload.get(
        "presentation_token_count"
    ):
        raise ValueError("MiniMax H3 precomputed presentation length mismatch")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in input_ids
    ):
        raise ValueError("MiniMax H3 precomputed input_ids are invalid")
    if any(value not in (0, 1) for value in token_tags):
        raise ValueError("MiniMax H3 precomputed token_tags are invalid")
    if sum(value == 0 for value in token_tags) != payload.get("vision_token_count"):
        raise ValueError("MiniMax H3 precomputed vision token count mismatch")
    if sum(value == 1 for value in token_tags) != payload.get("text_token_count"):
        raise ValueError("MiniMax H3 precomputed text token count mismatch")
    ids = torch.tensor(input_ids, dtype=torch.long)
    tags = torch.tensor(token_tags, dtype=torch.long)
    if not return_video_mask:
        return ids, tags
    if tokenizer is None:
        raise ValueError(
            "MiniMax H3 precomputed video mask requires the worker tokenizer"
        )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
        VIDEO_PAD,
    )

    video_token_id = tokenizer.convert_tokens_to_ids(VIDEO_PAD)
    return ids, tags, ids.eq(video_token_id)


__all__ = [
    "MINIMAX_H3_PRESENTATION_EXTRA_KEY",
    "MiniMaxH3PromptAdmissionConfig",
    "minimax_h3_initialize_prompt_admission",
    "minimax_h3_precomputed_presentation",
    "minimax_h3_prepare_prompt_admission",
    "minimax_h3_prompt_admission_config",
    "minimax_h3_prompt_admission_runtime_info",
]
