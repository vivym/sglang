# SPDX-License-Identifier: Apache-2.0
"""Public MiniMax H3 model-index admission contract."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.sample.sampling_params import QUALITY_LEVELS
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_plan_from_batch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    canonical_minimax_h3_task,
    partition_for_task,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

_MINIMAX_H3_QUALITY_WORKLOAD = {
    "task": "t2va",
    "width": 1344,
    "height": 768,
    "fps": 24,
    "frame_count": 124,
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
}

_MINIMAX_H3_FAST_WORKLOAD = {
    "task": "t2va",
    "width": 1344,
    "height": 768,
    "fps": 24,
    "frame_count": {124, 243, 362},
    "num_inference_steps": {7, 9},
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
}

_MINIMAX_H3_EXPERIMENTAL_LORA_WORKLOAD = {
    "task": "t2va",
    "width": 1344,
    "height": 768,
    "fps": 24,
    "frame_count": {124, 243, 362},
    # LightX2V publishes 4/8-NFE adapters as 5/9 sigma-point schedules.
    "num_inference_steps": {5, 9},
    "flow_shift": 6.0,
    "audio_flow_shift": 3.0,
}

_MINIMAX_H3_EXPERIMENTAL_FASTH3_WORKLOAD = {
    "task": "t2va",
    "width": 1344,
    "height": 768,
    "fps": 24,
    "frame_count": {124, 243, 362},
    # FastH3 publishes four forwards; Tutu publishes eight forwards. Both use
    # explicit base schedules on the H3 12/3 shifted clocks.
    "num_inference_steps": {5, 9},
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
}

_MINIMAX_H3_RES_MULTISTEP_CANARY_WORKLOAD = {
    "task": "t2va",
    "width": 1344,
    "height": 768,
    "fps": 24,
    "frame_count": {124, 243, 362},
    # 13/15/17/21 sigma points execute 12/14/16/20 DiT calls.
    "num_inference_steps": {13, 15, 17, 21},
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
}

_MINIMAX_H3_BAKED_LORA_CONFIG_KEY = "minimax_h3_baked_lora"
_MINIMAX_H3_BAKED_LORA_MODE = "bf16_lora_merge_then_convrot_int8_requantize"
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def _validate_baked_lora_profile(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"quantization_config.{_MINIMAX_H3_BAKED_LORA_CONFIG_KEY} must be an object"
        )
    if value.get("format_version") != "1":
        raise ValueError("MiniMax-H3 baked LoRA format_version must be '1'")
    if value.get("mode") != _MINIMAX_H3_BAKED_LORA_MODE:
        raise ValueError("MiniMax-H3 baked LoRA mode is unsupported")
    for field in ("adapter_sha256", "source_checkpoint_sha256", "converter_revision"):
        digest = value.get(field)
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"MiniMax-H3 baked LoRA {field} must be a SHA-256 digest")
    config_digest = value.get("adapter_config_sha256")
    if config_digest != "none" and (
        not isinstance(config_digest, str)
        or _SHA256_PATTERN.fullmatch(config_digest) is None
    ):
        raise ValueError(
            "MiniMax-H3 baked LoRA adapter_config_sha256 must be 'none' or a digest"
        )
    scale = value.get("adapter_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0
    ):
        raise ValueError("MiniMax-H3 baked LoRA adapter_scale must be positive")
    alphas = value.get("adapter_alphas")
    if not isinstance(alphas, list) or not alphas:
        raise ValueError("MiniMax-H3 baked LoRA adapter_alphas must be non-empty")
    if any(
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0
        for alpha in alphas
    ):
        raise ValueError("MiniMax-H3 baked LoRA adapter_alphas must be positive")
    if value.get("lora_pairs") != 208:
        raise ValueError("MiniMax-H3 baked LoRA must cover all 208 runtime LoRA layers")
    return dict(value)


def _string_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    values = tuple(value)
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{path} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{path} must not contain duplicates")
    return values


@dataclass(frozen=True)
class MiniMaxH3ReleaseMetadata:
    schema_version: int
    partition: str
    tasks: tuple[str, ...]
    task_aliases: Mapping[str, str]
    video_sigma_shift: float
    audio_sigma_shift: float
    base_schedule: tuple[float, ...] | None
    lora_scale: float | None

    @classmethod
    def from_model_index(
        cls, model_index: Mapping[str, Any]
    ) -> MiniMaxH3ReleaseMetadata:
        raw = model_index.get("_minimax_h3")
        if not isinstance(raw, Mapping):
            raise ValueError("model_index.json._minimax_h3 must be an object")
        if raw.get("schema_version") != 1:
            raise ValueError("model_index.json._minimax_h3.schema_version must be 1")
        partition = raw.get("partition")
        if partition not in {"fl2va", "ref2va"}:
            raise ValueError(
                "model_index.json._minimax_h3.partition must be one of fl2va, ref2va"
            )
        tasks = _string_list(raw.get("tasks"), "model_index.json._minimax_h3.tasks")
        aliases = raw.get("task_aliases", {})
        if not isinstance(aliases, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in aliases.items()
        ):
            raise ValueError(
                "model_index.json._minimax_h3.task_aliases must map strings to strings"
            )
        scales = raw.get("sigma_shift_scales")
        if not isinstance(scales, Mapping):
            raise ValueError(
                "model_index.json._minimax_h3.sigma_shift_scales must be an object"
            )
        try:
            video_sigma = float(scales["video"])
            audio_sigma = float(scales["audio"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "model_index.json._minimax_h3.sigma_shift_scales requires numeric "
                "video and audio values"
            ) from exc
        base_schedule_raw = raw.get("base_schedule")
        base_schedule = None
        if base_schedule_raw is not None:
            if not isinstance(base_schedule_raw, list):
                raise ValueError(
                    "model_index.json._minimax_h3.base_schedule must be a list"
                )
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
                minimax_h3_time_shift_sigmas,
            )

            # The helper owns the exact endpoint, monotonicity, and finite checks.
            minimax_h3_time_shift_sigmas(
                num_steps=len(base_schedule_raw),
                shift_scale=1.0,
                base_schedule=base_schedule_raw,
            )
            base_schedule = tuple(float(value) for value in base_schedule_raw)
        lora_scale_raw = raw.get("lora_scale")
        lora_scale = None
        if lora_scale_raw is not None:
            if isinstance(lora_scale_raw, bool) or not isinstance(
                lora_scale_raw, (int, float)
            ):
                raise ValueError(
                    "model_index.json._minimax_h3.lora_scale must be numeric"
                )
            lora_scale = float(lora_scale_raw)
            if not math.isfinite(lora_scale) or lora_scale <= 0:
                raise ValueError(
                    "model_index.json._minimax_h3.lora_scale must be positive and finite"
                )
        metadata = cls(
            schema_version=1,
            partition=partition,
            tasks=tasks,
            task_aliases=dict(aliases),
            video_sigma_shift=video_sigma,
            audio_sigma_shift=audio_sigma,
            base_schedule=base_schedule,
            lora_scale=lora_scale,
        )
        for task in metadata.tasks:
            if canonical_minimax_h3_task(task) != task:
                raise ValueError(
                    f"tasks must contain canonical task names, got {task!r}"
                )
            if partition_for_task(task) != partition:
                raise ValueError(
                    f"task {task!r} does not belong to partition {partition!r}"
                )
        for alias, target in metadata.task_aliases.items():
            if target not in metadata.tasks:
                raise ValueError(
                    f"task alias {alias!r} targets undeclared task {target!r}"
                )
            if canonical_minimax_h3_task(alias) != target:
                raise ValueError(
                    f"unsupported task alias mapping {alias!r} -> {target!r}"
                )
        return metadata

    @property
    def sigma_shift_scales(self) -> dict[str, float]:
        return {"video": self.video_sigma_shift, "audio": self.audio_sigma_shift}

    def canonical_task(self, task: str) -> str:
        normalized = task.strip().lower()
        canonical = self.task_aliases.get(normalized, normalized)
        if canonical not in self.tasks:
            raise ValueError(
                f"task {task!r} is not served by MiniMax H3 partition {self.partition!r}; "
                f"supported tasks: {list(self.tasks)!r}"
            )
        if partition_for_task(canonical) != self.partition:
            raise ValueError(
                f"task {task!r} resolves outside partition {self.partition!r}"
            )
        return canonical


class MiniMaxH3PartitionAdmissionStage(PipelineStage):
    def __init__(
        self,
        metadata: MiniMaxH3ReleaseMetadata,
        baked_lora_profile: Mapping[str, Any] | None = None,
        lora_identity_provider: Callable[[], Mapping[str, Any] | None] | None = None,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.baked_lora_profile = _validate_baked_lora_profile(baked_lora_profile)
        self.lora_identity_provider = lora_identity_provider

    def _validate_runtime_lora_identity(self, server_args: ServerArgs) -> None:
        expected_digest = getattr(server_args, "lora_expected_sha256", None)
        if (
            not isinstance(expected_digest, str)
            or _SHA256_PATTERN.fullmatch(expected_digest) is None
        ):
            raise ValueError(
                'MiniMax-H3 quality="fast" startup LoRA requires '
                "--lora-expected-sha256=sha256:<digest>"
            )
        if self.lora_identity_provider is None:
            raise ValueError(
                'MiniMax-H3 quality="fast" cannot verify the active LoRA identity'
            )
        identity = self.lora_identity_provider()
        if not isinstance(identity, Mapping):
            raise ValueError(
                'MiniMax-H3 quality="fast" requires an active transformer LoRA'
            )
        nicknames = identity.get("nicknames")
        paths = identity.get("paths")
        digests = identity.get("sha256")
        alphas = identity.get("alphas")
        strengths = identity.get("strengths")
        values = (nicknames, paths, digests, alphas, strengths)
        if any(
            not isinstance(value, (tuple, list)) or len(value) != 1 for value in values
        ):
            raise ValueError(
                'MiniMax-H3 quality="fast" requires exactly one active transformer LoRA'
            )
        expected_path = getattr(server_args, "lora_path", None)
        expected_alpha = getattr(server_args, "lora_alpha", None)
        expected_strength = float(getattr(server_args, "lora_scale", 1.0))
        mismatches = {}
        actual = {
            "path": paths[0],
            "sha256": digests[0],
            "alpha": alphas[0],
            "strength": float(strengths[0]),
            "merged": bool(identity.get("merged", False)),
            "adaln_cache_path": identity.get("adaln_cache_path"),
            "adaln_cache_sha256": identity.get("adaln_cache_sha256"),
        }
        expected_adaln_path = getattr(server_args, "minimax_h3_adaln_cache_path", None)
        expected_adaln_digest = getattr(
            server_args, "minimax_h3_adaln_cache_expected_sha256", None
        )
        if not isinstance(expected_adaln_path, str) or not expected_adaln_path:
            raise ValueError(
                'MiniMax-H3 quality="fast" startup LoRA requires an explicit '
                "--minimax-h3-adaln-cache-path"
            )
        if (
            not isinstance(expected_adaln_digest, str)
            or _SHA256_PATTERN.fullmatch(expected_adaln_digest) is None
        ):
            raise ValueError(
                'MiniMax-H3 quality="fast" startup LoRA requires '
                "--minimax-h3-adaln-cache-expected-sha256=sha256:<digest>"
            )
        expected = {
            "path": expected_path,
            "sha256": expected_digest,
            "alpha": expected_alpha,
            "strength": expected_strength,
            "merged": False,
            "adaln_cache_path": expected_adaln_path,
            "adaln_cache_sha256": expected_adaln_digest,
        }
        if expected_alpha is None:
            mismatches["alpha"] = {"expected": "explicit lora_alpha", "actual": None}
        for name, wanted in expected.items():
            if name == "alpha" and wanted is None:
                continue
            if actual[name] != wanted:
                mismatches[name] = {"expected": wanted, "actual": actual[name]}
        if mismatches:
            raise ValueError(
                f'MiniMax-H3 quality="fast" active LoRA identity mismatch: {mismatches}'
            )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        task = None if batch.sampling_params is None else batch.sampling_params.task
        if not isinstance(task, str) or not task.strip():
            raise ValueError("MiniMax H3 request task must be a non-empty string")
        self.metadata.canonical_task(task)
        if batch.num_inference_steps < 2:
            raise ValueError(
                "MiniMax H3 requires num_inference_steps >= 2 because its "
                "video/audio sigma schedules include both interval endpoints"
            )
        gpu_plans = envs.SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS
        if (
            server_args.minimax_h3_adaln_online
            and batch.num_inference_steps - 1 > gpu_plans
        ):
            # Fail here, before the encode stages spend GPU time on a request
            # whose AdaLN rebuild is guaranteed to overflow the slab.
            raise ValueError(
                f"num_inference_steps={batch.num_inference_steps} needs up to "
                f"{batch.num_inference_steps - 1} AdaLN plans but the online "
                f"slab holds {gpu_plans}; raise "
                "SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS"
            )
        quality = getattr(batch.sampling_params, "quality", "lossless")
        sampler_mode = getattr(batch.sampling_params, "sampler_mode", "euler")
        if quality not in QUALITY_LEVELS:
            raise ValueError(
                f"quality must be one of {list(QUALITY_LEVELS)}, got {quality!r}"
            )
        if sampler_mode not in {"euler", "res_multistep"}:
            raise ValueError(f"unsupported MiniMax-H3 sampler_mode {sampler_mode!r}")
        startup_lora = getattr(server_args, "lora_path", None)
        baked_lora = self.baked_lora_profile
        if startup_lora is not None and baked_lora is not None:
            raise ValueError(
                "MiniMax-H3 baked LoRA checkpoints cannot apply a startup LoRA"
            )
        has_lora_weights = startup_lora is not None or baked_lora is not None
        uses_res_multistep = sampler_mode == "res_multistep"
        if not batch.is_warmup:
            if has_lora_weights and uses_res_multistep:
                raise ValueError(
                    "MiniMax-H3 res_multistep cannot be combined with LoRA weights"
                )
            if has_lora_weights and quality != "fast":
                raise ValueError(
                    'MiniMax-H3 with startup or baked LoRA requires quality="fast"; '
                    f"quality={quality!r} cannot describe the effective model state"
                )
            if uses_res_multistep and quality != "fast":
                raise ValueError(
                    'MiniMax-H3 res_multistep requires quality="fast"; '
                    f"quality={quality!r} cannot describe the solver trajectory"
                )
            if quality == "fast" and not has_lora_weights and not uses_res_multistep:
                raise ValueError(
                    'MiniMax-H3 quality="fast" requires an explicitly configured '
                    "startup/baked LoRA or sampler_mode='res_multistep'"
                )
        high_quality = quality == "high"
        if high_quality and not batch.is_warmup:
            server_args.pipeline_config.validate_quality_deployment(server_args)
            plan = minimax_h3_plan_from_batch(batch)
            if plan is None:
                raise ValueError(
                    'MiniMax-H3 quality="high" requires a resolved request plan'
                )
            shape = plan.shape
            actual = {
                "task": plan.task,
                "width": int(shape["width"]),
                "height": int(shape["height"]),
                "fps": int(shape["fps"]),
                "frame_count": int(shape["frame_count"]),
                "num_inference_steps": int(batch.num_inference_steps),
                "flow_shift": float(
                    plan.flow_shift
                    if plan.flow_shift is not None
                    else plan.default_flow_shift
                ),
                "audio_flow_shift": float(
                    plan.audio_flow_shift
                    if plan.audio_flow_shift is not None
                    else plan.default_audio_flow_shift
                ),
            }
            exact_fields = (
                "task",
                "width",
                "height",
                "fps",
                "frame_count",
                "num_inference_steps",
            )
            exact = all(
                actual[name] == _MINIMAX_H3_QUALITY_WORKLOAD[name]
                for name in exact_fields
            )
            shifts = math.isclose(
                actual["flow_shift"],
                _MINIMAX_H3_QUALITY_WORKLOAD["flow_shift"],
                abs_tol=1e-9,
            ) and math.isclose(
                actual["audio_flow_shift"],
                _MINIMAX_H3_QUALITY_WORKLOAD["audio_flow_shift"],
                abs_tol=1e-9,
            )
            if not exact or not shifts:
                raise ValueError(
                    'MiniMax-H3 quality="high" is validated only for '
                    f"{_MINIMAX_H3_QUALITY_WORKLOAD}; got {actual}"
                )
        if quality == "fast" and not batch.is_warmup:
            if uses_res_multistep:
                enabled = os.environ.get(
                    "SGLANG_H3_EXPERIMENTAL_RES_MULTISTEP", "0"
                ).strip().lower() in {"1", "true", "yes", "on"}
                if not enabled:
                    raise ValueError(
                        "MiniMax-H3 res_multistep canary is disabled; set "
                        "SGLANG_H3_EXPERIMENTAL_RES_MULTISTEP=1"
                    )
                plan = minimax_h3_plan_from_batch(batch)
                if plan is None:
                    raise ValueError(
                        'MiniMax-H3 quality="fast" requires a resolved request plan'
                    )
                shape = plan.shape
                actual = {
                    "task": plan.task,
                    "width": int(shape["width"]),
                    "height": int(shape["height"]),
                    "fps": int(shape["fps"]),
                    "frame_count": int(shape["frame_count"]),
                    "num_inference_steps": int(batch.num_inference_steps),
                    "flow_shift": float(
                        plan.flow_shift
                        if plan.flow_shift is not None
                        else plan.default_flow_shift
                    ),
                    "audio_flow_shift": float(
                        plan.audio_flow_shift
                        if plan.audio_flow_shift is not None
                        else plan.default_audio_flow_shift
                    ),
                }
                exact = all(
                    actual[name] == _MINIMAX_H3_RES_MULTISTEP_CANARY_WORKLOAD[name]
                    for name in ("task", "width", "height", "fps")
                )
                sets_match = all(
                    actual[name] in _MINIMAX_H3_RES_MULTISTEP_CANARY_WORKLOAD[name]
                    for name in ("frame_count", "num_inference_steps")
                )
                shifts = math.isclose(
                    actual["flow_shift"],
                    _MINIMAX_H3_RES_MULTISTEP_CANARY_WORKLOAD["flow_shift"],
                    abs_tol=1e-9,
                ) and math.isclose(
                    actual["audio_flow_shift"],
                    _MINIMAX_H3_RES_MULTISTEP_CANARY_WORKLOAD["audio_flow_shift"],
                    abs_tol=1e-9,
                )
                if not exact or not sets_match or not shifts:
                    raise ValueError(
                        "MiniMax-H3 res_multistep canary is admitted only for "
                        f"{_MINIMAX_H3_RES_MULTISTEP_CANARY_WORKLOAD}; got {actual}"
                    )
                return batch
            expected_lora_scale = (
                self.metadata.lora_scale
                if self.metadata.lora_scale is not None
                else 1.0
            )
            if baked_lora is not None:
                if float(baked_lora["adapter_scale"]) != expected_lora_scale:
                    raise ValueError(
                        'MiniMax-H3 quality="fast" baked checkpoint requires '
                        f"adapter_scale={expected_lora_scale}"
                    )
            else:
                try:
                    lora_scale = float(getattr(server_args, "lora_scale", None))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        'MiniMax-H3 quality="fast" requires lora_scale='
                        f"{expected_lora_scale}"
                    ) from exc
                if (
                    lora_scale != expected_lora_scale
                    or getattr(server_args, "lora_merge_mode", None) != "dynamic"
                ):
                    raise ValueError(
                        'MiniMax-H3 quality="fast" requires lora_scale='
                        f"{expected_lora_scale} and "
                        'lora_merge_mode="dynamic"'
                    )
                self._validate_runtime_lora_identity(server_args)
            plan = minimax_h3_plan_from_batch(batch)
            if plan is None:
                raise ValueError(
                    'MiniMax-H3 quality="fast" requires a resolved request plan'
                )
            shape = plan.shape
            actual = {
                "task": plan.task,
                "width": int(shape["width"]),
                "height": int(shape["height"]),
                "fps": int(shape["fps"]),
                "frame_count": int(shape["frame_count"]),
                "num_inference_steps": int(batch.num_inference_steps),
                "flow_shift": float(
                    plan.flow_shift
                    if plan.flow_shift is not None
                    else plan.default_flow_shift
                ),
                "audio_flow_shift": float(
                    plan.audio_flow_shift
                    if plan.audio_flow_shift is not None
                    else plan.default_audio_flow_shift
                ),
            }
            exact = all(
                actual[name] == _MINIMAX_H3_FAST_WORKLOAD[name]
                for name in ("task", "width", "height", "fps")
            )
            sets_match = all(
                actual[name] in _MINIMAX_H3_FAST_WORKLOAD[name]
                for name in ("frame_count", "num_inference_steps")
            )
            shifts = math.isclose(
                actual["flow_shift"],
                _MINIMAX_H3_FAST_WORKLOAD["flow_shift"],
                abs_tol=1e-9,
            ) and math.isclose(
                actual["audio_flow_shift"],
                _MINIMAX_H3_FAST_WORKLOAD["audio_flow_shift"],
                abs_tol=1e-9,
            )
            validated_fast = exact and sets_match and shifts
            experimental_workloads = (
                _MINIMAX_H3_EXPERIMENTAL_LORA_WORKLOAD,
                _MINIMAX_H3_EXPERIMENTAL_FASTH3_WORKLOAD,
            )
            experimental_fast = any(
                all(
                    actual[name] == workload[name]
                    for name in ("task", "width", "height", "fps")
                )
                and all(
                    actual[name] in workload[name]
                    for name in ("frame_count", "num_inference_steps")
                )
                and math.isclose(
                    actual["flow_shift"], workload["flow_shift"], abs_tol=1e-9
                )
                and math.isclose(
                    actual["audio_flow_shift"],
                    workload["audio_flow_shift"],
                    abs_tol=1e-9,
                )
                for workload in experimental_workloads
            )
            experimental_enabled = os.environ.get(
                "SGLANG_H3_EXPERIMENTAL_LORA_WORKLOAD", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            if experimental_fast and not experimental_enabled:
                raise ValueError(
                    "MiniMax-H3 experimental LoRA workload is disabled; set "
                    "SGLANG_H3_EXPERIMENTAL_LORA_WORKLOAD=1"
                )
            if not validated_fast and not experimental_fast:
                raise ValueError(
                    'MiniMax-H3 quality="fast" canary is validated only for '
                    f"{_MINIMAX_H3_FAST_WORKLOAD}; experimental opt-in accepts "
                    f"{experimental_workloads}; got {actual}"
                )
        return batch


__all__ = ["MiniMaxH3PartitionAdmissionStage", "MiniMaxH3ReleaseMetadata"]
