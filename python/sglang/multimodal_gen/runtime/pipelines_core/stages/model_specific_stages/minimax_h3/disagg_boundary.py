# SPDX-License-Identifier: Apache-2.0
"""Versioned MiniMax H3 state contracts for disaggregated serving."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import msgspec
import torch

from sglang.multimodal_gen.runtime.disaggregation.boundary import (
    DISAGG_BOUNDARY_FIELD_PREFIX,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

from .constants import MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY
from .resolved_plan import (
    MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
    MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
    MiniMaxH3ResolvedPlan,
    minimax_h3_plan_from_batch,
    minimax_h3_resolve_plan,
)


MINIMAX_H3_DISAGG_SCHEMA = "minimax-h3.disagg-boundary/v1"


def _field(name: str) -> str:
    return f"{DISAGG_BOUNDARY_FIELD_PREFIX}minimax_h3_{name}"


MINIMAX_H3_DISAGG_SCHEMA_FIELD = _field("schema")
MINIMAX_H3_DISAGG_EDGE_FIELD = _field("edge")
MINIMAX_H3_DISAGG_CANONICAL_FIELD = _field("canonical_request")
MINIMAX_H3_DISAGG_PLAN_FIELD = _field("resolved_plan")
MINIMAX_H3_DISAGG_EXPLICIT_FIELDS_FIELD = _field("explicit_sampling_fields")
MINIMAX_H3_DISAGG_RELEASE_IDENTITY_FIELD = _field("release_identity")
MINIMAX_H3_DISAGG_TEXT_LEN_FIELD = _field("text_len")
MINIMAX_H3_DISAGG_TEXT_HIDDEN_FIELD = _field("text_hidden_states")
MINIMAX_H3_DISAGG_TEXT_TAGS_FIELD = _field("text_token_tags")
MINIMAX_H3_DISAGG_TEXT_VIDEO_MASK_FIELD = _field("text_video_token_mask")


_ENCODER_TO_DENOISER = "encoder_to_denoiser"
_DENOISER_TO_DECODER = "denoiser_to_decoder"

_COMMON_SCALAR_FIELDS = {
    MINIMAX_H3_DISAGG_SCHEMA_FIELD,
    MINIMAX_H3_DISAGG_EDGE_FIELD,
    MINIMAX_H3_DISAGG_CANONICAL_FIELD,
    MINIMAX_H3_DISAGG_PLAN_FIELD,
    MINIMAX_H3_DISAGG_EXPLICIT_FIELDS_FIELD,
    MINIMAX_H3_DISAGG_RELEASE_IDENTITY_FIELD,
}


def _edge(source_role: RoleType, destination_role: RoleType) -> str:
    roles = (source_role, destination_role)
    if roles == (RoleType.ENCODER, RoleType.DENOISER):
        return _ENCODER_TO_DENOISER
    if roles == (RoleType.DENOISER, RoleType.DECODER):
        return _DENOISER_TO_DECODER
    raise ValueError(
        "unsupported MiniMax H3 disaggregation role edge: "
        f"{source_role.value!r} -> {destination_role.value!r}"
    )


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, separators=(",", ":")))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"MiniMax H3 disaggregation {name} must be JSON-safe") from exc


def _require_t2va_plan(req: Req) -> tuple[dict[str, Any], MiniMaxH3ResolvedPlan]:
    canonical = req.extra.get(MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY)
    if not isinstance(canonical, Mapping):
        raise ValueError(
            "MiniMax H3 disaggregation requires a canonical request mapping"
        )
    plan = minimax_h3_plan_from_batch(req)
    if not isinstance(plan, MiniMaxH3ResolvedPlan):
        raise ValueError("MiniMax H3 disaggregation requires a resolved plan")
    if plan.task != "t2va" or plan.materials or canonical.get("conditions") != []:
        raise ValueError(
            "MiniMax H3 disaggregation boundary v1 supports text-only t2va; "
            "reference media and keyframes require a later boundary schema"
        )
    return dict(canonical), plan


def _explicit_sampling_fields(req: Req) -> list[str]:
    explicit = getattr(req.sampling_params, "_explicit_fields", ())
    if explicit is None:
        return []
    if not isinstance(explicit, (set, frozenset, list, tuple)) or any(
        not isinstance(name, str) or not name for name in explicit
    ):
        raise ValueError(
            "MiniMax H3 explicit sampling fields must be a collection of strings"
        )
    return sorted(set(explicit))


def _common_scalars(
    req: Req, edge: str, release_identity: Mapping[str, str]
) -> dict[str, Any]:
    if not isinstance(release_identity, Mapping) or not release_identity:
        raise ValueError("MiniMax H3 disaggregation release identity is unavailable")
    canonical, plan = _require_t2va_plan(req)
    return {
        MINIMAX_H3_DISAGG_SCHEMA_FIELD: MINIMAX_H3_DISAGG_SCHEMA,
        MINIMAX_H3_DISAGG_EDGE_FIELD: edge,
        MINIMAX_H3_DISAGG_CANONICAL_FIELD: _json_copy(
            canonical, name="canonical request"
        ),
        MINIMAX_H3_DISAGG_PLAN_FIELD: _json_copy(
            msgspec.to_builtins(plan), name="resolved plan"
        ),
        MINIMAX_H3_DISAGG_EXPLICIT_FIELDS_FIELD: _explicit_sampling_fields(req),
        MINIMAX_H3_DISAGG_RELEASE_IDENTITY_FIELD: _json_copy(
            release_identity, name="release identity"
        ),
    }


def _require_finite_tensor(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype,
    ndim: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"MiniMax H3 {name} must be a tensor")
    if value.dtype is not dtype:
        raise ValueError(
            f"MiniMax H3 {name} must have dtype {dtype}, got {value.dtype}"
        )
    if value.ndim != ndim:
        raise ValueError(f"MiniMax H3 {name} must have rank {ndim}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"MiniMax H3 {name} contains NaN or Inf")
    return value


def _validate_request_against_plan(req: Req, plan: MiniMaxH3ResolvedPlan) -> None:
    shape = plan.shape
    expected = {
        "width": int(shape["width"]),
        "height": int(shape["height"]),
        "fps": int(shape["fps"]),
        "num_frames": int(shape["frame_count"]),
        "seed": int(plan.seed if plan.seed is not None else 42),
    }
    for name, wanted in expected.items():
        actual = getattr(req, name, None)
        if isinstance(actual, bool) or actual != wanted:
            raise ValueError(
                f"MiniMax H3 disaggregation {name} mismatch: "
                f"expected {wanted!r}, got {actual!r}"
            )
    sampling_task = getattr(req.sampling_params, "task", None)
    if sampling_task != "t2va":
        raise ValueError(
            "MiniMax H3 disaggregation sampling task must be 't2va', got "
            f"{sampling_task!r}"
        )
    if getattr(req, "prompt", None) != plan.prompt:
        raise ValueError(
            "MiniMax H3 disaggregation prompt does not match resolved plan"
        )


def _restore_common(
    req: Req,
    *,
    edge: str,
    scalar_fields: dict[str, Any],
    release_identity: Mapping[str, str],
    extra_scalar_fields: set[str] | None = None,
) -> MiniMaxH3ResolvedPlan:
    if not isinstance(release_identity, Mapping) or not release_identity:
        raise ValueError("MiniMax H3 disaggregation release identity is unavailable")
    expected_fields = _COMMON_SCALAR_FIELDS | (extra_scalar_fields or set())
    if set(scalar_fields) != expected_fields:
        missing = sorted(expected_fields - set(scalar_fields))
        unknown = sorted(set(scalar_fields) - expected_fields)
        raise ValueError(
            "MiniMax H3 disaggregation scalar boundary mismatch: "
            f"missing={missing!r}, unknown={unknown!r}"
        )
    if scalar_fields[MINIMAX_H3_DISAGG_SCHEMA_FIELD] != MINIMAX_H3_DISAGG_SCHEMA:
        raise ValueError("unsupported MiniMax H3 disaggregation boundary schema")
    if scalar_fields[MINIMAX_H3_DISAGG_EDGE_FIELD] != edge:
        raise ValueError("MiniMax H3 disaggregation boundary edge mismatch")
    transferred_identity = scalar_fields[MINIMAX_H3_DISAGG_RELEASE_IDENTITY_FIELD]
    expected_identity = _json_copy(release_identity, name="release identity")
    if transferred_identity != expected_identity:
        raise ValueError("MiniMax H3 disaggregation release identity mismatch")

    canonical = scalar_fields[MINIMAX_H3_DISAGG_CANONICAL_FIELD]
    if not isinstance(canonical, Mapping):
        raise ValueError("MiniMax H3 canonical boundary field must be a mapping")
    plan = minimax_h3_resolve_plan(canonical)
    if plan.task != "t2va" or plan.materials or canonical.get("conditions") != []:
        raise ValueError(
            "MiniMax H3 disaggregation boundary v1 only restores text-only t2va"
        )
    transferred_plan = scalar_fields[MINIMAX_H3_DISAGG_PLAN_FIELD]
    expected_plan = _json_copy(msgspec.to_builtins(plan), name="resolved plan")
    if expected_plan != transferred_plan:
        raise ValueError(
            "MiniMax H3 canonical request and transferred resolved plan disagree"
        )

    req.extra[MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY] = _json_copy(
        canonical, name="canonical request"
    )
    req.extra[MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = plan
    explicit_fields = scalar_fields[MINIMAX_H3_DISAGG_EXPLICIT_FIELDS_FIELD]
    if not isinstance(explicit_fields, list) or any(
        not isinstance(name, str) or not name for name in explicit_fields
    ):
        raise ValueError(
            "MiniMax H3 explicit sampling boundary field must be a string list"
        )
    req.sampling_params._explicit_fields = set(explicit_fields)
    _validate_request_against_plan(req, plan)
    return plan


def filter_minimax_h3_disagg_transfer_fields(
    *,
    source_role: RoleType,
    destination_role: RoleType,
    tensor_fields: dict[str, Any],
    scalar_fields: dict[str, Any],
) -> None:
    """Remove implicit state superseded by the explicit H3 boundary schema."""
    edge = _edge(source_role, destination_role)
    for name in list(scalar_fields):
        if name.startswith("_extra_"):
            scalar_fields.pop(name)
    if edge == _ENCODER_TO_DENOISER:
        tensor_fields.clear()
    else:
        for name in list(tensor_fields):
            if name not in {"latents", "audio_latents"}:
                tensor_fields.pop(name)


def export_minimax_h3_disagg_boundary(
    req: Req,
    *,
    source_role: RoleType,
    destination_role: RoleType,
    release_identity: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    edge = _edge(source_role, destination_role)
    scalar_fields = _common_scalars(req, edge, release_identity)
    tensor_fields: dict[str, Any] = {}

    if edge == _ENCODER_TO_DENOISER:
        payload = req.extra.get(MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY)
        positive = payload.get("positive") if isinstance(payload, Mapping) else None
        if not isinstance(positive, Mapping):
            raise ValueError(
                "MiniMax H3 encoder boundary requires positive text embeddings"
            )
        hidden = _require_finite_tensor(
            positive.get("hidden_states"),
            name="text hidden states",
            dtype=torch.bfloat16,
            ndim=2,
        )
        if hidden.shape[1] != 5120:
            raise ValueError("MiniMax H3 text hidden states must have hidden size 5120")
        tags = _require_finite_tensor(
            positive.get("text_token_tags"),
            name="text token tags",
            dtype=torch.int64,
            ndim=1,
        )
        text_len = positive.get("text_len")
        if (
            isinstance(text_len, bool)
            or not isinstance(text_len, int)
            or text_len <= 0
            or text_len != hidden.shape[0]
            or text_len != tags.shape[0]
        ):
            raise ValueError(
                "MiniMax H3 text_len must match hidden states and token tags"
            )
        tensor_fields[MINIMAX_H3_DISAGG_TEXT_HIDDEN_FIELD] = hidden
        tensor_fields[MINIMAX_H3_DISAGG_TEXT_TAGS_FIELD] = tags
        scalar_fields[MINIMAX_H3_DISAGG_TEXT_LEN_FIELD] = text_len
        video_mask = positive.get("text_video_token_mask")
        if video_mask is not None:
            video_mask = _require_finite_tensor(
                video_mask,
                name="text video token mask",
                dtype=torch.bool,
                ndim=1,
            )
            if video_mask.shape[0] != text_len:
                raise ValueError(
                    "MiniMax H3 text video token mask length must match text_len"
                )
            tensor_fields[MINIMAX_H3_DISAGG_TEXT_VIDEO_MASK_FIELD] = video_mask
    else:
        plan = minimax_h3_plan_from_batch(req)
        assert plan is not None
        expected_video_shape = (
            1,
            24,
            int(plan.shape["video_latent_t"]),
            int(plan.shape["height"]) // 16,
            int(plan.shape["width"]) // 16,
        )
        expected_audio_shape = (2, 32, int(plan.shape["audio_latent_t"]))
        video = _require_finite_tensor(
            req.latents,
            name="final video latent",
            dtype=torch.float32,
            ndim=5,
        )
        audio = _require_finite_tensor(
            req.audio_latents,
            name="final audio latent",
            dtype=torch.float32,
            ndim=3,
        )
        if tuple(video.shape) != expected_video_shape:
            raise ValueError(
                "MiniMax H3 final video latent shape mismatch: "
                f"expected {expected_video_shape!r}, got {tuple(video.shape)!r}"
            )
        if tuple(audio.shape) != expected_audio_shape:
            raise ValueError(
                "MiniMax H3 final audio latent shape mismatch: "
                f"expected {expected_audio_shape!r}, got {tuple(audio.shape)!r}"
            )
    return tensor_fields, scalar_fields


def restore_minimax_h3_disagg_boundary(
    req: Req,
    *,
    source_role: RoleType,
    destination_role: RoleType,
    tensor_fields: dict[str, Any],
    scalar_fields: dict[str, Any],
    release_identity: Mapping[str, str],
) -> None:
    edge = _edge(source_role, destination_role)
    if edge == _ENCODER_TO_DENOISER:
        allowed_tensor_fields = {
            MINIMAX_H3_DISAGG_TEXT_HIDDEN_FIELD,
            MINIMAX_H3_DISAGG_TEXT_TAGS_FIELD,
            MINIMAX_H3_DISAGG_TEXT_VIDEO_MASK_FIELD,
        }
        required_tensor_fields = {
            MINIMAX_H3_DISAGG_TEXT_HIDDEN_FIELD,
            MINIMAX_H3_DISAGG_TEXT_TAGS_FIELD,
        }
        if (
            not required_tensor_fields <= set(tensor_fields)
            or not set(tensor_fields) <= allowed_tensor_fields
        ):
            raise ValueError(
                "MiniMax H3 encoder boundary tensor fields are incomplete or unknown"
            )
        _restore_common(
            req,
            edge=edge,
            scalar_fields=scalar_fields,
            release_identity=release_identity,
            extra_scalar_fields={MINIMAX_H3_DISAGG_TEXT_LEN_FIELD},
        )
        hidden = _require_finite_tensor(
            tensor_fields[MINIMAX_H3_DISAGG_TEXT_HIDDEN_FIELD],
            name="text hidden states",
            dtype=torch.bfloat16,
            ndim=2,
        )
        tags = _require_finite_tensor(
            tensor_fields[MINIMAX_H3_DISAGG_TEXT_TAGS_FIELD],
            name="text token tags",
            dtype=torch.int64,
            ndim=1,
        )
        text_len = scalar_fields[MINIMAX_H3_DISAGG_TEXT_LEN_FIELD]
        if (
            isinstance(text_len, bool)
            or not isinstance(text_len, int)
            or tuple(hidden.shape) != (text_len, 5120)
            or tuple(tags.shape) != (text_len,)
        ):
            raise ValueError("MiniMax H3 restored text conditioning shape mismatch")
        positive = {
            "hidden_states": hidden,
            "text_len": text_len,
            "text_token_tags": tags,
        }
        video_mask = tensor_fields.get(MINIMAX_H3_DISAGG_TEXT_VIDEO_MASK_FIELD)
        if video_mask is not None:
            video_mask = _require_finite_tensor(
                video_mask,
                name="text video token mask",
                dtype=torch.bool,
                ndim=1,
            )
            if tuple(video_mask.shape) != (text_len,):
                raise ValueError(
                    "MiniMax H3 restored text video token mask shape mismatch"
                )
            positive["text_video_token_mask"] = video_mask
        req.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY] = {"positive": positive}
        req.prompt_embeds = [hidden]
        req.prompt_seq_lens = [[text_len]]
        return

    if tensor_fields or set(scalar_fields) != _COMMON_SCALAR_FIELDS:
        raise ValueError(
            "MiniMax H3 denoiser boundary contains unexpected model fields"
        )
    plan = _restore_common(
        req,
        edge=edge,
        scalar_fields=scalar_fields,
        release_identity=release_identity,
    )
    expected_video_shape = (
        1,
        24,
        int(plan.shape["video_latent_t"]),
        int(plan.shape["height"]) // 16,
        int(plan.shape["width"]) // 16,
    )
    expected_audio_shape = (2, 32, int(plan.shape["audio_latent_t"]))
    video = _require_finite_tensor(
        req.latents,
        name="final video latent",
        dtype=torch.float32,
        ndim=5,
    )
    audio = _require_finite_tensor(
        req.audio_latents,
        name="final audio latent",
        dtype=torch.float32,
        ndim=3,
    )
    if tuple(video.shape) != expected_video_shape:
        raise ValueError("MiniMax H3 restored final video latent shape mismatch")
    if tuple(audio.shape) != expected_audio_shape:
        raise ValueError("MiniMax H3 restored final audio latent shape mismatch")


__all__ = [
    "MINIMAX_H3_DISAGG_CANONICAL_FIELD",
    "MINIMAX_H3_DISAGG_EDGE_FIELD",
    "MINIMAX_H3_DISAGG_EXPLICIT_FIELDS_FIELD",
    "MINIMAX_H3_DISAGG_PLAN_FIELD",
    "MINIMAX_H3_DISAGG_RELEASE_IDENTITY_FIELD",
    "MINIMAX_H3_DISAGG_SCHEMA",
    "MINIMAX_H3_DISAGG_SCHEMA_FIELD",
    "MINIMAX_H3_DISAGG_TEXT_HIDDEN_FIELD",
    "MINIMAX_H3_DISAGG_TEXT_LEN_FIELD",
    "MINIMAX_H3_DISAGG_TEXT_TAGS_FIELD",
    "MINIMAX_H3_DISAGG_TEXT_VIDEO_MASK_FIELD",
    "export_minimax_h3_disagg_boundary",
    "filter_minimax_h3_disagg_transfer_fields",
    "restore_minimax_h3_disagg_boundary",
]
