# SPDX-License-Identifier: Apache-2.0
"""Fail-closed internal tensor artifacts for MiniMax H3 equivalence gates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from sglang.multimodal_gen.runtime.disaggregation.boundary import (
    DISAGG_ATTEMPT_ID_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType


MINIMAX_H3_DEBUG_TENSOR_DUMP_SCHEMA = "minimax-h3.debug-tensor-dump/v1"
MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV = "MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT"
MINIMAX_H3_DEBUG_TENSOR_METADATA_KEY = "__metadata__"
MINIMAX_H3_DEBUG_TENSOR_NAMES = (
    "initial_video_rows",
    "initial_audio_rows",
    "text_hidden_states",
    "refined_prompt_embeds",
    "latents",
    "audio_latents",
)
_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _identity_segment(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTITY_RE.fullmatch(value) is None:
        raise ValueError(
            f"MiniMax H3 debug tensor {name} must match {_IDENTITY_RE.pattern!r}"
        )
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _role_name(server_args: Any) -> str:
    role = getattr(server_args, "disagg_role", RoleType.MONOLITHIC)
    value = getattr(role, "value", role)
    return _identity_segment(str(value), name="role")


def _model_identity(pipeline: Any, server_args: Any) -> dict[str, Any]:
    release_identity = getattr(pipeline, "debug_release_identity", None)
    if release_identity is None:
        release_identity = getattr(pipeline, "disagg_release_identity", None)
    if isinstance(release_identity, Mapping) and release_identity:
        role = _role_name(server_args)
        return {
            "kind": (
                "verified_monolithic_release"
                if role == RoleType.MONOLITHIC.value
                else "verified_disaggregated_release"
            ),
            "release_identity": dict(release_identity),
        }
    raise ValueError(
        "MiniMax H3 debug tensors require a verified production release identity"
    )


def _workload_identity(batch: Any) -> dict[str, Any]:
    sampling = getattr(batch, "sampling_params", None)
    prompt = getattr(batch, "prompt", None)
    if prompt is None and sampling is not None:
        prompt = getattr(sampling, "prompt", None)
    prompt_sha256 = (
        "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if isinstance(prompt, str)
        else None
    )
    return {
        "task": getattr(batch, "task", None),
        "width": getattr(batch, "width", None),
        "height": getattr(batch, "height", None),
        "fps": getattr(batch, "fps", None),
        "num_frames": getattr(batch, "num_frames", None),
        "num_inference_steps": getattr(batch, "num_inference_steps", None),
        "flow_shift": getattr(batch, "flow_shift", None),
        "audio_flow_shift": getattr(batch, "audio_flow_shift", None),
        "prompt_sha256": prompt_sha256,
    }


def _atomic_save_exclusive(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Publish a complete inode without replacing an artifact from an
        # earlier attempt or concurrent worker.
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_minimax_h3_debug_tensor_dump(
    *,
    batch: Any,
    tensors: Mapping[str, torch.Tensor],
    server_args: Any,
    pipeline: Any,
) -> Path | None:
    """Write one immutable, identity-bound denoiser artifact when enabled."""

    root_value = os.environ.get(MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV, "").strip()
    if not root_value:
        return None
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        raise ValueError(
            f"{MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV} must be an absolute path"
        )
    root = root.resolve()

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None

    request_id = _identity_segment(
        getattr(batch, "request_id", None), name="request_id"
    )
    role = _role_name(server_args)
    if role not in {RoleType.MONOLITHIC.value, RoleType.DENOISER.value}:
        raise ValueError(
            "MiniMax H3 debug tensors can only be written by monolithic or "
            "denoiser roles"
        )
    raw_attempt_id = getattr(batch, "extra", {}).get(DISAGG_ATTEMPT_ID_EXTRA_KEY)
    if role == RoleType.MONOLITHIC.value:
        if raw_attempt_id is not None:
            raise ValueError(
                "monolithic MiniMax H3 debug dump cannot carry a disagg attempt ID"
            )
        attempt_id = "direct"
    else:
        attempt_id = _identity_segment(raw_attempt_id, name="attempt_id")

    if set(tensors) != set(MINIMAX_H3_DEBUG_TENSOR_NAMES):
        missing = sorted(set(MINIMAX_H3_DEBUG_TENSOR_NAMES) - set(tensors))
        unknown = sorted(set(tensors) - set(MINIMAX_H3_DEBUG_TENSOR_NAMES))
        raise ValueError(
            "MiniMax H3 debug tensor set mismatch: "
            f"missing={missing!r}, unknown={unknown!r}"
        )

    cpu_tensors: dict[str, torch.Tensor] = {}
    for name in MINIMAX_H3_DEBUG_TENSOR_NAMES:
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise ValueError(f"MiniMax H3 debug tensor {name!r} must be floating point")
        cpu_tensors[name] = tensor.detach().cpu().contiguous()

    workload = _workload_identity(batch)
    metadata = {
        "schema": MINIMAX_H3_DEBUG_TENSOR_DUMP_SCHEMA,
        "phase": "denoiser_output",
        "request_id": request_id,
        "attempt_id": attempt_id,
        "role": role,
        "seed": getattr(batch, "seed", None),
        "workload": workload,
        "workload_sha256": _canonical_sha256(workload),
        "model_identity": _model_identity(pipeline, server_args),
        "tensor_names": list(MINIMAX_H3_DEBUG_TENSOR_NAMES),
    }
    payload: dict[str, Any] = {
        MINIMAX_H3_DEBUG_TENSOR_METADATA_KEY: metadata,
        **cpu_tensors,
    }
    path = root / request_id / role / attempt_id / "denoiser-output.pt"
    if not path.resolve().is_relative_to(root):
        raise ValueError("MiniMax H3 debug tensor path escaped its configured root")
    _atomic_save_exclusive(payload, path)
    return path


__all__ = [
    "MINIMAX_H3_DEBUG_TENSOR_DUMP_ROOT_ENV",
    "MINIMAX_H3_DEBUG_TENSOR_DUMP_SCHEMA",
    "MINIMAX_H3_DEBUG_TENSOR_METADATA_KEY",
    "MINIMAX_H3_DEBUG_TENSOR_NAMES",
    "write_minimax_h3_debug_tensor_dump",
]
