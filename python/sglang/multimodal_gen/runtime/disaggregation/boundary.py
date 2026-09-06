# SPDX-License-Identifier: Apache-2.0
"""Model-specific state carried across diffusion disaggregation boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DISAGG_BOUNDARY_FIELD_PREFIX = "_disagg_boundary_"
DISAGG_ATTEMPT_ID_EXTRA_KEY = "_sglang_disagg_attempt_id"


def validate_boundary_fields(fields: Mapping[str, Any], *, kind: str) -> None:
    """Require model boundary fields to stay in their reserved namespace."""
    if not isinstance(fields, Mapping):
        raise TypeError(f"disaggregation boundary {kind} fields must be a mapping")
    for name in fields:
        if not isinstance(name, str) or not name.startswith(
            DISAGG_BOUNDARY_FIELD_PREFIX
        ):
            raise ValueError(
                f"disaggregation boundary {kind} field {name!r} must start with "
                f"{DISAGG_BOUNDARY_FIELD_PREFIX!r}"
            )


def partition_boundary_fields(fields: Mapping[str, Any]) -> tuple[dict, dict]:
    """Split ordinary request fields from reserved model-boundary fields."""
    ordinary = {}
    boundary = {}
    for name, value in fields.items():
        target = (
            boundary
            if isinstance(name, str) and name.startswith(DISAGG_BOUNDARY_FIELD_PREFIX)
            else ordinary
        )
        target[name] = value
    return ordinary, boundary


__all__ = [
    "DISAGG_ATTEMPT_ID_EXTRA_KEY",
    "DISAGG_BOUNDARY_FIELD_PREFIX",
    "partition_boundary_fields",
    "validate_boundary_fields",
]
