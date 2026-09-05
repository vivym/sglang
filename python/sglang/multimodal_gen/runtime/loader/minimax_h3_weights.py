# SPDX-License-Identifier: Apache-2.0
"""Checkpoint inspection for MiniMax-H3 transformer overrides."""

import json
from typing import Any

from safetensors import safe_open

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
)
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    build_nvfp4_config_from_safetensors_list,
    inspect_comfy_quant_markers,
    resolve_comfy_checkpoint_quantization,
)
from sglang.srt.model_loader.checkpoint_quantization import (
    resolve_checkpoint_quant_spec,
)


def comfy_quant_key_filter(name: str) -> bool:
    return not name.endswith(".comfy_quant")


def inspect_minimax_h3_safetensors(
    safetensors_list: list[str],
    *,
    legacy_int8: bool = False,
) -> tuple[tuple[int, int] | None, dict[str, dict[str, Any]]]:
    """Read H3 architecture metadata and Comfy per-layer format markers."""
    adaln_curve_shape = None
    layer_markers = inspect_comfy_quant_markers(
        safetensors_list,
        allow_missing_markers=legacy_int8,
    )
    if legacy_int8:
        if layer_markers:
            raise ValueError(
                "Legacy MiniMax-H3 INT8 checkpoints cannot mix .comfy_quant markers"
            )
        _validate_legacy_int8_tensors(safetensors_list)

    for path in safetensors_list:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = checkpoint.keys()
            if "adaln_t_table" in keys:
                shape = tuple(checkpoint.get_slice("adaln_t_table").get_shape())
                if len(shape) != 2 or shape[0] < 2:
                    raise ValueError(
                        "MiniMax-H3 adaln_t_table must have shape [N, D] with "
                        f"N >= 2, got {shape} in {path}"
                    )
                if adaln_curve_shape is not None and adaln_curve_shape != shape:
                    raise ValueError(
                        "MiniMax-H3 checkpoint shards disagree on adaln_t_table "
                        f"shape: {adaln_curve_shape} vs {shape}"
                    )
                adaln_curve_shape = shape

    return adaln_curve_shape, layer_markers


def is_legacy_minimax_h3_int8_config(config_path: str | None) -> bool:
    """Recognize only the validated pre-Comfy ConvRot checkpoint declaration."""
    if config_path is None:
        return False
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    quant_spec = resolve_checkpoint_quant_spec(config)
    if quant_spec is None or quant_spec.declared_method != "int8":
        return False
    convrot = quant_spec.config.get("convrot")
    if convrot is not True:
        raise ValueError("Legacy MiniMax-H3 quant_method='int8' requires convrot=true")
    return True


def _validate_legacy_int8_tensors(safetensors_list: list[str]) -> None:
    quantized_layers = 0
    for path in safetensors_list:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = set(checkpoint.keys())
            if any(key.endswith(".comfy_quant") for key in keys):
                raise ValueError(
                    "Legacy MiniMax-H3 INT8 checkpoints cannot contain "
                    ".comfy_quant markers"
                )
            for key in keys:
                weight = checkpoint.get_slice(key)
                if not key.endswith(".weight") or weight.get_dtype() != "I8":
                    continue
                quantized_layers += 1
                scale_key = key.removesuffix(".weight") + ".weight_scale"
                if scale_key not in keys:
                    raise ValueError(
                        f"Legacy MiniMax-H3 INT8 tensor {key!r} is missing "
                        f"{scale_key!r}"
                    )
                weight_shape = tuple(weight.get_shape())
                scale = checkpoint.get_slice(scale_key)
                if (
                    len(weight_shape) != 2
                    or scale.get_dtype() != "F32"
                    or tuple(scale.get_shape()) != (weight_shape[0], 1)
                ):
                    raise ValueError(
                        "Legacy MiniMax-H3 INT8 weight/scale metadata mismatch for "
                        f"{key!r}: weight={weight.get_dtype()}{weight_shape}, "
                        f"scale={scale.get_dtype()}{tuple(scale.get_shape())}"
                    )
    if quantized_layers == 0:
        raise ValueError("Legacy MiniMax-H3 INT8 checkpoint contains no I8 weights")


def resolve_minimax_h3_checkpoint_quantization(
    layer_markers: dict[str, dict[str, Any]],
    safetensors_list: list[str] | None = None,
    param_names_mapping: dict | None = None,
    reverse_param_names_mapping: dict | None = None,
) -> QuantizationConfig | None:
    formats = {str(marker.get("format")) for marker in layer_markers.values()}
    if "nvfp4" in formats:
        unsupported = formats - {"nvfp4", "int8_tensorwise", "float8_e4m3fn"}
        if unsupported:
            raise NotImplementedError(
                "Unsupported Comfy NVFP4 companion format(s): "
                + ", ".join(sorted(unsupported))
            )
        if safetensors_list is None:
            raise ValueError("MiniMax-H3 NVFP4 metadata requires checkpoint files")
        config = build_nvfp4_config_from_safetensors_list(
            safetensors_list,
            param_names_mapping,
            reverse_param_names_mapping,
        )
        if not isinstance(config, ModelOptFp4Config):
            raise ValueError("Could not resolve MiniMax-H3 NVFP4 checkpoint layout")
        config.set_comfy_layer_markers(layer_markers)
        config.checkpoint_uses_comfy_quantization = True
        config.checkpoint_uses_native_qkv_layout = True
        config.checkpoint_weight_scale_layout = "swizzled"
        config.swap_weight_nibbles = True
        return config
    return resolve_comfy_checkpoint_quantization(layer_markers)


def validate_minimax_h3_checkpoint_variant(
    checkpoint_paths: list[str], selected_variant: str
) -> None:
    names = " ".join(path.lower() for path in checkpoint_paths)
    checkpoint_variants = {
        variant for variant in ("fl2va", "ref2va") if variant in names
    }
    if (
        len(checkpoint_variants) == 1
        and selected_variant.lower() not in checkpoint_variants
    ):
        (checkpoint_variant,) = checkpoint_variants
        raise ValueError(
            f"MiniMax-H3 checkpoint variant {checkpoint_variant!r} does not match "
            f"--model-variant {selected_variant!r}"
        )


__all__ = [
    "comfy_quant_key_filter",
    "inspect_minimax_h3_safetensors",
    "is_legacy_minimax_h3_int8_config",
    "resolve_minimax_h3_checkpoint_quantization",
    "validate_minimax_h3_checkpoint_variant",
]
