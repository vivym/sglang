# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTConfig
from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# VAE layerwise offload 在 PCIe Gen2 上是灾难性慢（本机实测 decode 10.3s -> 141s，
# 13.7x），主因是逐层同步开销而非带宽。H3 强制 video/audio VAE 常驻。
_MINIMAX_H3_VAE_OFFLOAD_NAMES = ("vae", "video_vae", "audio_vae")


@dataclass
class MiniMaxH3PipelineConfig(PipelineConfig):
    """MiniMax H3 native audio-video pipeline configuration."""

    # Canonical H3 materials are prepared by the model-specific stages. The
    # generic TI2V image resize would both duplicate that work and overwrite
    # the already-resolved target canvas.
    skip_input_image_preprocess: bool = True
    native_only_components = (
        "text_encoder",
        "transformer",
        "video_vae",
        "audio_vae",
    )
    task_type: ModelTaskType = ModelTaskType.TI2V
    dit_config: MiniMaxH3DiTConfig = field(default_factory=MiniMaxH3DiTConfig)
    vae_config: MiniMaxH3VideoVAEConfig = field(default_factory=MiniMaxH3VideoVAEConfig)
    audio_vae_config: MiniMaxH3AudioVAEConfig = field(
        default_factory=MiniMaxH3AudioVAEConfig
    )
    dit_precision: str = "bf16"
    # The video VAE remains fp32-resident because it also encodes keyframes.
    # Decode follows the released fp16-autocast recipe unless the user
    # explicitly disables autocast.
    vae_precision: str = "fp32"
    vae_decode_precision: str = "fp16"
    audio_vae_precision: str = "fp32"
    text_encoder_configs: tuple[MiniMaxH3Qwen3VLConfig, ...] = field(
        default_factory=lambda: (MiniMaxH3Qwen3VLConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_extra_args: list[dict] = field(default_factory=lambda: [{}])
    # The released checkpoint is CFG-distilled and has one positive branch.
    should_use_guidance: bool = False
    output_audio_sample_rate: int | None = 32000
    output_audio_channels: int | None = 2
    output_av_drift_tolerance_s: float | None = 0.25

    def accepts_audio_input(self) -> bool:
        return True

    def supports_disaggregation(self) -> bool:
        return False

    @property
    def requires_audio_output(self) -> bool:
        return True

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            speed_mode_enable_torch_compile_by_default=False,
            keep_resident_min_available_gb=120,
            keep_resident_components=("dit", "text_encoder", "vae"),
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
        )

    @staticmethod
    def _server_arg_value(value):
        return getattr(value, "value", value)

    def validate_quality_deployment(self, server_args) -> None:
        """Fail closed unless the resident server matches the deployment
        audited for quality="high"."""

        attention_backend = self._server_arg_value(server_args.attention_backend)
        attention_backend = (
            str(attention_backend).strip().lower()
            if attention_backend is not None
            else None
        )
        capability = current_platform.get_device_capability()
        capability_int = capability.to_int() if capability is not None else None
        device_name = (
            current_platform.get_device_name()
            if current_platform.is_cuda()
            else type(current_platform).__name__
        )
        model_variant = str(server_args.model_variant or "fl2va").lower()
        actual = {
            "attention_backend": attention_backend,
            "backend": self._server_arg_value(server_args.backend),
            "component_attention_backends": {},
            "enable_breakable_cuda_graph": server_args.enable_breakable_cuda_graph,
            "enable_torch_compile": server_args.enable_torch_compile,
            "is_dit_layerwise_offload_selected": (
                server_args.is_dit_layerwise_offload_selected
            ),
            "model_variant": model_variant,
            "num_gpus": server_args.num_gpus,
            "performance_mode": server_args.performance_mode,
            "quantization": server_args.quantization,
            "regional_compile": server_args.regional_compile,
            "ring_degree": server_args.ring_degree,
            "sp_degree": server_args.sp_degree,
            "tp_size": server_args.tp_size,
            "ulysses_degree": server_args.ulysses_degree,
            "use_fsdp_inference": server_args.use_fsdp_inference,
        }
        actual["component_attention_backends"] = dict(
            server_args.component_attention_backends or {}
        )
        expected = {
            "attention_backend": {None, "fa"},
            "backend": {"auto", "sglang"},
            "component_attention_backends": {},
            "enable_breakable_cuda_graph": False,
            "enable_torch_compile": False,
            "is_dit_layerwise_offload_selected": False,
            "model_variant": "fl2va",
            "num_gpus": 4,
            "performance_mode": "speed",
            "quantization": None,
            "regional_compile": False,
            "ring_degree": 1,
            "sp_degree": 4,
            "tp_size": 1,
            "ulysses_degree": 4,
            "use_fsdp_inference": False,
        }
        mismatches = {
            name: {"expected": wanted, "actual": actual[name]}
            for name, wanted in expected.items()
            if (
                actual[name] not in wanted
                if isinstance(wanted, set)
                else actual[name] != wanted
            )
        }
        if (
            not current_platform.is_cuda()
            or "H200" not in device_name.upper()
            or capability_int != 90
        ):
            mismatches["device"] = {
                "expected": "NVIDIA H200 (compute capability 9.0)",
                "actual": f"{device_name} (compute capability {capability_int})",
            }
        if mismatches:
            raise ValueError(
                'MiniMax-H3 quality="high" is validated only for '
                f"the strict 4xH200 fl2va deployment; mismatches: {mismatches}"
            )

    def validate_server_args(self, server_args) -> None:
        # H3 的 video/audio VAE 必须常驻：VAE layerwise offload 在 PCIe Gen2 上
        # 因逐层同步开销导致 decode ~13.7x 慢（实测 10.3s -> 141s）。这里从所有
        # offload 选择里剥掉 VAE，防止后续实验误触。
        self._force_vae_resident(server_args)
        # Reject known-inexact VAE modes before any large component download.
        self.vae_config.resolved_parallel_decode_mode()
        component_backends = server_args.component_attention_backends or {}
        attention_backend = component_backends.get(
            "transformer", self._server_arg_value(server_args.attention_backend)
        )
        if attention_backend is None:
            return
        selected_backend = (
            attention_backend
            if isinstance(attention_backend, AttentionBackendEnum)
            else AttentionBackendEnum[str(attention_backend).strip().upper()]
        )
        get_attn_backend(
            self.dit_config.arch_config.attention_head_dim,
            torch.bfloat16,
            selected_attention_backend=selected_backend,
            attention_requirements=AttentionRequirements(packed_varlen=True),
        )

    def _force_vae_resident(self, server_args) -> None:
        """Strip VAE offload from every offload selector.

        The video VAE decoder's layerwise offload is ~13.7x slower than resident
        on PCIe Gen2 (per-layer sync overhead, not bandwidth), so MiniMax-H3 pins
        the VAE resident regardless of ``performance_mode`` or explicit flags.

        诊断逃生门：``MINIMAX_H3_FORCE_VAE_RESIDENT=0`` 可关闭强制常驻（仅用于
        对照实验，如 AdaLN on/off 需要腾内存的场景），默认强制。
        """
        if (
            os.environ.get("MINIMAX_H3_FORCE_VAE_RESIDENT", "1").strip().lower()
            in ("0", "false", "no", "off")
        ):
            logger.warning(
                "MiniMax-H3 VAE force-resident DISABLED via "
                "MINIMAX_H3_FORCE_VAE_RESIDENT=0 (diagnostic only)."
            )
            return
        changed = []

        components = server_args.layerwise_offload_components
        if components:
            stripped = [
                name for name in components if name not in _MINIMAX_H3_VAE_OFFLOAD_NAMES
            ]
            if len(stripped) != len(components):
                server_args.layerwise_offload_components = stripped or None
                changed.append(
                    f"layerwise_offload_components={server_args.layerwise_offload_components}"
                )

        if server_args.vae_cpu_offload:
            server_args.vae_cpu_offload = False
            changed.append("vae_cpu_offload=False")

        cpu_components = server_args.cpu_offload_components
        if cpu_components:
            stripped = [
                name
                for name in cpu_components
                if name not in _MINIMAX_H3_VAE_OFFLOAD_NAMES
            ]
            if len(stripped) != len(cpu_components):
                server_args.cpu_offload_components = stripped or None
                changed.append(
                    f"cpu_offload_components={server_args.cpu_offload_components}"
                )

        if changed:
            logger.warning(
                "MiniMax-H3 forces the video/audio VAE resident (VAE offload is "
                "~13.7x slower on PCIe Gen2): %s",
                "; ".join(changed),
            )

    def select_vae_weight_files(
        self,
        safetensors_list: list[str],
        component_model_path: str,
        component_name: str,
        vae_precision: str,
    ) -> list[str]:
        if component_name == "video_vae":
            return [os.path.join(component_model_path, "source", "model.safetensors")]
        return safetensors_list


__all__ = ["MiniMaxH3PipelineConfig"]
