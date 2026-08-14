# SPDX-License-Identifier: Apache-2.0
"""W8A8 INT8 quantization method for the diffusion runtime (fast-h3 定制).

在线量化约定（权重以源 dtype 加载、process_weights_after_loading 中量化），
建模自 diffusion 侧 fp8.py 与 srt 侧 w8a8_int8.py：
- 权重：per-channel 对称量化（每输出行 amax / 127），存 (K, N) int8
- 激活：apply 时 per-token 动态量化
- GEMM：sgl_kernel.int8_scaled_mm（sm80 CUTLASS IMMA 路径）
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import ModelWeightParameter
from sglang.srt.layers.quantization.utils import is_layer_skipped

try:
    from sgl_kernel import int8_scaled_mm
except ImportError:  # pragma: no cover
    int8_scaled_mm = None


def _convrot_enabled() -> bool:
    """ConvRot Hadamard 旋转开关。必须与 checkpoint 的权重旋转（build_int8_transformer.py
    的 --convrot）保持一致，否则权重/激活旋转不匹配会输出错误。"""
    value = os.environ.get("MINIMAX_H3_CONVROT", "0")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class Int8Config(QuantizationConfig):
    """W8A8 INT8 在线量化配置（扩散运行时）。

    无参构造 = 在线量化（权重加载后量化）；--quantization-ignored-layers
    传入需豁免的层（子串匹配，与 fp8 约定一致）。
    """

    def __init__(
        self,
        ignored_layers: Optional[List[str]] = None,
        is_checkpoint_int8_serialized: bool = False,
    ):
        super().__init__()
        self.ignored_layers = ignored_layers or []
        self.is_checkpoint_int8_serialized = is_checkpoint_int8_serialized
        # H3 的 fused 权重直接按完整名字匹配（如 blocks.0.attn.qkv_proj），无拆分映射
        self.packed_modules_mapping: Dict[str, List[str]] = {}

    @classmethod
    def get_name(cls) -> str:
        return "int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # sm80 IMMA（CMP 170HX）
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Int8Config":
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        if ignored_layers:
            ignored_layers = [layer.replace("model.", "") for layer in ignored_layers]
        # HF/sglang 约定字段名为 quant_method（非 quantization_method）
        quant_method = config.get("quant_method", "")
        is_serialized = "int8" in quant_method
        return cls(
            ignored_layers=ignored_layers,
            is_checkpoint_int8_serialized=is_serialized,
        )

    def get_quant_method(
        self, layer: Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            return Int8LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Int8LinearMethod(QuantizeMethodBase):
    """W8A8 INT8 在线量化线性方法（扩散运行时）。"""

    def __init__(self, quant_config: Int8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # 序列化 checkpoint：直接建 int8 权重 + scale；否则以源 dtype 创建，
        # 量化在 process_weights_after_loading
        weight_dtype = (
            torch.int8
            if self.quant_config.is_checkpoint_int8_serialized
            else params_dtype
        )
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.quant_config.is_checkpoint_int8_serialized:
            # per-channel scale (N, 1)，从 checkpoint 加载
            weight_scale = ModelWeightParameter(
                data=torch.empty(
                    (output_size_per_partition, 1),
                    dtype=torch.float32,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        if self.quant_config.is_checkpoint_int8_serialized:
            # 序列化 checkpoint：权重已量化，只需列主序视图
            layer.weight = Parameter(layer.weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(
                layer.weight_scale.data, requires_grad=False
            )
            return

        weight = layer.weight.data  # (N, K)，源 dtype

        # per-channel 对称量化：每输出行 amax / 127
        weight_fp = weight.to(torch.float32)
        scale = weight_fp.abs().amax(dim=1, keepdim=True) / 127.0  # (N, 1)
        scale = torch.clamp(scale, min=1e-12)
        qweight = torch.round(weight_fp / scale).clamp(-128, 127).to(torch.int8)

        # int8_scaled_mm 期望 mat_b 为列主序（.t() 视图，勿 .contiguous()——与 srt 约定一致）
        layer.weight = Parameter(qweight.t(), requires_grad=False)
        layer.weight_scale = Parameter(scale.to(torch.float32), requires_grad=False)

    def apply(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if int8_scaled_mm is None:
            raise ImportError("sgl_kernel 不可用：缺少 int8_scaled_mm")
        if _convrot_enabled():
            from sglang.multimodal_gen.runtime.layers.quantization.convrot import (
                CONVROT_GROUP_SIZE,
                build_hadamard,
                rotate_activation,
            )

            hadamard = build_hadamard(
                CONVROT_GROUP_SIZE, device=x.device, dtype=torch.float32
            )
            x = rotate_activation(x, hadamard, CONVROT_GROUP_SIZE)
        x_q, x_scale = per_token_quant_int8(x)

        x_q_2d = x_q.view(-1, x_q.shape[-1])
        x_scale_2d = x_scale.view(-1, x_scale.shape[-1])
        output_shape = [*x_q.shape[:-1], layer.weight.shape[1]]

        output = int8_scaled_mm(
            x_q_2d,
            layer.weight,
            x_scale_2d,
            layer.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )
        return output.view(output_shape)
