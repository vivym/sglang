# SPDX-License-Identifier: Apache-2.0
"""ConvRot 旋转辅助（权重/激活的 regular 正交 Hadamard 旋转）。

参考实现：Comfy-Org/comfy-kitchen 的 ``comfy_kitchen/tensor/int8_utils.py``
（官方 ComfyUI 组织的 kernel 库）。ConvRot 论文：arXiv 2512.03673。

要点：
- 使用 "regular" 正交 Hadamard（size 必须是 4 的幂，如 256），基矩阵为特定 H4；
- 权重离线旋转 W_rot = W @ H^T（按 group 分组）；
- 激活在线旋转 x_rot = x @ H（按 group 分组）；
- H 自逆（H·H = I），所以旋转数学上无损，但能抹平行/列双向的激活 outlier。
"""

from __future__ import annotations

import math

import torch

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

CONVROT_GROUP_SIZE = 256


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """构建归一化的 regular 正交 Hadamard 矩阵（与 comfy-kitchen 一致）。"""
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )

    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4

    h_normalized = h / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_normalized
    return h_normalized


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """离线旋转权重：W_rot = W @ H^T（按 group 分组）。"""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size
    weight_grouped = weight.reshape(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    weight_rotated = torch.matmul(weight_grouped, h_t)
    return weight_rotated.reshape(out_f, in_f)


def rotate_activation(x: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """在线旋转激活：x_rot = x @ H（按 group 分组）。"""
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size
    x_grouped = x.reshape(-1, n_groups, group_size)
    h = h.to(dtype=x.dtype, device=x.device)
    x_rotated = torch.matmul(x_grouped, h)
    return x_rotated.reshape(orig_shape)
