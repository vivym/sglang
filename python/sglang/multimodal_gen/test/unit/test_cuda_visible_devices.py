# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from sglang.multimodal_gen.runtime.platforms import cuda


def test_numeric_cuda_visible_devices_preserve_reordering():
    with patch.dict(cuda.os.environ, {"CUDA_VISIBLE_DEVICES": "6, 2"}):
        assert cuda.device_id_to_physical_device_id(0) == 6
        assert cuda.device_id_to_physical_device_id(1) == 2


def test_uuid_cuda_visible_device_resolves_through_nvml():
    handle = object()
    with (
        patch.dict(cuda.os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test"}),
        patch.object(cuda.pynvml, "nvmlDeviceGetHandleByUUID", return_value=handle),
        patch.object(cuda.pynvml, "nvmlDeviceGetIndex", return_value=3),
    ):
        assert cuda.device_id_to_physical_device_id(0) == 3
        cuda.pynvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-test")


def test_cuda_visible_devices_rejects_out_of_range_logical_device():
    with (
        patch.dict(cuda.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}),
        pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"),
    ):
        cuda.device_id_to_physical_device_id(1)


def test_cuda_visible_devices_rejects_unknown_identifier():
    with (
        patch.dict(cuda.os.environ, {"CUDA_VISIBLE_DEVICES": "not-a-device"}),
        pytest.raises(ValueError, match="unsupported CUDA_VISIBLE_DEVICES entry"),
    ):
        cuda.device_id_to_physical_device_id(0)
