# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from sglang.multimodal_gen.runtime.utils import perf_logger


def test_perf_log_dir_does_not_require_package_file(monkeypatch, tmp_path):
    monkeypatch.delenv("SGLANG_PERF_LOG_DIR", raising=False)
    expected = (
        Path(perf_logger.__file__).resolve().parents[3] / "../../.cache/logs"
    ).resolve()

    assert Path(perf_logger.get_diffusion_perf_log_dir()) == expected

    override = tmp_path / "perf"
    monkeypatch.setenv("SGLANG_PERF_LOG_DIR", str(override))
    assert Path(perf_logger.get_diffusion_perf_log_dir()) == override
