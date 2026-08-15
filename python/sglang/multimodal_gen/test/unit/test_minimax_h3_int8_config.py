# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang.multimodal_gen.runtime.layers.quantization.int8 import (
    Int8Config,
    _convrot_enabled,
)


def test_convrot_config_requires_json_boolean():
    with pytest.raises(ValueError, match="JSON boolean"):
        Int8Config.from_config({"quant_method": "int8", "convrot": "false"})


def test_adaln_provenance_flows_through_int8_override_config():
    artifact = {
        "format_version": "1",
        "table_layout": "full",
        "source_fingerprint": "sha256:test",
    }
    config = Int8Config.from_config(
        {
            "quant_method": "int8",
            "convrot": True,
            "minimax_h3_adaln_table": artifact,
        }
    )
    assert config.minimax_h3_adaln_table == artifact

    with pytest.raises(ValueError, match="must be an object"):
        Int8Config.from_config(
            {
                "quant_method": "int8",
                "minimax_h3_adaln_table": "sha256:test",
            }
        )


def test_convrot_env_cannot_override_checkpoint_marker(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_CONVROT", "0")
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        _convrot_enabled(True)

    monkeypatch.setenv("MINIMAX_H3_CONVROT", "1")
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        _convrot_enabled(False)

    assert _convrot_enabled(True)
