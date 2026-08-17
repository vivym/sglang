from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.text_encoding import (
    _apply_debug_text_hidden_states_override,
    _clone_text_embedding_payload,
)


def _batch_with_hidden_states(hidden_states: torch.Tensor):
    return SimpleNamespace(
        extra={
            MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY: {
                "positive": {
                    "hidden_states": hidden_states,
                    "text_len": int(hidden_states.shape[0]),
                }
            }
        }
    )


def test_debug_text_embedding_payload_clone_is_independent():
    source = {
        "positive": {
            "hidden_states": torch.arange(6).reshape(2, 3),
            "text_len": 2,
            "text_token_tags": torch.ones(2, dtype=torch.long),
        },
        "metadata": ("t2va", [1, 2]),
    }

    cloned = _clone_text_embedding_payload(source)

    assert cloned is not source
    assert cloned["positive"] is not source["positive"]
    assert torch.equal(
        cloned["positive"]["hidden_states"],
        source["positive"]["hidden_states"],
    )
    assert (
        cloned["positive"]["hidden_states"].data_ptr()
        != source["positive"]["hidden_states"].data_ptr()
    )
    cloned["positive"]["hidden_states"].zero_()
    assert torch.equal(
        source["positive"]["hidden_states"], torch.arange(6).reshape(2, 3)
    )


def test_debug_text_hidden_states_override_is_exact_and_independent(
    monkeypatch, tmp_path
):
    native = torch.zeros((2, 3), dtype=torch.bfloat16)
    saved = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    artifact = tmp_path / "latents.pt"
    torch.save({"text_hidden_states": saved}, artifact)
    monkeypatch.setenv("MINIMAX_H3_DEBUG_TEXT_HIDDEN_STATES_PATH", str(artifact))
    batch = _batch_with_hidden_states(native)

    applied = _apply_debug_text_hidden_states_override(batch)
    actual = batch.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]["positive"][
        "hidden_states"
    ]

    assert applied == artifact.resolve()
    assert torch.equal(actual, saved)
    assert actual.dtype == native.dtype
    assert actual.data_ptr() != saved.data_ptr()


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({}, "must contain tensor key"),
        (
            {"text_hidden_states": torch.zeros((3, 2), dtype=torch.bfloat16)},
            "shape mismatch",
        ),
        (
            {"text_hidden_states": torch.zeros((2, 3), dtype=torch.float32)},
            "dtype mismatch",
        ),
        (
            {
                "text_hidden_states": torch.tensor(
                    [[float("nan"), 0, 0], [0, 0, 0]], dtype=torch.bfloat16
                )
            },
            "contains NaN or Inf",
        ),
    ],
)
def test_debug_text_hidden_states_override_rejects_invalid_artifact(
    monkeypatch, tmp_path, payload, error
):
    artifact = tmp_path / "invalid.pt"
    torch.save(payload, artifact)
    monkeypatch.setenv("MINIMAX_H3_DEBUG_TEXT_HIDDEN_STATES_PATH", str(artifact))
    batch = _batch_with_hidden_states(torch.zeros((2, 3), dtype=torch.bfloat16))

    with pytest.raises(ValueError, match=error):
        _apply_debug_text_hidden_states_override(batch)
