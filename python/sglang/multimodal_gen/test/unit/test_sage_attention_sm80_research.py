from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import sage_attn


def test_sm80_research_mode_is_default_off_and_rejects_stale_identity() -> None:
    assert sage_attn.resolve_sage_sm80_research_config({}) is None

    with pytest.raises(RuntimeError, match="is required when setting"):
        sage_attn.resolve_sage_sm80_research_config(
            {sage_attn.SAGE_SM80_ROOT_ENV: "/persistent/sageattention"}
        )


def test_sm80_research_mode_rejects_unknown_variant_before_import() -> None:
    with pytest.raises(RuntimeError, match="unsupported"):
        sage_attn.resolve_sage_sm80_research_config(
            {sage_attn.SAGE_SM80_VARIANT_ENV: "candidate"}
        )


def test_sm80_research_mode_requires_persistent_root() -> None:
    with pytest.raises(RuntimeError, match="persistent storage"):
        sage_attn.resolve_sage_sm80_research_config(
            {
                sage_attn.SAGE_SM80_VARIANT_ENV: "cuda_fp32",
                sage_attn.SAGE_SM80_ROOT_ENV: "/tmp/sageattention",
                sage_attn.SAGE_SM80_QATTN_SHA256_ENV: "0" * 64,
            }
        )


def test_sm80_research_mode_binds_root_variant_and_extension_hash(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "persistent-sageattention"
    package_dir = root / "sageattention"
    package_dir.mkdir(parents=True)
    package_path = package_dir / "__init__.py"
    binding_path = package_dir / "sm80.py"
    extension_path = package_dir / "_qattn_sm80.so"
    for path in (package_path, binding_path, extension_path):
        path.write_bytes(b"unit")

    expected_sha256 = "a" * 64
    from sageattention import sm80 as sm80_module

    monkeypatch.setattr(sage_attn, "_persistent_root", lambda raw: root)
    monkeypatch.setattr(sage_attn.sageattention, "__file__", str(package_path))
    monkeypatch.setattr(sm80_module, "__file__", str(binding_path))
    monkeypatch.setattr(
        sage_attn.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(extension_path)),
    )
    monkeypatch.setattr(sage_attn, "_sha256", lambda path: expected_sha256)

    config = sage_attn.resolve_sage_sm80_research_config(
        {
            sage_attn.SAGE_SM80_VARIANT_ENV: (
                "cuda_per_warp_fp32_precombined_skip_noop"
            ),
            sage_attn.SAGE_SM80_ROOT_ENV: str(root),
            sage_attn.SAGE_SM80_QATTN_SHA256_ENV: expected_sha256,
        }
    )

    assert config is not None
    assert config.variant == "cuda_per_warp_fp32_precombined_skip_noop"
    assert config.root == root
    assert config.qattn_extension == extension_path
    assert config.qattn_sha256 == expected_sha256


def test_sm80_research_mode_rejects_extension_hash_mismatch(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "persistent-sageattention"
    package_dir = root / "sageattention"
    package_dir.mkdir(parents=True)
    package_path = package_dir / "__init__.py"
    binding_path = package_dir / "sm80.py"
    extension_path = package_dir / "_qattn_sm80.so"
    for path in (package_path, binding_path, extension_path):
        path.write_bytes(b"unit")

    from sageattention import sm80 as sm80_module

    monkeypatch.setattr(sage_attn, "_persistent_root", lambda raw: root)
    monkeypatch.setattr(sage_attn.sageattention, "__file__", str(package_path))
    monkeypatch.setattr(sm80_module, "__file__", str(binding_path))
    monkeypatch.setattr(
        sage_attn.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(extension_path)),
    )
    monkeypatch.setattr(sage_attn, "_sha256", lambda path: "b" * 64)

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        sage_attn.resolve_sage_sm80_research_config(
            {
                sage_attn.SAGE_SM80_VARIANT_ENV: "cuda_fp32",
                sage_attn.SAGE_SM80_ROOT_ENV: str(root),
                sage_attn.SAGE_SM80_QATTN_SHA256_ENV: "a" * 64,
            }
        )


def test_sm80_research_forward_cannot_fall_back_to_generic_sage(monkeypatch) -> None:
    root = sage_attn.Path("/persistent/sageattention")
    config = sage_attn.SageSM80ResearchConfig(
        variant="cuda_per_warp_fp32_precombined_skip_noop",
        root=root,
        package_path=root / "sageattention/__init__.py",
        binding_path=root / "sageattention/sm80.py",
        qattn_extension=root / "sageattention/_qattn_sm80.so",
        qattn_sha256="a" * 64,
    )
    monkeypatch.setattr(
        sage_attn, "current_sage_sm80_research_config", lambda: config
    )
    monkeypatch.setattr(
        sage_attn,
        "sageattn",
        Mock(side_effect=AssertionError("generic SageAttention fallback executed")),
    )

    from sageattention import sm80 as sm80_module

    expected = torch.ones((1, 4, 2, 128), dtype=torch.bfloat16)
    research_call = Mock(
        return_value=(
            expected,
            {
                "requested_variant": config.variant,
                "fallback_allowed": False,
            },
        )
    )
    monkeypatch.setattr(sm80_module, "sm80_attention", research_call)

    impl = sage_attn.SageAttentionImpl(
        num_heads=2,
        head_size=128,
        causal=False,
        softmax_scale=128**-0.5,
    )
    query = torch.zeros_like(expected)
    output = impl.forward(query, query, query, None)

    assert output is expected
    research_call.assert_called_once_with(
        query,
        query,
        query,
        variant=config.variant,
        sm_scale=128**-0.5,
        return_trace=True,
        check_finite=False,
    )
