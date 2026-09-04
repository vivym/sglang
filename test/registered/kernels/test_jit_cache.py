from pathlib import Path

import tvm_ffi

from sglang.kernels.jit.utils.compile import _load_cached_module


def test_load_cached_module_reuses_valid_output(tmp_path: Path, monkeypatch) -> None:
    prebuilt = tmp_path / "module.so"
    prebuilt.write_bytes(b"valid")
    expected = object()

    monkeypatch.setattr(tvm_ffi, "load_module", lambda path: expected)

    assert _load_cached_module("module", prebuilt) is expected
    assert prebuilt.read_bytes() == b"valid"


def test_load_cached_module_removes_unreadable_output(
    tmp_path: Path, monkeypatch
) -> None:
    prebuilt = tmp_path / "module.so"
    prebuilt.write_bytes(b"corrupt")

    def fail_to_load(path: str):
        raise RuntimeError(f"cannot load {path}")

    monkeypatch.setattr(tvm_ffi, "load_module", fail_to_load)

    assert _load_cached_module("module", prebuilt) is None
    assert not prebuilt.exists()
