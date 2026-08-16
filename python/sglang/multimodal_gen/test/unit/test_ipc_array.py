# SPDX-License-Identifier: Apache-2.0

import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from sglang.multimodal_gen.runtime import ipc_array
from sglang.multimodal_gen.runtime.ipc_array import (
    NumpyArrayFileRef,
    PendingOutputFileRef,
    is_local_endpoint,
    materialize_file_refs,
    spill_large_arrays_to_file_refs,
)


def test_spill_large_arrays_round_trips_and_removes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    array = np.arange(ipc_array._MIN_FILE_REF_BYTES, dtype=np.uint8)

    spilled = spill_large_arrays_to_file_refs([array])

    assert isinstance(spilled[0], NumpyArrayFileRef)
    spilled_path = Path(spilled[0].path)
    assert spilled_path.exists()

    materialized = materialize_file_refs(spilled)

    assert np.array_equal(materialized[0], array)
    assert not spilled_path.exists()


def test_small_arrays_are_kept_inline():
    array = np.arange(16, dtype=np.uint8)

    spilled = spill_large_arrays_to_file_refs((array,))

    assert spilled[0] is array


def test_large_arrays_are_kept_inline_without_shm(monkeypatch):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: None)
    array = np.arange(ipc_array._MIN_FILE_REF_BYTES, dtype=np.uint8)

    spilled = spill_large_arrays_to_file_refs(array)

    assert spilled is array


def test_spill_removes_temp_file_when_save_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    array = np.arange(ipc_array._MIN_FILE_REF_BYTES, dtype=np.uint8)
    created_paths = []

    def fail_save(*args, **kwargs):
        raise OSError("simulated write failure")

    original_mkstemp = tempfile.mkstemp

    def tracked_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        created_paths.append(Path(path))
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(np, "save", fail_save)

    with pytest.raises(OSError, match="simulated write failure"):
        spill_large_arrays_to_file_refs(array)

    assert created_paths
    assert not created_paths[0].exists()


def test_local_endpoint_detection():
    assert is_local_endpoint("tcp://127.0.0.1:30000")
    assert is_local_endpoint("tcp://localhost:30000")
    assert is_local_endpoint("ipc:///tmp/sgl.sock")
    assert is_local_endpoint("inproc://scheduler")
    assert not is_local_endpoint("tcp://10.0.0.2:30000")


def test_pending_output_file_ref_publishes_atomically(tmp_path):
    final_path = tmp_path / "output.mp4"
    ref = PendingOutputFileRef.create(str(final_path), timeout_seconds=1.0)
    Path(ref.staging_path).write_bytes(b"complete-media")

    publisher = threading.Thread(
        target=lambda: (time.sleep(0.02), ref.publish_success())
    )
    publisher.start()
    try:
        assert materialize_file_refs([ref]) == [str(final_path)]
    finally:
        publisher.join()

    assert final_path.read_bytes() == b"complete-media"
    assert not Path(ref.staging_path).exists()
    assert not Path(ref.ready_path).exists()


def test_pending_output_file_ref_propagates_background_error(tmp_path):
    ref = PendingOutputFileRef.create(str(tmp_path / "failed.mp4"), timeout_seconds=1.0)
    Path(ref.staging_path).write_bytes(b"partial")
    ref.publish_error(RuntimeError("encoder failed"))

    with pytest.raises(RuntimeError, match="encoder failed"):
        ref.materialize()

    assert not Path(ref.staging_path).exists()
    assert not Path(ref.error_path).exists()


def test_pending_output_file_ref_times_out(tmp_path):
    ref = PendingOutputFileRef.create(
        str(tmp_path / "timeout.mp4"), timeout_seconds=0.01
    )

    with pytest.raises(TimeoutError, match="Timed out"):
        ref.materialize()
