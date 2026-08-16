import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints import utils as output_utils
from sglang.multimodal_gen.runtime.managers import gpu_worker as gpu_worker_module
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


class _Request(SimpleNamespace):
    pass


def _worker(endpoint: str = "tcp://127.0.0.1:30000") -> GPUWorker:
    worker = GPUWorker.__new__(GPUWorker)
    worker.is_output_rank = True
    worker.server_args = SimpleNamespace(scheduler_endpoint=endpoint)
    worker._output_persistence_executor = ThreadPoolExecutor(max_workers=1)
    worker._output_persistence_slots = threading.BoundedSemaphore(2)
    return worker


def _request(output_path: Path):
    return _Request(
        data_type=DataType.VIDEO,
        fps=24,
        extra={},
        enable_frame_interpolation=False,
        frame_interpolation_exp=1,
        frame_interpolation_scale=1.0,
        frame_interpolation_model_path=None,
        enable_upscaling=False,
        upscaling_model_path=None,
        upscaling_scale=4,
        output_compression=None,
        output_file_path=lambda _count, _index: str(output_path),
    )


def _output_batch() -> OutputBatch:
    return OutputBatch(
        output=[torch.zeros((3, 2, 4, 5), dtype=torch.float32)],
        audio=torch.zeros((1, 2, 16), dtype=torch.float32),
        audio_sample_rate=32000,
    )


def test_async_output_persistence_stages_cpu_only_state(tmp_path, monkeypatch):
    worker = _worker()
    output_path = tmp_path / "output.mp4"
    captured = {}

    def save_spy(materialized, _data_type, staging_path, **kwargs):
        captured["materialized"] = materialized
        captured["audio_sample_rate"] = kwargs["audio_sample_rate"]
        Path(staging_path).write_bytes(b"encoded")

    monkeypatch.setattr(gpu_worker_module, "save_materialized_output", save_spy)
    output_batch = _output_batch()
    try:
        worker._save_output_paths(_request(output_path), output_batch)
        ref = output_batch.output_file_paths[0]

        assert ref.materialize() == str(output_path)
        materialized = captured["materialized"]
        assert materialized.sample is None
        assert all(isinstance(frame, np.ndarray) for frame in materialized.frames)
        assert isinstance(materialized.audio, torch.Tensor)
        assert materialized.audio.device.type == "cpu"
        assert captured["audio_sample_rate"] == 32000
        assert output_path.read_bytes() == b"encoded"
    finally:
        worker.shutdown()


def test_async_output_persistence_matches_direct_output_conversion(
    tmp_path, monkeypatch
):
    video = torch.tensor(
        [
            [[[-0.1, 0.0, 0.5], [1.0, 1.1, 127.9 / 255]]],
            [[[0.25, 0.75, 1.0 / 255], [254.9 / 255, 0.999, 2.0]]],
            [[[1.0, 0.0, 0.1], [0.9, 128.1 / 255, -1.0]]],
        ],
        dtype=torch.float32,
    )
    audio = torch.tensor(
        [[[-1.5, -0.25, 0.25, 1.5], [0.75, -0.75, 2.0, -2.0]]],
        dtype=torch.float16,
    )
    direct_sample = {}

    def direct_save_spy(**kwargs):
        direct_sample["value"] = kwargs["sample"]
        return True

    monkeypatch.setattr(
        output_utils,
        "_try_save_cuda_video_direct",
        direct_save_spy,
    )
    output_utils.save_outputs(
        [video],
        DataType.VIDEO,
        fps=24,
        save_output=True,
        build_output_path=lambda _index: str(tmp_path / "direct.mp4"),
        audio=audio,
        audio_sample_rate=32000,
    )

    worker = _worker()
    output_path = tmp_path / "async.mp4"
    captured = {}

    def save_spy(materialized, _data_type, staging_path, **_kwargs):
        captured["materialized"] = materialized
        Path(staging_path).write_bytes(b"encoded")

    monkeypatch.setattr(gpu_worker_module, "save_materialized_output", save_spy)
    output_batch = OutputBatch(
        output=[video],
        audio=audio,
        audio_sample_rate=32000,
    )
    try:
        worker._save_output_paths(_request(output_path), output_batch)
        assert output_batch.output_file_paths[0].materialize() == str(output_path)

        direct_video, direct_audio = direct_sample["value"]
        expected_frames = (
            (direct_video * 255)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .contiguous()
            .numpy()
        )
        materialized = captured["materialized"]
        np.testing.assert_array_equal(np.stack(materialized.frames), expected_frames)
        torch.testing.assert_close(
            materialized.audio,
            direct_audio.detach().float().clamp(-1.0, 1.0),
            rtol=0,
            atol=0,
        )
        assert materialized.audio.device.type == "cpu"
        assert materialized.audio.dtype == torch.float32
        assert materialized.fps == 24
    finally:
        worker.shutdown()


def test_async_output_persistence_propagates_encoder_failure(tmp_path, monkeypatch):
    worker = _worker()
    output_path = tmp_path / "failed.mp4"

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(gpu_worker_module, "save_materialized_output", fail_save)
    output_batch = _output_batch()
    try:
        worker._save_output_paths(_request(output_path), output_batch)
        ref = output_batch.output_file_paths[0]

        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            ref.materialize()
        assert not output_path.exists()
    finally:
        worker.shutdown()


def test_async_output_persistence_requires_local_scheduler(tmp_path, monkeypatch):
    worker = _worker(endpoint="tcp://10.0.0.2:30000")
    output_path = tmp_path / "sync.mp4"
    monkeypatch.setattr(
        gpu_worker_module,
        "save_outputs",
        lambda *_args, **_kwargs: [str(output_path)],
    )
    output_batch = _output_batch()
    try:
        worker._save_output_paths(_request(output_path), output_batch)
        assert output_batch.output_file_paths == [str(output_path)]
    finally:
        worker.shutdown()


def test_async_output_persistence_preserves_dynamic_output_paths(tmp_path, monkeypatch):
    worker = _worker()
    output_path = tmp_path / "dynamic.mp4"
    request = _request(tmp_path / "ignored.mp4")
    request.extra["dynamic_batch_output_paths"] = [str(output_path)]
    captured = {}

    def save_spy(
        _outputs, _data_type, _fps, _save_output, build_output_path, **_kwargs
    ):
        captured["path"] = build_output_path(0)
        return [captured["path"]]

    monkeypatch.setattr(gpu_worker_module, "save_outputs", save_spy)
    output_batch = _output_batch()
    try:
        worker._save_output_paths(request, output_batch)

        assert captured["path"] == str(output_path)
        assert output_batch.output_file_paths == [str(output_path)]
    finally:
        worker.shutdown()


def test_async_output_persistence_does_not_retain_request(tmp_path, monkeypatch):
    worker = _worker()
    output_path = tmp_path / "output.mp4"
    save_started = threading.Event()
    allow_save = threading.Event()

    def save_spy(_materialized, _data_type, staging_path, **_kwargs):
        save_started.set()
        assert allow_save.wait(timeout=1.0)
        Path(staging_path).write_bytes(b"encoded")

    monkeypatch.setattr(gpu_worker_module, "save_materialized_output", save_spy)
    output_batch = _output_batch()
    request = _request(output_path)
    request_ref = weakref.ref(request)
    try:
        worker._save_output_paths(request, output_batch)
        assert save_started.wait(timeout=1.0)
        del request
        gc.collect()

        assert request_ref() is None
        allow_save.set()
        assert output_batch.output_file_paths[0].materialize() == str(output_path)
    finally:
        allow_save.set()
        worker.shutdown()


def test_async_output_persistence_shutdown_drains_pending_save(tmp_path, monkeypatch):
    worker = _worker()
    output_path = tmp_path / "output.mp4"
    save_started = threading.Event()
    allow_save = threading.Event()

    def save_spy(_materialized, _data_type, staging_path, **_kwargs):
        save_started.set()
        assert allow_save.wait(timeout=1.0)
        Path(staging_path).write_bytes(b"encoded")

    monkeypatch.setattr(gpu_worker_module, "save_materialized_output", save_spy)
    output_batch = _output_batch()
    worker._save_output_paths(_request(output_path), output_batch)
    assert save_started.wait(timeout=1.0)

    shutdown_thread = threading.Thread(target=worker.shutdown)
    shutdown_thread.start()
    try:
        shutdown_thread.join(timeout=0.02)
        assert shutdown_thread.is_alive()
        allow_save.set()
        shutdown_thread.join(timeout=1.0)
        assert not shutdown_thread.is_alive()
        assert output_batch.output_file_paths[0].materialize() == str(output_path)
    finally:
        allow_save.set()
        shutdown_thread.join(timeout=1.0)
        worker.shutdown()
