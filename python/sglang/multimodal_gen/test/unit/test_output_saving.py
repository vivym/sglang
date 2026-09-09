from contextlib import contextmanager

import numpy as np
import pytest
import torch
from PIL import Image

import sglang.multimodal_gen.runtime.entrypoints.utils as output_utils
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    MaterializedOutput,
    post_process_sample,
    save_materialized_output,
)


class _FakeCudaTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data, False)

    @property
    def device(self):
        return torch.device("cuda:0")


def _rgb_frame() -> np.ndarray:
    return np.array(
        [
            [[0, 32, 255], [64, 128, 192], [255, 224, 16]],
            [[9, 17, 33], [127, 128, 129], [240, 12, 88]],
        ],
        dtype=np.uint8,
    )


@pytest.mark.parametrize("output_compression", [None, 0, 75])
def test_png_output_saving_preserves_pixels(tmp_path, output_compression):
    frame = _rgb_frame()
    output_path = tmp_path / f"sample_{output_compression}.png"

    frames = post_process_sample(
        frame,
        DataType.IMAGE,
        fps=1,
        save_file_path=str(output_path),
        output_compression=output_compression,
    )

    assert output_path.exists()
    np.testing.assert_array_equal(frames[0], frame)
    np.testing.assert_array_equal(np.array(Image.open(output_path)), frame)


@pytest.mark.parametrize(
    ("output_compression", "expected_compress_level"), [(None, 1), (0, 0), (75, 1)]
)
def test_png_output_saving_uses_fast_pillow_path(
    tmp_path, monkeypatch, output_compression, expected_compress_level
):
    frame = _rgb_frame()
    output_path = tmp_path / f"sample_{output_compression}.png"

    def fail_imageio_imwrite(*args, **kwargs):
        raise AssertionError("PNG output should use Pillow's PNG fast path")

    original_save = Image.Image.save
    save_calls = []

    def save_spy(self, fp, format=None, **params):
        save_calls.append((format, params.get("compress_level")))
        return original_save(self, fp, format=format, **params)

    monkeypatch.setattr(output_utils.imageio, "imwrite", fail_imageio_imwrite)
    monkeypatch.setattr(Image.Image, "save", save_spy)

    post_process_sample(
        frame,
        DataType.IMAGE,
        fps=1,
        save_file_path=str(output_path),
        output_compression=output_compression,
    )

    assert save_calls == [("PNG", expected_compress_level)]


def test_video_with_audio_uses_single_pass_encoder(tmp_path, monkeypatch):
    output_path = tmp_path / "sample.mp4"
    calls = []

    class FakeWavFile:
        @staticmethod
        def write(*_args, **_kwargs):
            pass

    def mimsave_spy(path, frames, **kwargs):
        calls.append((path, frames, kwargs))
        assert kwargs["audio_path"].endswith(".wav")
        assert kwargs["audio_codec"] == "aac"

    def fail_legacy_mux(**_kwargs):
        raise AssertionError("the two-pass mux path should not run")

    monkeypatch.setattr(output_utils.imageio, "mimsave", mimsave_spy)
    monkeypatch.setattr(output_utils, "scipy_wavfile", FakeWavFile)
    monkeypatch.setattr(output_utils, "_maybe_mux_audio_into_mp4", fail_legacy_mux)

    materialized = MaterializedOutput(
        sample=None,
        frames=[_rgb_frame()],
        audio=np.zeros((320, 2), dtype=np.float32),
        fps=24,
    )
    save_materialized_output(
        materialized,
        DataType.VIDEO,
        str(output_path),
        audio_sample_rate=32000,
    )

    assert len(calls) == 1


def test_video_audio_single_pass_failure_falls_back(tmp_path, monkeypatch):
    output_path = tmp_path / "sample.mp4"
    calls = []
    mux_calls = []

    class FakeWavFile:
        @staticmethod
        def write(*_args, **_kwargs):
            pass

    def mimsave_spy(path, frames, **kwargs):
        calls.append((path, frames, kwargs))
        if "audio_path" in kwargs:
            raise RuntimeError("unsupported audio input")

    monkeypatch.setattr(output_utils.imageio, "mimsave", mimsave_spy)
    monkeypatch.setattr(output_utils, "scipy_wavfile", FakeWavFile)
    monkeypatch.setattr(
        output_utils,
        "_maybe_mux_audio_into_mp4",
        lambda **kwargs: mux_calls.append(kwargs),
    )

    materialized = MaterializedOutput(
        sample=None,
        frames=[_rgb_frame()],
        audio=np.zeros((320, 2), dtype=np.float32),
        fps=24,
    )
    save_materialized_output(
        materialized,
        DataType.VIDEO,
        str(output_path),
        audio_sample_rate=32000,
    )

    assert len(calls) == 2
    assert "audio_path" in calls[0][2]
    assert "audio_path" not in calls[1][2]
    assert len(mux_calls) == 1


@pytest.mark.parametrize(
    ("height", "available_cpus", "expected_threads"),
    [
        (768, 256, 24),
        (720, 256, 22),
        (2160, 16, 24),
        (4320, 256, 128),
        (16, 1, 1),
    ],
)
def test_x264_auto_thread_count(monkeypatch, height, available_cpus, expected_threads):
    monkeypatch.setattr(
        output_utils.os,
        "sched_getaffinity",
        lambda _pid: set(range(available_cpus)),
    )

    assert output_utils._x264_auto_thread_count(height) == expected_threads


def test_x264_auto_thread_count_can_use_target_affinity():
    assert output_utils._x264_auto_thread_count(768, cpu_count=96) == 24
    assert output_utils._x264_auto_thread_count(768, cpu_count=1) == 1


def test_video_encoder_cpu_count_prefers_target_affinity(monkeypatch):
    monkeypatch.setattr(output_utils.os, "sched_getaffinity", lambda _pid: {0})

    assert output_utils._video_encoder_cpu_count(frozenset(range(96))) == 96
    assert output_utils._video_encoder_cpu_count(None) == 1


def test_parse_linux_cpu_list():
    assert output_utils._parse_linux_cpu_list("0-3,8,10-11") == {
        0,
        1,
        2,
        3,
        8,
        10,
        11,
    }


def test_format_linux_cpu_list():
    assert output_utils._format_linux_cpu_list([11, 3, 2, 1, 8, 10]) == "1-3,8,10-11"
    assert output_utils._format_linux_cpu_list([]) == ""


@pytest.mark.parametrize("value", ["", "3-1", "-1", "1-"])
def test_parse_linux_cpu_list_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        output_utils._parse_linux_cpu_list(value)


def test_cuda_device_numa_cpu_affinity_intersects_process_cpuset(tmp_path, monkeypatch):
    pci_root = tmp_path / "bus/pci/devices/0000:23:00.0"
    node_root = tmp_path / "devices/system/node/node1"
    pci_root.mkdir(parents=True)
    node_root.mkdir(parents=True)
    (pci_root / "numa_node").write_text("1\n", encoding="ascii")
    (node_root / "cpulist").write_text("4-7,12-15\n", encoding="ascii")

    properties = type(
        "Properties",
        (),
        {"pci_domain_id": 0, "pci_bus_id": 0x23, "pci_device_id": 0},
    )()
    monkeypatch.setattr(output_utils, "_SYSFS_ROOT", tmp_path)
    monkeypatch.setattr(
        output_utils.torch.cuda,
        "get_device_properties",
        lambda _device: properties,
    )
    monkeypatch.setattr(
        output_utils.os,
        "sched_getaffinity",
        lambda _pid: {0, 1, 4, 5, 12},
    )
    monkeypatch.setattr(
        output_utils,
        "_INHERITED_PROCESS_CPU_AFFINITY",
        frozenset({0, 1, 4, 5, 12}),
    )

    receipt = {}
    assert output_utils._cuda_device_numa_cpu_affinity(0, receipt=receipt) == frozenset(
        {4, 5, 12}
    )
    assert receipt == {
        "cuda_device": "0",
        "pci_address": "0000:23:00.0",
        "numa_node": 1,
        "calling_thread_cpu_affinity": {"count": 5, "cpulist": "0-1,4-5,12"},
        "inherited_cpu_affinity": {"count": 5, "cpulist": "0-1,4-5,12"},
        "numa_local_cpu_affinity": {"count": 8, "cpulist": "4-7,12-15"},
        "target_cpu_affinity": {"count": 3, "cpulist": "4-5,12"},
        "resolver_status": "target_resolved",
    }


def test_cuda_device_numa_cpu_affinity_keeps_inherited_single_node_cpuset(
    tmp_path, monkeypatch
):
    pci_root = tmp_path / "bus/pci/devices/0000:23:00.0"
    node_root = tmp_path / "devices/system/node/node0"
    pci_root.mkdir(parents=True)
    node_root.mkdir(parents=True)
    (pci_root / "numa_node").write_text("0\n", encoding="ascii")
    (node_root / "cpulist").write_text("0-3\n", encoding="ascii")

    properties = type(
        "Properties",
        (),
        {"pci_domain_id": 0, "pci_bus_id": 0x23, "pci_device_id": 0},
    )()
    monkeypatch.setattr(output_utils, "_SYSFS_ROOT", tmp_path)
    monkeypatch.setattr(
        output_utils.torch.cuda,
        "get_device_properties",
        lambda _device: properties,
    )
    monkeypatch.setattr(
        output_utils.os,
        "sched_getaffinity",
        lambda _pid: {0, 1, 2, 3},
    )
    monkeypatch.setattr(
        output_utils,
        "_INHERITED_PROCESS_CPU_AFFINITY",
        frozenset({0, 1, 2, 3}),
    )

    receipt = {}
    assert output_utils._cuda_device_numa_cpu_affinity(0, receipt=receipt) is None
    assert receipt["resolver_status"] == "already_local"
    assert receipt["target_cpu_affinity"] == {"count": 4, "cpulist": "0-3"}


def test_temporary_cpu_affinity_restores_calling_thread(monkeypatch):
    state = {"affinity": frozenset({0, 1, 2, 3})}
    transitions = []

    monkeypatch.setattr(
        output_utils.os,
        "sched_getaffinity",
        lambda _pid: state["affinity"],
    )
    monkeypatch.setattr(
        output_utils,
        "_INHERITED_PROCESS_CPU_AFFINITY",
        frozenset({0, 1, 2, 3, 8}),
    )

    def set_affinity(_pid, cpus):
        state["affinity"] = frozenset(cpus)
        transitions.append(state["affinity"])

    monkeypatch.setattr(output_utils.os, "sched_setaffinity", set_affinity)

    receipt = {}
    with output_utils._temporary_cpu_affinity(frozenset({1, 2, 8}), receipt=receipt):
        assert state["affinity"] == frozenset({1, 2, 8})

    assert state["affinity"] == frozenset({0, 1, 2, 3})
    assert transitions == [frozenset({1, 2, 8}), frozenset({0, 1, 2, 3})]
    assert receipt["bind_status"] == "bound"
    assert receipt["bound_cpu_affinity"] == {"count": 3, "cpulist": "1-2,8"}
    assert receipt["restore_status"] == "restored"
    assert receipt["restored_cpu_affinity"] == {"count": 4, "cpulist": "0-3"}


def test_temporary_cpu_affinity_falls_back_when_binding_fails(monkeypatch):
    monkeypatch.setattr(
        output_utils.os,
        "sched_getaffinity",
        lambda _pid: frozenset({0, 1, 2, 3}),
    )
    monkeypatch.setattr(
        output_utils,
        "_INHERITED_PROCESS_CPU_AFFINITY",
        frozenset({0, 1, 2, 3}),
    )

    def fail_set_affinity(_pid, _cpus):
        raise OSError("not permitted")

    monkeypatch.setattr(output_utils.os, "sched_setaffinity", fail_set_affinity)

    entered = False
    with output_utils._temporary_cpu_affinity(frozenset({1, 2})):
        entered = True

    assert entered


def test_video_save_uses_materialized_encoder_affinity(tmp_path, monkeypatch):
    transitions = []

    @contextmanager
    def affinity_spy(cpus):
        transitions.append(("enter", cpus))
        yield
        transitions.append(("exit", cpus))

    monkeypatch.setattr(output_utils, "_temporary_cpu_affinity", affinity_spy)
    monkeypatch.setattr(output_utils, "_try_save_video_with_audio", lambda **_: True)
    materialized = MaterializedOutput(
        sample=None,
        frames=[_rgb_frame()],
        audio=np.zeros((320, 2), dtype=np.float32),
        fps=24,
        video_encoder_cpu_affinity=frozenset({4, 5}),
    )

    save_materialized_output(
        materialized,
        DataType.VIDEO,
        str(tmp_path / "sample.mp4"),
        audio_sample_rate=32000,
    )

    assert transitions == [
        ("enter", frozenset({4, 5})),
        ("exit", frozenset({4, 5})),
    ]


def test_video_direct_save_short_circuits_materialization(tmp_path, monkeypatch):
    output_path = tmp_path / "sample.mp4"
    direct_calls = []
    timings = {}

    def direct_save(**kwargs):
        direct_calls.append(kwargs)
        kwargs["stage_recorder"]("OutputSave.direct.pipe_write", 1.25)
        return True

    monkeypatch.setattr(
        output_utils,
        "_try_save_cuda_video_direct",
        direct_save,
    )
    monkeypatch.setattr(
        output_utils,
        "post_process_sample",
        lambda *_args, **_kwargs: pytest.fail(
            "successful direct save should skip frame materialization"
        ),
    )

    paths = output_utils.save_outputs(
        [torch.zeros((3, 1, 2, 3))],
        DataType.VIDEO,
        fps=24,
        save_output=True,
        build_output_path=lambda _idx: str(output_path),
        stage_recorder=lambda name, duration: timings.__setitem__(name, duration),
    )

    assert paths == [str(output_path)]
    assert len(direct_calls) == 1
    assert timings == {"OutputSave.direct.pipe_write": 1.25}


def test_cuda_video_direct_save_uses_target_affinity_for_x264_threads(
    tmp_path, monkeypatch
):
    video = _FakeCudaTensor(torch.zeros((3, 1, 768, 16)))
    target_affinity = frozenset(range(96))
    commands = []
    bound_affinities = []

    class FakeStream:
        @staticmethod
        def synchronize():
            pass

    class FakeStdin:
        @staticmethod
        def fileno():
            return 1

        @staticmethod
        def close():
            pass

    class FakeProcess:
        pid = 4321

        def __init__(self):
            self.stdin = FakeStdin()

        @staticmethod
        def wait():
            return 0

        @staticmethod
        def poll():
            return 0

        @staticmethod
        def kill():
            pass

    class FakeBuffer:
        fd = 1
        tensor = torch.empty((1, 768, 16, 3), dtype=torch.uint8)

    @contextmanager
    def fake_buffer(_shape):
        yield FakeBuffer()

    @contextmanager
    def affinity_spy(cpu_affinity, **_kwargs):
        bound_affinities.append(cpu_affinity)
        yield

    def popen_spy(command, **_kwargs):
        commands.append(command)
        return FakeProcess()

    def resolve_affinity(_device, *, receipt=None):
        assert receipt is not None
        receipt["resolver_status"] = "target_resolved"
        return target_affinity

    monkeypatch.setattr(
        output_utils, "_cuda_device_numa_cpu_affinity", resolve_affinity
    )
    monkeypatch.setattr(output_utils, "_resolve_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(output_utils, "_acquire_cuda_video_buffer", fake_buffer)
    monkeypatch.setattr(output_utils, "_temporary_cpu_affinity", affinity_spy)
    monkeypatch.setattr(output_utils, "_sendfile_all", lambda *_args: None)
    monkeypatch.setattr(
        output_utils.torch.cuda, "current_stream", lambda _device: FakeStream()
    )
    monkeypatch.setattr(
        output_utils.os, "sched_getaffinity", lambda pid: target_affinity
    )
    monkeypatch.setattr(output_utils.subprocess, "Popen", popen_spy)
    metadata = {}

    assert output_utils._try_save_cuda_video_direct(
        save_file_path=str(tmp_path / "sample.mp4"),
        sample=video,
        fps=24,
        audio_sample_rate=None,
        output_compression=None,
        metadata_recorder=lambda name, value: metadata.__setitem__(name, value),
    )
    assert len(commands) == 1
    threads_index = commands[0].index("-threads")
    assert commands[0][threads_index + 1] == "24"
    assert bound_affinities == [target_affinity]
    assert metadata["video_encoder"]["schema"] == "sglang.video-encoder/v1"
    assert metadata["video_encoder"]["status"] == "success"
    assert metadata["video_encoder"]["ffmpeg"]["threads"] == 24
    assert metadata["video_encoder"]["affinity"]["resolver_status"] == (
        "target_resolved"
    )
    assert metadata["video_encoder"]["affinity"]["child_affinity_status"] == (
        "captured"
    )
    assert metadata["video_encoder"]["affinity"]["child_pid"] == 4321
    assert metadata["video_encoder"]["affinity"]["ffmpeg_child_cpu_affinity"] == {
        "count": 96,
        "cpulist": "0-95",
    }


def test_parallel_cuda_video_admission_uses_target_affinity(tmp_path, monkeypatch):
    videos = [
        _FakeCudaTensor(torch.zeros((3, 1, 768, 16))),
        _FakeCudaTensor(torch.ones((3, 1, 768, 16))),
    ]
    target_affinity = frozenset(range(96))
    calls = []

    monkeypatch.setattr(output_utils.os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(
        output_utils, "_cuda_device_numa_cpu_affinity", lambda _device: target_affinity
    )
    monkeypatch.setattr(
        output_utils.torch.cuda,
        "mem_get_info",
        lambda _device: (1 << 40, 1 << 40),
    )
    monkeypatch.setattr(
        output_utils,
        "_try_save_cuda_video_direct",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    result = output_utils._try_save_cuda_videos_direct(
        videos,
        [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")],
        fps=24,
        audio_sample_rate=None,
        output_compression=None,
    )

    assert result == [True, True]
    assert len(calls) == 2


def test_multiple_videos_use_parallel_direct_save_with_serial_fallback(
    tmp_path, monkeypatch
):
    outputs = [torch.zeros((3, 1, 2, 3)), torch.ones((3, 1, 2, 3))]
    direct_calls = []

    def parallel_save(samples, paths, **kwargs):
        direct_calls.append((samples, paths, kwargs))
        return [True, False]

    serial_calls = []

    monkeypatch.setattr(output_utils, "_try_save_cuda_videos_direct", parallel_save)
    monkeypatch.setattr(
        output_utils,
        "_try_save_cuda_video_direct",
        lambda **kwargs: serial_calls.append(kwargs) or True,
    )
    monkeypatch.setattr(
        output_utils,
        "post_process_sample",
        lambda *_args, **_kwargs: pytest.fail(
            "successful parallel direct saves should skip frame materialization"
        ),
    )

    paths = output_utils.save_outputs(
        outputs,
        DataType.VIDEO,
        fps=24,
        save_output=True,
        build_output_path=lambda idx: str(tmp_path / f"sample_{idx}.mp4"),
    )

    assert paths == [str(tmp_path / "sample_0.mp4"), str(tmp_path / "sample_1.mp4")]
    assert len(direct_calls) == 1
    samples, save_paths, kwargs = direct_calls[0]
    assert all(actual is expected for actual, expected in zip(samples, outputs))
    assert save_paths == paths
    assert kwargs["fps"] == 24
    assert len(serial_calls) == 1
    assert serial_calls[0]["save_file_path"] == paths[1]
