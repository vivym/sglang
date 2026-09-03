# SPDX-License-Identifier: Apache-2.0
"""Numerical boundaries for the one-pass Ref2VA media path."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    material_io,
    reference_encoding,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.decoding import (
    MiniMaxH3DecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
    decoding,
)


def test_video_vae_decode_weights_are_prepared_before_first_denoise(monkeypatch):
    class FakeVideoVAE:
        def __init__(self):
            self.dtypes = []

        def prepare_decoder_autocast_weights(self, dtype):
            self.dtypes.append(dtype)
            return 144

    video_vae = FakeVideoVAE()
    server_args = SimpleNamespace(
        disable_autocast=False,
        pipeline_config=SimpleNamespace(vae_decode_precision="fp16"),
    )
    monkeypatch.setattr(decoding, "autocast_enabled", lambda *_args: True)

    stage = MiniMaxH3DecodingStage(
        video_vae=video_vae,
        audio_vae=None,
        server_args=server_args,
    )

    assert video_vae.dtypes == [torch.float16]
    assert stage._startup_converted_video_vae_linears == 144


@pytest.mark.parametrize(
    ("decode_precision", "disable_autocast"),
    (("fp32", False), ("fp16", True)),
)
def test_video_vae_decode_weight_preparation_respects_autocast(
    monkeypatch, decode_precision, disable_autocast
):
    class FakeVideoVAE:
        def __init__(self):
            self.dtypes = []

        def prepare_decoder_autocast_weights(self, dtype):
            self.dtypes.append(dtype)
            return 144

    video_vae = FakeVideoVAE()
    server_args = SimpleNamespace(
        disable_autocast=disable_autocast,
        pipeline_config=SimpleNamespace(vae_decode_precision=decode_precision),
    )
    monkeypatch.setattr(
        decoding,
        "autocast_enabled",
        lambda dtype, disabled: dtype != torch.float32 and not disabled,
    )

    stage = MiniMaxH3DecodingStage(
        video_vae=video_vae,
        audio_vae=None,
        server_args=server_args,
    )

    assert video_vae.dtypes == []
    assert stage._startup_converted_video_vae_linears == 0


def test_audio_vae_decode_warms_once_per_module():
    stage = MiniMaxH3DecodingStage(video_vae=None, audio_vae=None)
    first_audio_vae = object()
    second_audio_vae = object()
    latent = torch.ones(2, 32, 12)
    calls = []

    def decode(value):
        calls.append(value.clone())
        return value

    stage._warmup_audio_vae_decode(first_audio_vae, decode, latent)
    stage._warmup_audio_vae_decode(first_audio_vae, decode, latent)
    stage._warmup_audio_vae_decode(second_audio_vae, decode, latent)

    assert len(calls) == 2
    assert calls[0].shape == (2, 32, 8)
    assert calls[1].shape == (2, 32, 8)
    assert torch.count_nonzero(calls[0]) == 0
    assert torch.count_nonzero(calls[1]) == 0


def test_ffprobe_falls_back_when_stream_side_data_is_unknown(monkeypatch):
    material_io._ffprobe_entries = None
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        entries = command[command.index("-show_entries") + 1]
        if "stream_side_data" in entries:
            raise subprocess.CalledProcessError(
                1,
                command,
                stderr="ffprobe: No match for section 'stream_side_data'",
            )
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [{"codec_type": "audio", "sample_rate": "44100"}],
                    "format": {"format_name": "mp3", "duration": "1.0"},
                }
            )
        )

    monkeypatch.setattr(subprocess, "run", run)
    payload = material_io._ffprobe_media("/input/ref.mp3")

    assert len(calls) == 2
    assert "stream_side_data" in calls[0][calls[0].index("-show_entries") + 1]
    assert "stream_side_data" not in calls[1][calls[1].index("-show_entries") + 1]
    assert payload["format"]["format_name"] == "mp3"
    assert material_io._ffprobe_entries is not None
    assert "stream_side_data" not in material_io._ffprobe_entries


def test_video_transform_runs_once_and_qwen_samples_shared_rgb(monkeypatch):
    expected = np.arange(25 * 4 * 6 * 3, dtype=np.uint8).reshape(25, 4, 6, 3)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[-1] == "pipe:1":
            return SimpleNamespace(stdout=expected.tobytes())
        output_fd = int(command[-1].removeprefix("pipe:"))
        assert kwargs["pass_fds"] == (output_fd,)
        os.write(output_fd, expected.tobytes())
        return SimpleNamespace(stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    frames = reference_encoding.minimax_h3_decode_reference_video_frames(
        "/input/ref.mp4",
        target_width=6,
        target_height=4,
        target_frame_count=25,
        fps=24.0,
        start_time_seconds=2.25,
    )
    sampled = reference_encoding.minimax_h3_sample_reference_video_frames(frames)

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-vf") + 1] == (
        "fps=24,scale=6:4:flags=lanczos,setsar=1"
    )
    assert command[command.index("-frames:v") + 1] == "25"
    assert command[command.index("-ss") + 1] == "2.25"
    assert command.index("-ss") < command.index("-i")
    assert command[-5:-1] == ["-f", "rawvideo", "-pix_fmt", "rgb24"]
    assert command[-1].startswith("pipe:")
    assert "libx264" not in command
    if command[-1] != "pipe:1":
        assert frames.flags.writeable
    assert all(np.shares_memory(frame, frames) for frame in sampled["frames"])
    assert [int(frame[0, 0, 0]) for frame in sampled["frames"]] == [
        int(expected[index, 0, 0, 0]) for index in (0, 12, 24)
    ]
    assert sampled["block_timestamps"] == [0.25, 1.0]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux memfd")
def test_video_transform_can_share_one_host_decode(monkeypatch):
    expected = np.arange(25 * 4 * 6 * 3, dtype=np.uint8).reshape(25, 4, 6, 3)
    commands = []

    class FakeGroup:
        world_size = 2
        rank_in_group = 0
        cpu_group = object()

        def barrier(self):
            return None

    def all_gather_object(outputs, value, **_kwargs):
        outputs[:] = [value, value]

    def run(command, **_kwargs):
        commands.append(command)
        output_fd = int(command[-1].removeprefix("pipe:"))
        os.write(output_fd, expected.tobytes())
        return SimpleNamespace(stderr=b"")

    monkeypatch.setattr(reference_encoding, "get_world_group", FakeGroup)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(subprocess, "run", run)
    reference_encoding._reference_video_host_leader.cache_clear()

    try:
        frames = reference_encoding.minimax_h3_decode_reference_video_frames(
            "/input/ref.mp4",
            target_width=6,
            target_height=4,
            target_frame_count=25,
            share_across_replicas=True,
        )
    finally:
        reference_encoding._reference_video_host_leader.cache_clear()

    assert np.array_equal(frames, expected)
    assert frames.flags.writeable
    assert len(commands) == 1
    assert commands[0][-1].startswith("pipe:")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux memfd")
def test_shared_video_transform_falls_back_when_proc_fd_is_blocked(monkeypatch):
    expected = np.arange(25 * 4 * 6 * 3, dtype=np.uint8).reshape(25, 4, 6, 3)
    commands = []

    class FakeGroup:
        world_size = 2
        rank_in_group = 0
        cpu_group = object()

    def all_gather_object(outputs, value, **_kwargs):
        outputs[:] = [value, value]

    def run(command, **_kwargs):
        commands.append(command)
        os.write(int(command[-1].removeprefix("pipe:")), expected.tobytes())
        return SimpleNamespace(stderr=b"")

    real_open = os.open

    def guarded_open(path, flags):
        if str(path).startswith("/proc/"):
            raise PermissionError("blocked by test policy")
        return real_open(path, flags)

    monkeypatch.setattr(reference_encoding, "get_world_group", FakeGroup)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(os, "open", guarded_open)
    reference_encoding._reference_video_host_leader.cache_clear()
    try:
        frames = reference_encoding.minimax_h3_decode_reference_video_frames(
            "/input/ref.mp4",
            target_width=6,
            target_height=4,
            target_frame_count=25,
            share_across_replicas=True,
        )
    finally:
        reference_encoding._reference_video_host_leader.cache_clear()

    assert np.array_equal(frames, expected)
    assert len(commands) == 2


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux memfd")
def test_shared_video_transform_propagates_any_host_decode_failure(monkeypatch):
    class FakeGroup:
        world_size = 4
        rank_in_group = 0
        cpu_group = object()

    gather_index = 0

    def all_gather_object(outputs, value, **_kwargs):
        nonlocal gather_index
        if gather_index == 0:
            outputs[:] = ["host-a", "host-a", "host-b", "host-b"]
        else:
            outputs[:] = [
                value,
                None,
                (None, 0, "CalledProcessError: remote decode failed"),
                None,
            ]
        gather_index += 1

    monkeypatch.setattr(reference_encoding, "get_world_group", FakeGroup)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        reference_encoding,
        "_write_reference_video_to_fd",
        lambda _command, _fd: 1,
    )
    reference_encoding._reference_video_host_leader.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="remote decode failed"):
            reference_encoding._decode_reference_video_shared(["ffmpeg"])
    finally:
        reference_encoding._reference_video_host_leader.cache_clear()

    # The failure is resolved immediately after the shared state exchange;
    # no rank enters a mapping collective that another host skipped.
    assert gather_index == 2


def test_audio_decode_is_bounded_float_pcm_without_temp_files(monkeypatch):
    pcm = torch.arange(8, dtype=torch.float32).numpy().tobytes()
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(
                stdout=json.dumps(
                    {"streams": [{"channels": 6, "sample_rate": "44100"}]}
                )
            )
        return SimpleNamespace(stdout=pcm)

    monkeypatch.setattr(subprocess, "run", run)
    waveform, source_rate = reference_encoding._load_waveform(
        "/input/ref.mp4",
        material_chain="video.reference_preserve",
        max_duration_seconds=3.5,
        start_time_seconds=2.25,
    )

    ffmpeg = next(command for command in commands if command[0] == "ffmpeg")
    assert source_rate == 44100
    torch.testing.assert_close(
        waveform,
        torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=torch.float32),
    )
    assert ffmpeg[ffmpeg.index("-t") + 1] == "3.5"
    assert ffmpeg[ffmpeg.index("-ss") + 1] == "2.25"
    assert ffmpeg.index("-ss") < ffmpeg.index("-i")
    assert ffmpeg[-3:] == ["-f", "f32le", "pipe:1"]
