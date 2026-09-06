from __future__ import annotations

import pickle
import time
from types import SimpleNamespace

import pytest
import zmq

from sglang.multimodal_gen.runtime.disaggregation.orchestrator import (
    DiffusionServer,
    _TransferRequestState,
)
from sglang.multimodal_gen.runtime.disaggregation.request_state import RequestState
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
)
from sglang.multimodal_gen.runtime.disaggregation.transport import protocol
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferMsgType,
    decode_transfer_msg,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import pack_tensors
from sglang.multimodal_gen.runtime.media_encoder.protocol import MediaEncodeManifest


class _RecordingSocket:
    def __init__(self):
        self.messages = []

    def send_multipart(self, frames, *args, **kwargs):
        del args, kwargs
        self.messages.append(frames)


class _RequestSocket(_RecordingSocket):
    def __init__(self, request):
        super().__init__()
        self.request = request

    def recv_multipart(self, *args, **kwargs):
        del args, kwargs
        return [b"client", b"", pickle.dumps(self.request)]


@pytest.fixture
def server():
    instance = DiffusionServer(
        frontend_endpoint="inproc://frontend",
        encoder_work_endpoints=["inproc://encoder"],
        denoiser_work_endpoints=["inproc://denoiser"],
        decoder_work_endpoints=["inproc://decoder"],
        encoder_result_endpoint="inproc://encoder-result",
        denoiser_result_endpoint="inproc://denoiser-result",
        decoder_result_endpoint="inproc://decoder-result",
        encoder_capacity=1,
        denoiser_capacity_per_worker=1,
        decoder_capacity=1,
    )
    instance._encoder_pushes = [_RecordingSocket()]
    instance._denoiser_pushes = [_RecordingSocket()]
    instance._decoder_pushes = [_RecordingSocket()]
    instance._frontend = _RecordingSocket()
    try:
        yield instance
    finally:
        instance._context.term()


def _track_to_denoising(server: DiffusionServer, request_id: str) -> None:
    server.tracker.submit(request_id)
    server.tracker.transition(
        request_id, RequestState.ENCODER_RUNNING, encoder_instance=0
    )
    server.tracker.transition(request_id, RequestState.ENCODER_DONE)
    server.tracker.transition(request_id, RequestState.DENOISING_WAITING)
    server.tracker.transition(
        request_id, RequestState.DENOISING_RUNNING, denoiser_instance=0
    )


def test_event_loop_starts_with_bounded_socket_options():
    instance = DiffusionServer(
        frontend_endpoint="inproc://event-loop-frontend",
        encoder_work_endpoints=["inproc://event-loop-encoder"],
        denoiser_work_endpoints=["inproc://event-loop-denoiser"],
        decoder_work_endpoints=["inproc://event-loop-decoder"],
        encoder_result_endpoint="inproc://event-loop-encoder-result",
        denoiser_result_endpoint="inproc://event-loop-denoiser-result",
        decoder_result_endpoint="inproc://event-loop-decoder-result",
        max_pending_requests=3,
        control_queue_size=5,
    )
    instance.start()
    try:
        assert instance.wait_ready(timeout=2.0)
        assert instance._thread is not None and instance._thread.is_alive()
        assert instance._frontend.getsockopt(zmq.SNDHWM) == 5
        assert instance._frontend.getsockopt(zmq.RCVHWM) == 3
        for socket in (
            *instance._encoder_pushes,
            *instance._denoiser_pushes,
            *instance._decoder_pushes,
        ):
            assert socket.getsockopt(zmq.SNDHWM) == 5
            assert socket.getsockopt(zmq.SNDTIMEO) == 0
            assert socket.getsockopt(zmq.LINGER) == 0
            assert socket.getsockopt(zmq.IMMEDIATE) == 1
    finally:
        instance.stop()


def test_registration_readiness_requires_every_configured_role(server):
    assert server.registration_readiness() == {
        "status": "not_ready",
        "ready": False,
        "roles": {
            "encoder": {"configured": 1, "registered": 0},
            "denoiser": {"configured": 1, "registered": 0},
            "decoder": {"configured": 1, "registered": 0},
        },
    }

    for role, endpoint in (
        ("encoder", "inproc://encoder"),
        ("denoiser", "inproc://denoiser"),
        ("decoder", "inproc://decoder"),
    ):
        server._handle_transfer_register(
            {
                "role": role,
                "work_endpoint": endpoint,
                "transfer_backend": "tcp",
                "session_id": f"zmq+tcp://127.0.0.1:{32000 + len(role)}",
                "pool_ptr": 1,
                "pool_size": 1024,
                "preallocated_slots": [],
            }
        )

    assert server.registration_readiness()["ready"] is True
    assert server.registration_readiness()["status"] == "ok"


def test_registration_readiness_is_available_over_head_control_rpc(server):
    frontend = _RequestSocket({"method": "disagg_registration_readiness"})

    server._handle_client_request(frontend)

    assert pickle.loads(frontend.messages[-1][-1]) == server.registration_readiness()


def test_registration_readiness_rejects_stale_worker(server):
    for role, endpoint in (
        ("encoder", "inproc://encoder"),
        ("denoiser", "inproc://denoiser"),
        ("decoder", "inproc://decoder"),
    ):
        server._handle_transfer_register(
            {
                "role": role,
                "work_endpoint": endpoint,
                "transfer_backend": "tcp",
                "session_id": f"session-{role}",
            }
        )

    server._denoiser_peers[0]["registered_at"] -= (
        protocol.DISAGG_REGISTRATION_STALE_AFTER_S + 1
    )

    readiness = server.registration_readiness()
    assert readiness["ready"] is False
    assert readiness["roles"]["encoder"]["registered"] == 1
    assert readiness["roles"]["denoiser"]["registered"] == 0
    assert readiness["roles"]["decoder"]["registered"] == 1


def test_same_session_heartbeat_preserves_allocated_prealloc_slots(server):
    registration = {
        "role": "denoiser",
        "work_endpoint": "inproc://denoiser",
        "transfer_backend": "tcp",
        "session_id": "stable-session",
        "pool_ptr": 100,
        "pool_size": 4096,
        "preallocated_slots": [
            {"slot_id": 0, "offset": 0, "size": 1024, "addr": 100},
            {"slot_id": 1, "offset": 1024, "size": 1024, "addr": 1124},
        ],
    }
    server._handle_transfer_register(registration)
    peer = server._denoiser_peers[0]
    peer["free_preallocated_slots"].pop()
    first_registered_at = peer["registered_at"]

    server._handle_transfer_register(registration)

    assert server._denoiser_peers[0] is peer
    assert peer["free_preallocated_slots"] == registration["preallocated_slots"][:1]
    assert peer["registered_at"] >= first_registered_at


class _HeartbeatScheduler(SchedulerDisaggMixin):
    pass


def test_registration_heartbeat_recovers_after_head_restart(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin."
        "DISAGG_REGISTRATION_HEARTBEAT_INTERVAL_S",
        0.01,
    )
    context = zmq.Context()
    first_head = context.socket(zmq.PULL)
    first_head.setsockopt(zmq.LINGER, 0)
    first_head.bind("tcp://127.0.0.1:*")
    endpoint = first_head.getsockopt_string(zmq.LAST_ENDPOINT)
    scheduler = _HeartbeatScheduler()
    scheduler.context = context
    scheduler.server_args = SimpleNamespace(pool_result_endpoint=endpoint)
    scheduler._disagg_role = RoleType.ENCODER
    scheduler._registration_heartbeat_stop = None
    scheduler._registration_heartbeat_thread = None
    scheduler._registration_heartbeat_zmq = None
    register_msg = protocol.TransferRegisterMsg(
        role="encoder",
        transfer_backend="tcp",
        session_id="heartbeat-session",
        work_endpoint="tcp://encoder:31000",
    )

    scheduler._start_disagg_registration_heartbeat(register_msg)
    try:
        assert first_head.poll(1000, zmq.POLLIN)
        first_registration = decode_transfer_msg(first_head.recv_multipart())
        assert first_registration["session_id"] == "heartbeat-session"

        first_head.close()
        replacement_head = context.socket(zmq.PULL)
        replacement_head.setsockopt(zmq.LINGER, 0)
        replacement_head.bind(endpoint)
        try:
            assert replacement_head.poll(1000, zmq.POLLIN)
            replacement_registration = decode_transfer_msg(
                replacement_head.recv_multipart()
            )
            assert replacement_registration == first_registration
        finally:
            replacement_head.close()
    finally:
        scheduler._stop_disagg_registration_heartbeat()
        context.term()

    assert scheduler._registration_heartbeat_thread is None
    assert scheduler._registration_heartbeat_zmq is None


def _encoder_to_denoiser_state(transfer_id: str) -> _TransferRequestState:
    return _TransferRequestState(
        transfer_id=transfer_id,
        source_role=RoleType.ENCODER,
        destination_role=RoleType.DENOISER,
        sender_transfer_backend="tcp",
        data_size=1024,
        manifest={"value": [{"offset": 0, "shape": [1024], "dtype": "uint8"}]},
        sender_instance=0,
        receiver_instance=0,
    )


def _transfer_messages(socket: _RecordingSocket) -> list[dict]:
    return [decode_transfer_msg(frames) for frames in socket.messages]


def _media_manifest(request_id: str, attempt_id: str) -> dict:
    return MediaEncodeManifest(
        request_id=request_id,
        attempt_id=attempt_id,
        uri="https://objects.example/h3-output.mp4",
        object_key="h3-output.mp4",
        storage="s3",
        byte_size=1234,
        sha256="1" * 64,
        container="mp4",
        video_codec="h264",
        pixel_format="yuv420p",
        audio_codec="aac",
        width=1344,
        height=768,
        frame_count=362,
        fps=24,
        duration_seconds=362 / 24,
        audio_sample_rate=32_000,
        audio_channels=2,
        source_payload_sha256="2" * 64,
        model_identity={"partition": "fl2va"},
    ).to_dict()


def _track_to_decoder(
    server: DiffusionServer, request_id: str, transfer_id: str
) -> None:
    _track_to_denoising(server, request_id)
    server.tracker.transition(request_id, RequestState.DENOISING_DONE)
    server.tracker.transition(request_id, RequestState.DECODER_WAITING)
    server.tracker.transition(
        request_id, RequestState.DECODER_RUNNING, decoder_instance=0
    )
    server._pending[request_id] = b"client"
    server._transfer_state[request_id] = _TransferRequestState(
        transfer_id=transfer_id,
        source_role=RoleType.DENOISER,
        destination_role=RoleType.DECODER,
        sender_transfer_backend="tcp",
        data_size=1024,
        manifest={"value": [{"offset": 0, "shape": [1024], "dtype": "uint8"}]},
        sender_instance=0,
        receiver_instance=0,
    )
    server._decoder_free_slots[0] = 0


def test_decoder_manifest_is_validated_and_returned_without_tensors(server):
    request_id = "manifest-result"
    transfer_id = "manifest-attempt"
    _track_to_decoder(server, request_id, transfer_id)
    metadata, _buffers = pack_tensors(
        {},
        {
            "request_id": request_id,
            "_transfer_id": transfer_id,
            "media_manifest": _media_manifest(request_id, transfer_id),
        },
    )

    server._handle_decoder_result_frames([metadata])

    assert server._decoder_free_slots == [1]
    assert request_id not in server._pending
    returned = pickle.loads(server._frontend.messages[-1][-1])
    assert returned.error is None
    assert returned.output is None
    assert returned.audio is None
    assert returned.media_manifest["uri"].endswith("h3-output.mp4")


def test_decoder_manifest_rejects_transfer_identity_mismatch(server):
    request_id = "manifest-mismatch"
    transfer_id = "expected-attempt"
    _track_to_decoder(server, request_id, transfer_id)
    metadata, _buffers = pack_tensors(
        {},
        {
            "request_id": request_id,
            "_transfer_id": transfer_id,
            "media_manifest": _media_manifest(request_id, "stale-attempt"),
        },
    )

    server._handle_decoder_result_frames([metadata])

    returned = pickle.loads(server._frontend.messages[-1][-1])
    assert returned.media_manifest is None
    assert "identity does not match" in returned.error


def test_push_failure_releases_each_capacity_once_and_never_sends_ready(server):
    request_id = "push-failure"
    transfer_id = "edge-attempt-1"
    _track_to_denoising(server, request_id)
    state = _encoder_to_denoiser_state(transfer_id)
    server._transfer_state[request_id] = state
    server._encoder_free_slots[0] = 0
    server._denoiser_free_slots[0] = 0

    failed = {
        "request_id": request_id,
        "transfer_id": transfer_id,
        "error": "checksum mismatch",
    }
    server._handle_transfer_pushed(failed, RoleType.ENCODER)
    server._handle_transfer_pushed(failed, RoleType.ENCODER)

    assert server._encoder_free_slots == [1]
    assert server._denoiser_free_slots == [1]
    assert request_id not in server._transfer_state
    assert all(
        msg["msg_type"] == TransferMsgType.ABORT
        for msg in _transfer_messages(server._denoiser_pushes[0])
    )


def test_duplicate_pushed_and_done_messages_do_not_inflate_capacity(server):
    request_id = "duplicate-control"
    transfer_id = "edge-attempt-2"
    _track_to_denoising(server, request_id)
    state = _encoder_to_denoiser_state(transfer_id)
    server._transfer_state[request_id] = state
    server._encoder_free_slots[0] = 0
    server._denoiser_free_slots[0] = 0

    pushed = {"request_id": request_id, "transfer_id": transfer_id}
    server._handle_transfer_pushed(pushed, RoleType.ENCODER)
    server._handle_transfer_pushed(pushed, RoleType.ENCODER)

    ready = [
        msg
        for msg in _transfer_messages(server._denoiser_pushes[0])
        if msg["msg_type"] == TransferMsgType.READY
    ]
    assert len(ready) == 1
    assert server._encoder_free_slots == [1]
    assert server._denoiser_free_slots == [0]

    done = {
        "request_id": request_id,
        "completed_transfer_id": transfer_id,
        "staged_for_decoder": False,
    }
    server._handle_transfer_done(done, RoleType.DENOISER)
    server._handle_transfer_done(done, RoleType.DENOISER)

    assert server._denoiser_free_slots == [1]
    assert request_id not in server._transfer_state


def test_preallocated_slot_is_recycled_once_after_failure(server):
    request_id = "prealloc-failure"
    transfer_id = "edge-attempt-3"
    _track_to_denoising(server, request_id)
    state = _encoder_to_denoiser_state(transfer_id)
    state.receiver_pool_ptr = 1000
    state.receiver_slot_offset = 128
    state.receiver_slot_size = 4096
    state.prealloc_slot_id = 7
    server._transfer_state[request_id] = state
    server._denoiser_peers[0] = {"free_preallocated_slots": []}
    server._encoder_free_slots[0] = 0
    server._denoiser_free_slots[0] = 0

    server._fail_active_transfer(request_id, state, "failed")
    server._fail_active_transfer(request_id, state, "failed again")

    free_slots = server._denoiser_peers[0]["free_preallocated_slots"]
    assert free_slots == [{"offset": 128, "size": 4096, "slot_id": 7, "addr": 1128}]


def test_backend_mismatch_fails_before_receiver_allocation(server):
    request_id = "backend-mismatch"
    transfer_id = "edge-attempt-4"
    _track_to_denoising(server, request_id)
    state = _encoder_to_denoiser_state(transfer_id)
    server._transfer_state[request_id] = state
    server._encoder_free_slots[0] = 0
    server._denoiser_free_slots[0] = 1
    server._denoiser_peers[0] = {
        "transfer_backend": "mooncake",
        "free_preallocated_slots": [],
    }

    server._transfer_dispatch_to_denoiser(request_id, state, 0)

    assert server._encoder_free_slots == [1]
    assert server._denoiser_free_slots == [1]
    assert request_id not in server._transfer_state
    assert server._denoiser_pushes[0].messages == []


def test_timeout_after_ready_quarantines_receiver_until_matching_done(server):
    request_id = "ready-timeout"
    transfer_id = "edge-attempt-5"
    _track_to_denoising(server, request_id)
    record = server.tracker.get(request_id)
    assert record is not None
    record.submit_time = time.monotonic() - 10
    state = _encoder_to_denoiser_state(transfer_id)
    state.push_completed = True
    state.ready_sent = True
    state.sender_capacity_released = True
    server._transfer_state[request_id] = state
    server._encoder_free_slots[0] = 1
    server._denoiser_free_slots[0] = 0
    server._timeout_s = 1

    server._handle_timeouts()

    assert state.client_completed
    assert request_id in server._transfer_state
    assert server._denoiser_free_slots == [0]

    server._handle_transfer_done(
        {"request_id": request_id, "completed_transfer_id": transfer_id},
        RoleType.DENOISER,
    )
    server._handle_transfer_done(
        {"request_id": request_id, "completed_transfer_id": transfer_id},
        RoleType.DENOISER,
    )

    assert server._denoiser_free_slots == [1]
    assert request_id not in server._transfer_state


def test_timeout_during_encoder_compute_keeps_capacity_until_late_stage(server):
    request_id = "encoder-timeout"
    transfer_id = "edge-attempt-6"
    server.tracker.submit(request_id)
    server.tracker.transition(
        request_id, RequestState.ENCODER_RUNNING, encoder_instance=0
    )
    record = server.tracker.get(request_id)
    assert record is not None
    record.submit_time = time.monotonic() - 10
    server._encoder_free_slots[0] = 0
    server._timeout_s = 1

    server._handle_timeouts()

    assert server._encoder_free_slots == [0]
    assert server._orphaned_compute_slots[request_id] == (RoleType.ENCODER, 0)

    server._handle_transfer_staged(
        {
            "request_id": request_id,
            "transfer_id": transfer_id,
            "data_size": 1024,
        },
        RoleType.ENCODER,
    )
    server._handle_transfer_staged(
        {
            "request_id": request_id,
            "transfer_id": transfer_id,
            "data_size": 1024,
        },
        RoleType.ENCODER,
    )

    assert server._encoder_free_slots == [1]
    assert request_id not in server._orphaned_compute_slots
    aborts = _transfer_messages(server._encoder_pushes[0])
    assert len(aborts) == 1
    assert aborts[0]["msg_type"] == TransferMsgType.ABORT
    assert aborts[0]["transfer_id"] == transfer_id
