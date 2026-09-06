from __future__ import annotations

import ctypes

import torch
import pytest

from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    SlotHandle,
    validate_tensor_manifest,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    MockTransferEngine,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.manager import (
    DiffusionTransferManager,
)


class _HostBuffer:
    def __init__(self, pool_size=4096):
        self._storage = ctypes.create_string_buffer(pool_size)
        self.pool_data_ptr = ctypes.addressof(self._storage)
        self.pool_size = pool_size
        self.freed = []
        self.allocations = 0

    def allocate(self, size, request_id):
        if size > self.pool_size:
            return None
        self.allocations += 1
        return SlotHandle(request_id=request_id, offset=0, size=self.pool_size)

    def free(self, slot):
        self.freed.append(slot)
        return True

    def write_tensors_from_gpu(self, slot, tensor_fields, stream):
        del slot, tensor_fields, stream
        return {}


def test_transfer_manager_sends_manifest_high_water_not_buddy_slot_size():
    sender_engine = MockTransferEngine()
    receiver_engine = MockTransferEngine()
    sender_buffer = _HostBuffer()
    receiver_buffer = _HostBuffer()
    sender = DiffusionTransferManager(sender_engine, sender_buffer)
    receiver = DiffusionTransferManager(receiver_engine, receiver_buffer)

    try:
        staged, event = sender.stage_tensors_async(
            "request-a",
            {
                "first": torch.empty(513, dtype=torch.uint8),
                "second": torch.empty(2, dtype=torch.uint8),
            },
        )

        if event is not None:
            event.synchronize()
        assert staged is not None
        assert staged.data_size == 1026
        assert staged.slot.size == 4096
        assert sender.push_to_peer(
            "request-a",
            receiver.session_id,
            receiver.pool_data_ptr,
            staged.data_size,
        )
    finally:
        sender.cleanup()
        receiver.cleanup()


def test_transfer_manager_rejects_control_plane_size_mismatch():
    sender_engine = MockTransferEngine()
    receiver_engine = MockTransferEngine()
    sender = DiffusionTransferManager(sender_engine, _HostBuffer())
    receiver = DiffusionTransferManager(receiver_engine, _HostBuffer())

    try:
        staged, _ = sender.stage_tensors_async(
            "request-b", {"value": torch.empty(31, dtype=torch.uint8)}
        )
        assert staged is not None

        assert not sender.push_to_peer(
            "request-b",
            receiver.session_id,
            receiver.pool_data_ptr,
            staged.data_size + 1,
        )
        assert "does not match staged payload" in sender.last_error
    finally:
        sender.cleanup()
        receiver.cleanup()


def test_receive_allocation_is_idempotent_and_conflicts_fail_closed():
    engine = MockTransferEngine()
    buffer = _HostBuffer()
    manager = DiffusionTransferManager(engine, buffer)

    try:
        first = manager.allocate_receive_slot("request-c", 1024)
        duplicate = manager.allocate_receive_slot("request-c", 1024)
        conflict = manager.allocate_receive_slot("request-c", 2048)

        assert first is duplicate
        assert conflict is None
        assert buffer.allocations == 1
        assert "conflicting receive allocation" in manager.last_error
    finally:
        manager.free_receive_slot("request-c")
        manager.cleanup()


def test_abort_does_not_return_preallocated_slot_to_buddy_allocator():
    engine = MockTransferEngine()
    buffer = _HostBuffer()
    manager = DiffusionTransferManager(engine, buffer)
    slot = SlotHandle(request_id="prealloc", offset=0, size=4096)

    try:
        manager.register_prealloc_as_receive("request-d", slot, 512)
        manager.free_receive_slot("request-d")

        assert buffer.freed == []
    finally:
        manager.cleanup()


def test_tensor_manifest_accepts_aligned_actual_payload_high_water():
    validate_tensor_manifest(
        {
            "first": [{"offset": 0, "shape": [513], "dtype": "uint8"}],
            "second": [{"offset": 1024, "shape": [2], "dtype": "uint8"}],
        },
        payload_size=1026,
        slot_size=4096,
    )


@pytest.mark.parametrize(
    ("manifest", "payload_size", "message"),
    [
        (
            {"value": [{"offset": -1, "shape": [1], "dtype": "uint8"}]},
            1,
            "invalid offset",
        ),
        (
            {"value": [{"offset": 0, "shape": [2], "dtype": "unknown"}]},
            2,
            "invalid dtype",
        ),
        (
            {
                "a": [{"offset": 0, "shape": [4], "dtype": "uint8"}],
                "b": [{"offset": 3, "shape": [2], "dtype": "uint8"}],
            },
            5,
            "overlapping entries",
        ),
        (
            {"value": [{"offset": 0, "shape": [4], "dtype": "uint8"}]},
            5,
            "high water",
        ),
        (
            {
                "value": [
                    {"offset": 0, "shape": [1], "dtype": "uint8"},
                    {
                        "offset": 1,
                        "shape": [1],
                        "dtype": "uint8",
                        "list_index": 1,
                    },
                ]
            },
            2,
            "mixes list and scalar",
        ),
    ],
)
def test_tensor_manifest_rejects_untrusted_layouts(manifest, payload_size, message):
    with pytest.raises(ValueError, match=message):
        validate_tensor_manifest(manifest, payload_size=payload_size, slot_size=4096)
