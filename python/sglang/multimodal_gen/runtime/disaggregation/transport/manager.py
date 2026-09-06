# SPDX-License-Identifier: Apache-2.0
"""Per-instance transfer manager for disaggregated diffusion roles."""

import logging
import threading
import uuid
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    SlotHandle,
    TransferTensorBuffer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    BaseTransferEngine,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

logger = logging.getLogger(__name__)


@dataclass
class StagedTransfer:
    request_id: str
    transfer_id: str
    slot: SlotHandle | None
    data_size: int
    manifest: dict
    scalar_fields: dict = field(default_factory=dict)


@dataclass
class PendingReceive:
    request_id: str
    transfer_id: str
    slot: SlotHandle
    data_size: int
    preallocated: bool = False


class DiffusionTransferManager:
    """Manages tensor transfers for a single role instance.

    Owns a TransferTensorBuffer (memory pool) and a BaseTransferEngine (RDMA or mock).
    """

    def __init__(
        self,
        engine: BaseTransferEngine,
        buffer: TransferTensorBuffer,
    ):
        self._engine = engine
        self._buffer = buffer
        self._lock = threading.Lock()
        self._last_error: str | None = None

        self._engine.register_buffer(self._buffer.pool_data_ptr, self._buffer.pool_size)

        self._staged: dict[str, StagedTransfer] = {}
        self._pending_receives: dict[str, PendingReceive] = {}

        logger.info(
            "DiffusionTransferManager initialized: session=%s, pool=%d bytes",
            self._engine.session_id,
            self._buffer.pool_size,
        )

    @property
    def session_id(self) -> str:
        return self._engine.session_id

    @property
    def backend_name(self) -> str:
        return self._engine.backend_name

    @property
    def pool_data_ptr(self) -> int:
        return self._buffer.pool_data_ptr

    @property
    def pool_size(self) -> int:
        return self._buffer.pool_size

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def stage_tensors_async(
        self,
        request_id: str,
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
        scalar_fields: dict | None = None,
        stream: torch.Stream | None = None,
        transfer_id: str | None = None,
    ) -> tuple[StagedTransfer | None, torch.Event | None]:
        """Stage GPU tensors, returning a CUDA event instead of blocking.

        Caller MUST wait on the event before reading buffer data.
        """
        transfer_id = transfer_id or uuid.uuid4().hex
        if not isinstance(transfer_id, str) or not transfer_id:
            raise ValueError("transfer_id must be a non-empty string")

        total_size = 0
        for t in tensor_fields.values():
            if t is None:
                continue
            if isinstance(t, list):
                for ti in t:
                    if ti is None:
                        continue
                    if total_size:
                        total_size = (total_size + 511) & ~511
                    total_size += ti.nelement() * ti.element_size()
            else:
                if total_size:
                    total_size = (total_size + 511) & ~511
                total_size += t.nelement() * t.element_size()

        if total_size == 0:
            staged = StagedTransfer(
                request_id=request_id,
                transfer_id=transfer_id,
                slot=None,
                data_size=0,
                manifest={},
                scalar_fields=scalar_fields or {},
            )
            with self._lock:
                if request_id in self._staged:
                    self._last_error = f"duplicate staged request {request_id!r}"
                    return None, None
                self._staged[request_id] = staged
            return staged, None

        with self._lock:
            if request_id in self._staged:
                self._last_error = f"duplicate staged request {request_id!r}"
                return None, None
            slot = self._buffer.allocate(total_size, request_id)
            if slot is None:
                logger.warning(
                    "TransferManager: failed to allocate %d bytes for %s",
                    total_size,
                    request_id,
                )
                return None, None
            try:
                manifest = self._buffer.write_tensors_from_gpu(
                    slot, tensor_fields, stream
                )
            except Exception:
                self._buffer.free(slot)
                raise
            staged = StagedTransfer(
                request_id=request_id,
                transfer_id=transfer_id,
                slot=slot,
                data_size=total_size,
                manifest=manifest,
                scalar_fields=scalar_fields or {},
            )
            self._staged[request_id] = staged

        d2h_event = None
        if stream is not None:
            d2h_event = torch.get_device_module().Event()
            d2h_event.record(stream)
        elif torch.get_device_module().is_available():
            d2h_event = torch.get_device_module().Event()
            d2h_event.record(torch.get_device_module().current_stream())

        logger.debug(
            "TransferManager: staged_async %s (%d bytes, offset=%d)",
            request_id,
            total_size,
            slot.offset,
        )
        return staged, d2h_event

    def load_tensors_async(
        self,
        request_id: str,
        manifest: dict,
        device: torch.device | str = current_platform.device_type,
        stream: torch.Stream | None = None,
        transfer_id: str | None = None,
    ) -> tuple[
        dict[str, torch.Tensor | list[torch.Tensor]],
        torch.get_device_module().Event | None,
    ]:
        """Load tensors from receive slot to GPU, returning a CUDA event.

        Caller MUST wait on the event before using the returned tensors.
        """
        with self._lock:
            pending = self._pending_receives.get(request_id)

        if pending is None:
            raise ValueError(
                f"TransferManager: no pending receive slot for {request_id}"
            )
        if transfer_id is not None and pending.transfer_id != transfer_id:
            raise ValueError(
                f"TransferManager: stale receive transfer {transfer_id!r} for "
                f"{request_id}; expected {pending.transfer_id!r}"
            )

        tensors = self._buffer.read_tensors_from_manifest(
            pending.slot,
            manifest,
            device=device,
            stream=stream,
            payload_size=pending.data_size,
        )

        load_event = None
        if stream is not None:
            load_event = torch.get_device_module().Event()
            load_event.record(stream)
        elif torch.get_device_module().is_available():
            load_event = torch.get_device_module().Event()
            load_event.record(torch.get_device_module().current_stream())

        logger.debug(
            "TransferManager: loaded_async %d tensor fields for %s to %s",
            len(tensors),
            request_id,
            device,
        )
        return tensors, load_event

    def push_to_peer(
        self,
        request_id: str,
        dest_session_id: str,
        dest_addr: int,
        transfer_size: int,
        transfer_id: str | None = None,
    ) -> bool:
        """Push staged data to a remote peer's buffer via RDMA. Returns True on success."""
        with self._lock:
            staged = self._staged.get(request_id)

        if staged is None:
            logger.error("TransferManager: no staged transfer for %s", request_id)
            return False

        transfer_id = transfer_id or staged.transfer_id
        if staged.transfer_id != transfer_id:
            self._last_error = (
                f"stale transfer {transfer_id!r} for {request_id!r}; "
                f"expected {staged.transfer_id!r}"
            )
            return False

        if staged.slot is None:
            return True

        if transfer_size != staged.data_size:
            self._last_error = (
                f"transfer size {transfer_size} does not match staged payload "
                f"size {staged.data_size}"
            )
            logger.error("TransferManager: %s", self._last_error)
            return False

        src_addr = self._buffer.pool_data_ptr + staged.slot.offset
        ret = self._engine.transfer_sync(
            dest_session_id,
            src_addr,
            dest_addr,
            transfer_size,
            transfer_id=transfer_id,
        )

        if ret == 0:
            logger.debug(
                "TransferManager: pushed %s (%d bytes) to %s",
                request_id,
                transfer_size,
                dest_session_id,
            )
        else:
            self._last_error = self._engine.last_error or f"engine error {ret}"
            logger.error(
                "TransferManager: push failed for %s (ret=%d, error=%s)",
                request_id,
                ret,
                self._last_error,
            )

        if ret == 0:
            self._last_error = None
        return ret == 0

    def free_staged(self, request_id: str, transfer_id: str | None = None) -> bool:
        with self._lock:
            staged = self._staged.get(request_id)
            if staged is None or (
                transfer_id is not None and staged.transfer_id != transfer_id
            ):
                return False
            self._staged.pop(request_id)

        if staged and staged.slot is not None:
            self._buffer.free(staged.slot)
            logger.debug("TransferManager: freed staged slot for %s", request_id)
        return True

    def allocate_receive_slot(
        self, request_id: str, size: int, transfer_id: str | None = None
    ) -> PendingReceive | None:
        """Allocate a local buffer slot to receive incoming data."""
        transfer_id = transfer_id or request_id
        if not isinstance(transfer_id, str) or not transfer_id:
            self._last_error = "receive transfer_id must be a non-empty string"
            return None
        with self._lock:
            existing = self._pending_receives.get(request_id)
            if existing is not None:
                if (
                    existing.transfer_id == transfer_id
                    and existing.data_size == size
                    and not existing.preallocated
                ):
                    return existing
                self._last_error = f"conflicting receive allocation for {request_id!r}"
                return None
            slot = self._buffer.allocate(size, request_id)
            if slot is None:
                logger.warning(
                    "TransferManager: failed to allocate receive slot (%d bytes) for %s",
                    size,
                    request_id,
                )
                return None
            pending = PendingReceive(
                request_id=request_id,
                transfer_id=transfer_id,
                slot=slot,
                data_size=size,
            )
            self._pending_receives[request_id] = pending

        logger.debug(
            "TransferManager: allocated receive slot for %s (offset=%d, size=%d)",
            request_id,
            slot.offset,
            slot.size,
        )
        return pending

    def register_prealloc_as_receive(
        self,
        request_id: str,
        slot: "SlotHandle",
        data_size: int,
        transfer_id: str | None = None,
    ) -> "PendingReceive":
        """Register a pre-allocated slot as a pending receive (fast path)."""
        if (
            isinstance(data_size, bool)
            or not isinstance(data_size, int)
            or data_size <= 0
            or data_size > slot.size
        ):
            raise ValueError("preallocated receive payload exceeds its slot")
        transfer_id = transfer_id or request_id
        if not isinstance(transfer_id, str) or not transfer_id:
            raise ValueError("receive transfer_id must be a non-empty string")
        pending = PendingReceive(
            request_id=request_id,
            transfer_id=transfer_id,
            slot=slot,
            data_size=data_size,
            preallocated=True,
        )
        with self._lock:
            existing = self._pending_receives.get(request_id)
            if existing is not None:
                if (
                    existing.preallocated
                    and existing.transfer_id == transfer_id
                    and existing.slot.offset == slot.offset
                    and existing.data_size == data_size
                ):
                    return existing
                raise ValueError(f"conflicting preallocated receive for {request_id!r}")
            self._pending_receives[request_id] = pending
        return pending

    def free_receive_slot(
        self, request_id: str, transfer_id: str | None = None
    ) -> bool:
        with self._lock:
            pending = self._pending_receives.get(request_id)
            if pending is None or (
                transfer_id is not None and pending.transfer_id != transfer_id
            ):
                return False
            self._pending_receives.pop(request_id)

        if pending and not pending.preallocated:
            self._buffer.free(pending.slot)
            logger.debug("TransferManager: freed receive slot for %s", request_id)
        return True

    def cleanup(self) -> None:
        try:
            self._engine.deregister_buffer(self._buffer.pool_data_ptr)
        finally:
            self._engine.close()
        logger.info("DiffusionTransferManager cleaned up")
