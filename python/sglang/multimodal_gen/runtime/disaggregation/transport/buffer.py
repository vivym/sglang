# SPDX-License-Identifier: Apache-2.0
"""TransferTensorBuffer: memory staging area for disaggregated tensor transfer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.allocator import (
    BuddyAllocator,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import (
    str_to_dtype,
)

logger = logging.getLogger(__name__)


def validate_tensor_manifest(
    manifest: dict[str, list[dict]], *, payload_size: int, slot_size: int
) -> None:
    """Validate every tensor view before exposing received buffer memory."""
    if not isinstance(manifest, dict):
        raise ValueError("transfer tensor manifest must be an object")
    if (
        isinstance(payload_size, bool)
        or not isinstance(payload_size, int)
        or payload_size < 0
        or payload_size > slot_size
    ):
        raise ValueError("transfer payload size is outside the receive slot")

    intervals: list[tuple[int, int, str]] = []
    for name, entries in manifest.items():
        if not isinstance(name, str) or not name:
            raise ValueError("transfer tensor field names must be non-empty strings")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"transfer tensor field {name!r} has no entries")
        has_list_indices = [
            "list_index" in entry for entry in entries if isinstance(entry, dict)
        ]
        if len(has_list_indices) != len(entries) or (
            any(has_list_indices) and not all(has_list_indices)
        ):
            raise ValueError(
                f"transfer tensor field {name!r} mixes list and scalar entries"
            )
        if not any(has_list_indices) and len(entries) != 1:
            raise ValueError(
                f"transfer tensor field {name!r} has duplicate scalar entries"
            )

        seen_list_indices: set[int] = set()
        for entry in entries:
            unknown = set(entry) - {"offset", "shape", "dtype", "list_index"}
            if unknown:
                raise ValueError(
                    f"transfer tensor field {name!r} has unknown entry keys {sorted(unknown)!r}"
                )
            offset = entry.get("offset")
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ValueError(f"transfer tensor field {name!r} has invalid offset")
            shape = entry.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) > 16
                or any(
                    isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
                    for dim in shape
                )
            ):
                raise ValueError(f"transfer tensor field {name!r} has invalid shape")
            try:
                dtype = str_to_dtype(entry.get("dtype"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"transfer tensor field {name!r} has invalid dtype"
                ) from exc
            if "list_index" in entry:
                list_index = entry["list_index"]
                if (
                    isinstance(list_index, bool)
                    or not isinstance(list_index, int)
                    or list_index < 0
                    or list_index in seen_list_indices
                ):
                    raise ValueError(
                        f"transfer tensor field {name!r} has invalid list_index"
                    )
                seen_list_indices.add(list_index)

            numel = 1
            for dim in shape:
                numel *= dim
                if numel > payload_size:
                    break
            nbytes = numel * torch.empty((), dtype=dtype).element_size()
            end = offset + nbytes
            if end > payload_size:
                raise ValueError(
                    f"transfer tensor field {name!r} exceeds the received payload"
                )
            if nbytes:
                intervals.append((offset, end, name))

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                "transfer tensor manifest contains overlapping entries: "
                f"{previous[2]!r} and {current[2]!r}"
            )
    high_water = max((end for _, end, _ in intervals), default=0)
    if high_water != payload_size:
        raise ValueError(
            f"transfer tensor manifest high water {high_water} does not match "
            f"payload size {payload_size}"
        )


@dataclass
class SlotHandle:
    request_id: str
    offset: int  # byte offset in the pool
    size: int  # allocated size in bytes
    tensor_views: dict[str, torch.Tensor | list[torch.Tensor]] = field(
        default_factory=dict
    )


class TransferTensorBuffer:
    """Memory pool for staging tensor payloads between roles.

    Wraps a contiguous block of memory (CPU pinned or GPU) with a BuddyAllocator.
    """

    def __init__(
        self,
        pool_size: int,
        min_block_size: int = 1 << 20,
        role_name: str = "unknown",
        device: str = "cpu",
    ):
        self._role_name = role_name
        self._device = device
        self._allocator = BuddyAllocator(pool_size, min_block_size)
        actual_size = self._allocator.pool_size

        if device == "cpu":
            self._pool = torch.empty(actual_size, dtype=torch.uint8, pin_memory=True)
        else:
            self._pool = torch.empty(actual_size, dtype=torch.uint8, device=device)
        self._pool_ptr = self._pool.data_ptr()

        pool_location = "pinned CPU" if device == "cpu" else f"GPU ({device})"
        logger.info(
            "TransferTensorBuffer[%s]: allocated %d MiB %s memory (min_block=%d KiB)",
            role_name,
            actual_size >> 20,
            pool_location,
            min_block_size >> 10,
        )

    @property
    def pool_size(self) -> int:
        return self._allocator.pool_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def pool_data_ptr(self) -> int:
        return self._pool_ptr

    def allocate(self, size: int, request_id: str) -> SlotHandle | None:
        """Allocate a slot. Returns None if pool is full."""
        offset = self._allocator.allocate(size, request_id=request_id)
        if offset is None:
            logger.warning(
                "TransferTensorBuffer[%s]: allocation failed for %s (%d bytes). "
                "Pool stats: %s",
                self._role_name,
                request_id,
                size,
                self._allocator.get_stats(),
            )
            return None

        block = self._allocator.get_block_info(offset)
        return SlotHandle(
            request_id=request_id,
            offset=offset,
            size=block.size if block else size,
        )

    def free(self, handle: SlotHandle) -> bool:
        return self._allocator.free(handle.offset)

    def write_tensor(
        self,
        handle: SlotHandle,
        tensor: torch.Tensor,
        byte_offset: int = 0,
        stream: torch.Stream | None = None,
    ) -> int:
        """Copy a tensor into the pool slot. Returns bytes written."""
        src_tensor = tensor.contiguous()
        nbytes = src_tensor.numel() * src_tensor.element_size()

        if byte_offset + nbytes > handle.size:
            raise ValueError(
                f"Write exceeds slot: offset={byte_offset}, nbytes={nbytes}, "
                f"slot_size={handle.size}"
            )

        dst = self._pool[
            handle.offset + byte_offset : handle.offset + byte_offset + nbytes
        ]
        src_bytes = src_tensor.view(torch.uint8).reshape(-1)

        if stream is not None:
            with torch.get_device_module().stream(stream):
                dst.copy_(src_bytes, non_blocking=True)
        else:
            dst.copy_(src_bytes, non_blocking=True)

        return nbytes

    def read_tensor(
        self,
        handle: SlotHandle,
        shape: list[int],
        dtype: torch.dtype,
        byte_offset: int = 0,
        device: torch.device | str = "cpu",
        stream: torch.Stream | None = None,
    ) -> torch.Tensor:
        """Read a tensor from the pool slot. Returns a clone on target device."""
        nbytes = 1
        for s in shape:
            nbytes *= s
        nbytes *= torch.tensor([], dtype=dtype).element_size()

        if byte_offset < 0 or byte_offset + nbytes > handle.size:
            raise ValueError(
                f"Read exceeds slot: offset={byte_offset}, nbytes={nbytes}, "
                f"slot_size={handle.size}"
            )

        raw = self._pool[
            handle.offset + byte_offset : handle.offset + byte_offset + nbytes
        ]
        src = raw.view(dtype).reshape(shape)

        pool_dev = str(self._pool.device)
        target_dev = str(device)

        same_device = pool_dev == target_dev

        if same_device:
            # Clone to decouple tensor lifetime from pool slot
            if stream is not None:
                with torch.get_device_module().stream(stream):
                    return src.clone()
            return src.clone()

        if stream is not None:
            with torch.get_device_module().stream(stream):
                return src.to(device, non_blocking=True)
        return src.to(device, non_blocking=True)

    def write_tensors_from_gpu(
        self,
        handle: SlotHandle,
        tensors: dict[str, torch.Tensor | list[torch.Tensor] | None],
        stream: torch.Stream | None = None,
    ) -> dict[str, list[dict]]:
        """Batch-write GPU tensors into a slot. Returns a manifest for later reads."""
        manifest: dict[str, list[dict]] = {}
        byte_offset = 0

        # Ensure copy stream sees all prior compute kernels
        if stream is not None:
            stream.wait_stream(torch.get_device_module().current_stream())

        for name, value in tensors.items():
            if value is None:
                continue

            entries = []
            if isinstance(value, torch.Tensor):
                nbytes = self.write_tensor(handle, value, byte_offset, stream)
                entries.append(
                    {
                        "offset": byte_offset,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype).replace("torch.", ""),
                    }
                )
                byte_offset += nbytes
                byte_offset = (byte_offset + 511) & ~511  # align to 512B

            elif isinstance(value, list):
                for i, t in enumerate(value):
                    if t is None:
                        continue
                    nbytes = self.write_tensor(handle, t, byte_offset, stream)
                    entries.append(
                        {
                            "offset": byte_offset,
                            "shape": list(t.shape),
                            "dtype": str(t.dtype).replace("torch.", ""),
                            "list_index": i,
                        }
                    )
                    byte_offset += nbytes
                    byte_offset = (byte_offset + 511) & ~511

            if entries:
                manifest[name] = entries

        return manifest

    def read_tensors_from_manifest(
        self,
        handle: SlotHandle,
        manifest: dict[str, list[dict]],
        device: torch.device | str = "cpu",
        stream: torch.Stream | None = None,
        payload_size: int | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Batch-read tensors from a slot using a manifest."""
        if payload_size is not None:
            validate_tensor_manifest(
                manifest, payload_size=payload_size, slot_size=handle.size
            )
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {}

        for name, entries in manifest.items():
            if not entries:
                continue
            has_list_index = any("list_index" in e for e in entries)

            if has_list_index:
                max_idx = max(e.get("list_index", 0) for e in entries) + 1
                tensors = [None] * max_idx
                for entry in entries:
                    t = self.read_tensor(
                        handle,
                        entry["shape"],
                        str_to_dtype(entry["dtype"]),
                        entry["offset"],
                        device,
                        stream,
                    )
                    tensors[entry["list_index"]] = t
                result[name] = tensors
            else:
                entry = entries[0]
                result[name] = self.read_tensor(
                    handle,
                    entry["shape"],
                    str_to_dtype(entry["dtype"]),
                    entry["offset"],
                    device,
                    stream,
                )

        return result
