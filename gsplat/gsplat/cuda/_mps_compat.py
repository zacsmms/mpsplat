# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MPS op-coverage workarounds.

PyTorch's MPS backend has gaps: no fp64, occasional missing ops, slightly
different semantics for some scatter/gather variants. This module collects
the small helpers that paper over those gaps so the call sites in `_wrapper.py`
stay readable.
"""

from typing import Callable, TypeVar

import torch
from torch import Tensor

T = TypeVar("T")


def is_mps(t: Tensor) -> bool:
    return t.device.type == "mps"


def to_supported_dtype(t: Tensor) -> Tensor:
    """Cast fp64 tensors to fp32 on MPS (MPS has no fp64 support)."""
    if is_mps(t) and t.dtype == torch.float64:
        return t.to(torch.float32)
    return t


def cpu_roundtrip(fn: Callable[..., T], *args, **kwargs) -> T:
    """Run `fn` on CPU when an op isn't yet MPS-supported.

    Tensor inputs are moved to CPU; the result (Tensor or tuple-of-Tensors)
    is moved back to the original device. Non-tensor args pass through.
    """
    target_device = None
    for a in args:
        if isinstance(a, Tensor):
            target_device = a.device
            break

    def _move(x, dev):
        if isinstance(x, Tensor):
            return x.to(dev)
        return x

    cpu_args = tuple(_move(a, "cpu") for a in args)
    cpu_kwargs = {k: _move(v, "cpu") for k, v in kwargs.items()}
    out = fn(*cpu_args, **cpu_kwargs)
    if target_device is None:
        return out
    if isinstance(out, Tensor):
        return out.to(target_device)
    if isinstance(out, tuple):
        return tuple(_move(o, target_device) for o in out)  # type: ignore[return-value]
    if isinstance(out, list):
        return [_move(o, target_device) for o in out]  # type: ignore[return-value]
    return out


__all__ = ["cpu_roundtrip", "is_mps", "to_supported_dtype"]
