# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selection for the MPS port.

The library has two execution paths: a Metal-backed kernel (registered in
Phase B as kernels are written) and a pure-PyTorch reference path that runs
on MPS or CPU via the existing `_torch_impl*` modules. `select_backend` is
called by every public op in `_wrapper.py` to pick between them.

When a Metal kernel is registered for a given op, MPS tensors flow through
that path; everything else flows through the torch reference.
"""

from enum import Enum
from typing import Optional

import torch
from torch import Tensor


class Backend(Enum):
    METAL = "metal"
    TORCH = "torch"


_METAL_KERNELS_AVAILABLE: set[str] = set()


def register_metal_kernel(op_name: str) -> None:
    """Mark an op as having a native Metal kernel. Phase B helper."""
    _METAL_KERNELS_AVAILABLE.add(op_name)


def has_metal(op_name: str) -> bool:
    return op_name in _METAL_KERNELS_AVAILABLE


def select_backend(t: Tensor, op_name: Optional[str] = None) -> Backend:
    """Pick the backend for an op call based on the input tensor.

    Args:
        t: A representative input tensor (used to read the device).
        op_name: Optional op name. If a Metal kernel is registered for this
            op and `t` lives on MPS, returns METAL. Otherwise TORCH.
    """
    if op_name is not None and t.device.type == "mps" and has_metal(op_name):
        return Backend.METAL
    return Backend.TORCH


def auto_device() -> str:
    """Pick the best available device string for example scripts."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


__all__ = [
    "Backend",
    "auto_device",
    "has_metal",
    "register_metal_kernel",
    "select_backend",
]
