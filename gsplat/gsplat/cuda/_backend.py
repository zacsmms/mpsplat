# SPDX-FileCopyrightText: Copyright 2023-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-detection shims after the CUDA backend was removed.

The historical `_C` symbol is preserved as `None` so that any remaining
`from ._backend import _C` imports continue to resolve. Stage 1 of the MPS
port replaces every dispatch site that consulted `_C`; the remaining helpers
here (`_sync`, `_empty_cache`) provide a device-agnostic wrapper around the
synchronization / cache calls that used to be CUDA-specific.
"""

import warnings

import torch

_C = None  # legacy alias; the CUDA extension no longer exists

if not torch.backends.mps.is_available():
    warnings.warn(
        "mpsplat: MPS device not available; falling back to CPU via PyTorch."
    )


def _sync(device=None) -> None:
    """Block until pending kernels on `device` have finished."""
    if device is None:
        return
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cache(device=None) -> None:
    """Release cached allocator memory back to the OS for `device`."""
    if device is None:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = ["_C", "_sync", "_empty_cache"]
