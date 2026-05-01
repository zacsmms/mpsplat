# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal compute kernels for the MPS port (Phase B).

This package owns the native-Metal hot paths. Each `.metal` file in
`_shaders/` declares one or more `kernel void` entry points; `_kernels.py`
loads them via `torch.mps.compile_shader` and exposes thin Python wrappers
that the dispatcher in `gsplat/cuda/_dispatch.py` routes to when the input
tensor lives on MPS and the op has been registered through
`register_metal_kernel(...)`.

Kernels are JIT-compiled on first use and cached per process via
`functools.lru_cache`.
"""

import torch

_HAS_COMPILE_SHADER = hasattr(getattr(torch, "mps", None), "compile_shader")


def is_metal_available() -> bool:
    """True iff this PyTorch build supports `torch.mps.compile_shader`."""
    return _HAS_COMPILE_SHADER and torch.backends.mps.is_available()


# Register native-Metal-backed ops in the dispatch table once on import. The
# wrapper in `gsplat/cuda/_wrapper.py` consults this table per call, so all
# 3DGS rasterization on MPS hits the Metal kernel automatically.
if is_metal_available():
    from ..cuda._dispatch import register_metal_kernel

    register_metal_kernel("rasterize_to_pixels_3dgs_fwd")
    register_metal_kernel("intersect_tile")
    register_metal_kernel("projection_ewa_3dgs_fused")
    register_metal_kernel("projection_ewa_3dgs_fused_bwd")
    register_metal_kernel("projection_ut_3dgs_fused_fwd")
    register_metal_kernel("rasterize_to_pixels_eval3d_fwd")
    register_metal_kernel("rasterize_to_pixels_eval3d_bwd")
    register_metal_kernel("spherical_harmonics")
    register_metal_kernel("spherical_harmonics_bwd")
    register_metal_kernel("projection_2dgs_fused")
    register_metal_kernel("rasterize_to_pixels_2dgs_fwd")
    register_metal_kernel("rasterize_to_pixels_2dgs_bwd")


__all__ = ["is_metal_available"]
