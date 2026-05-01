# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-backed spherical harmonics (forward + backward).

Both fwd and bwd are native Metal kernels. Polynomial basis gradients are
hardcoded inline in `spherical_harmonics_bwd.metal`; the L2-normalize VJP
is applied at the end of the bwd kernel.
"""

from typing import Optional

import torch
from torch import Tensor

from ._kernels import _load


def _spherical_harmonics_fwd_metal(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    coeffs: Tensor,  # [..., K, 3]
) -> Tensor:
    """Returns SH-evaluated colors of shape [..., 3]."""
    assert dirs.device.type == "mps"
    K = coeffs.shape[-2]
    assert (degrees_to_use + 1) ** 2 <= K
    assert K <= 25, "Stage 10 Metal SH supports up to degree 4 (K<=25)"

    batch_dims = dirs.shape[:-1]
    M = 1
    for d in batch_dims:
        M *= int(d)
    dirs_flat = dirs.contiguous().reshape(M, 3)
    coeffs_flat = coeffs.contiguous().reshape(M, K, 3).reshape(M, K * 3)
    out_flat = torch.zeros(M, 3, device=dirs.device, dtype=torch.float32)

    lib = _load("spherical_harmonics_fwd")
    lib.spherical_harmonics_fwd(
        out_flat,
        dirs_flat,
        coeffs_flat,
        M,
        K,
        int(degrees_to_use),
        threads=(M,),
        group_size=(min(M, 256),),
    )
    return out_flat.reshape(batch_dims + (3,))


def _spherical_harmonics_bwd_metal(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    coeffs: Tensor,  # [..., K, 3]
    v_out: Tensor,  # [..., 3]
) -> tuple[Tensor, Tensor]:
    """Returns (v_dirs [..., 3], v_coeffs [..., K, 3])."""
    assert dirs.device.type == "mps"
    K = coeffs.shape[-2]
    assert (degrees_to_use + 1) ** 2 <= K
    assert K <= 25

    batch_dims = dirs.shape[:-1]
    M = 1
    for d in batch_dims:
        M *= int(d)
    dirs_flat = dirs.contiguous().reshape(M, 3)
    coeffs_flat = coeffs.contiguous().reshape(M, K * 3)
    v_out_flat = v_out.contiguous().reshape(M, 3)
    v_dirs = torch.zeros(M, 3, device=dirs.device, dtype=torch.float32)
    v_coeffs = torch.zeros(M, K * 3, device=dirs.device, dtype=torch.float32)

    lib = _load("spherical_harmonics_bwd")
    lib.spherical_harmonics_bwd(
        v_dirs,
        v_coeffs,
        dirs_flat,
        coeffs_flat,
        v_out_flat,
        M,
        K,
        int(degrees_to_use),
        threads=(M,),
        group_size=(min(M, 256),),
    )
    return (
        v_dirs.reshape(batch_dims + (3,)),
        v_coeffs.reshape(batch_dims + (K, 3)),
    )


class _SphericalHarmonicsMetal(torch.autograd.Function):
    """Metal forward + Metal backward."""

    @staticmethod
    def forward(
        ctx,
        degrees_to_use: int,
        dirs: Tensor,
        coeffs: Tensor,
        masks: Optional[Tensor],
    ) -> Tensor:
        out = _spherical_harmonics_fwd_metal(degrees_to_use, dirs, coeffs)
        if masks is not None:
            out = torch.where(masks[..., None].to(torch.bool), out, torch.zeros_like(out))
        ctx.save_for_backward(dirs, coeffs, masks if masks is not None else torch.empty(0))
        ctx.degrees_to_use = degrees_to_use
        ctx.has_masks = masks is not None
        return out

    @staticmethod
    def backward(ctx, v_out: Tensor):
        dirs, coeffs, masks = ctx.saved_tensors
        if ctx.has_masks:
            v_out = torch.where(masks[..., None].to(torch.bool), v_out, torch.zeros_like(v_out))
        v_dirs, v_coeffs = _spherical_harmonics_bwd_metal(
            ctx.degrees_to_use, dirs, coeffs, v_out,
        )
        return None, v_dirs, v_coeffs, None  # degrees_to_use, dirs, coeffs, masks


def spherical_harmonics_metal(
    degrees_to_use: int,
    dirs: Tensor,
    coeffs: Tensor,
    masks: Optional[Tensor] = None,
) -> Tensor:
    return _SphericalHarmonicsMetal.apply(degrees_to_use, dirs, coeffs, masks)


__all__ = ["_spherical_harmonics_fwd_metal", "spherical_harmonics_metal"]
