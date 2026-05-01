# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-backed `projection_ewa_3dgs_fused` (forward + backward).

Both fwd and bwd are native Metal kernels. The backward chains through
quat→rotmat→covar3D→covar2D→conics with the FOV-clamped projection
Jacobian, all sym matrices tracked full (off-diag mirrored).

Hard-coded constraints (drop back to the torch path if any are violated):
  * camera_model == "pinhole"
  * covars is None (must use quats + scales)
  * calc_compensations is False
  * opacities is None or [B, N] float32
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ._kernels import _load


def _projection_ewa_3dgs_fused_fwd_metal(
    means: Tensor,  # [B, N, 3]
    quats: Tensor,  # [B, N, 4]
    scales: Tensor,  # [B, N, 3]
    viewmats: Tensor,  # [B, C, 4, 4]
    Ks: Tensor,  # [B, C, 3, 3]
    image_width: int,
    image_height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    opacities: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns (radii [B,C,N,2], means2d [B,C,N,2], depths [B,C,N], conics [B,C,N,3])."""
    assert means.device.type == "mps"
    B, N = means.shape[0], means.shape[1]
    C = viewmats.shape[1]
    device = means.device

    means_flat = means.contiguous().reshape(B * N, 3)
    quats_flat = quats.contiguous().reshape(B * N, 4)
    scales_flat = scales.contiguous().reshape(B * N, 3)
    viewmats_flat = viewmats.contiguous().reshape(B * C, 16)
    Ks_flat = Ks.contiguous().reshape(B * C, 9)

    if opacities is None:
        # Pass a 1-element sentinel buffer to satisfy the kernel signature;
        # `has_opacities=0` ensures it is never read.
        op_flat = torch.zeros(1, device=device, dtype=torch.float32)
        has_opacities = 0
    else:
        op_flat = opacities.contiguous().reshape(B * N).to(torch.float32)
        has_opacities = 1

    total = B * C * N
    radii = torch.zeros(B * C * N, 2, device=device, dtype=torch.int32)
    means2d = torch.zeros(B * C * N, 2, device=device, dtype=torch.float32)
    depths = torch.zeros(B * C * N, device=device, dtype=torch.float32)
    conics = torch.zeros(B * C * N, 3, device=device, dtype=torch.float32)

    lib = _load("projection_ewa_3dgs_fused_fwd")
    lib.projection_ewa_3dgs_fused_fwd(
        radii,
        means2d,
        depths,
        conics,
        means_flat,
        quats_flat,
        scales_flat,
        viewmats_flat,
        Ks_flat,
        op_flat,
        B,
        C,
        N,
        image_width,
        image_height,
        float(eps2d),
        float(near_plane),
        float(far_plane),
        float(radius_clip),
        has_opacities,
        threads=(total,),
        group_size=(min(total, 256),),
    )

    return (
        radii.reshape(B, C, N, 2),
        means2d.reshape(B, C, N, 2),
        depths.reshape(B, C, N),
        conics.reshape(B, C, N, 3),
    )


def _projection_ewa_3dgs_fused_bwd_metal(
    means: Tensor,  # [B, N, 3]
    quats: Tensor,  # [B, N, 4]
    scales: Tensor,  # [B, N, 3]
    viewmats: Tensor,  # [B, C, 4, 4]
    Ks: Tensor,  # [B, C, 3, 3]
    radii: Tensor,  # [B, C, N, 2]
    conics: Tensor,  # [B, C, N, 3]
    v_means2d: Tensor,  # [B, C, N, 2]
    v_depths: Tensor,  # [B, C, N]
    v_conics: Tensor,  # [B, C, N, 3]
    image_width: int,
    image_height: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns (v_means [B,N,3], v_quats [B,N,4], v_scales [B,N,3])."""
    assert means.device.type == "mps"
    B, N = means.shape[0], means.shape[1]
    C = viewmats.shape[1]
    device = means.device

    means_flat = means.contiguous().reshape(B * N, 3)
    quats_flat = quats.contiguous().reshape(B * N, 4)
    scales_flat = scales.contiguous().reshape(B * N, 3)
    viewmats_flat = viewmats.contiguous().reshape(B * C, 16)
    Ks_flat = Ks.contiguous().reshape(B * C, 9)
    radii_flat = radii.to(torch.int32).contiguous().reshape(B * C * N, 2)
    conics_flat = conics.contiguous().reshape(B * C * N, 3)
    v_means2d_flat = v_means2d.contiguous().reshape(B * C * N, 2)
    v_depths_flat = v_depths.contiguous().reshape(B * C * N)
    v_conics_flat = v_conics.contiguous().reshape(B * C * N, 3)

    v_means_out = torch.zeros(B * N, 3, device=device, dtype=torch.float32)
    v_quats_out = torch.zeros(B * N, 4, device=device, dtype=torch.float32)
    v_scales_out = torch.zeros(B * N, 3, device=device, dtype=torch.float32)

    total = B * C * N
    lib = _load("projection_ewa_3dgs_fused_bwd")
    lib.projection_ewa_3dgs_fused_bwd(
        v_means_out,
        v_quats_out,
        v_scales_out,
        means_flat,
        quats_flat,
        scales_flat,
        viewmats_flat,
        Ks_flat,
        radii_flat,
        conics_flat,
        v_means2d_flat,
        v_depths_flat,
        v_conics_flat,
        B,
        C,
        N,
        image_width,
        image_height,
        threads=(total,),
        group_size=(min(total, 256),),
    )
    return (
        v_means_out.reshape(B, N, 3),
        v_quats_out.reshape(B, N, 4),
        v_scales_out.reshape(B, N, 3),
    )


class _FullyFusedProjection3DGSMetal(torch.autograd.Function):
    """Metal forward + Metal backward for `fully_fused_projection`."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        viewmats: Tensor,
        Ks: Tensor,
        image_width: int,
        image_height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        opacities: Optional[Tensor],
    ):
        radii, means2d, depths, conics = _projection_ewa_3dgs_fused_fwd_metal(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            opacities,
        )
        ctx.save_for_backward(means, quats, scales, viewmats, Ks, radii, conics)
        ctx.image_width = image_width
        ctx.image_height = image_height
        return radii, means2d, depths, conics, None

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations):
        means, quats, scales, viewmats, Ks, radii, conics = ctx.saved_tensors
        # The kernel reads upstream grads unconditionally; pass zero buffers
        # for any output that didn't receive a grad.
        if v_means2d is None:
            v_means2d = torch.zeros_like(conics[..., :2])
        if v_depths is None:
            v_depths = torch.zeros(*radii.shape[:-1], device=radii.device, dtype=torch.float32)
        if v_conics is None:
            v_conics = torch.zeros_like(conics)
        v_means, v_quats, v_scales = _projection_ewa_3dgs_fused_bwd_metal(
            means, quats, scales, viewmats, Ks,
            radii, conics,
            v_means2d, v_depths, v_conics,
            ctx.image_width, ctx.image_height,
        )
        return (
            v_means,
            v_quats,
            v_scales,
            None,  # viewmats
            None,  # Ks
            None,  # image_width
            None,  # image_height
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # radius_clip
            None,  # opacities
        )


def fully_fused_projection_3dgs_metal(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    image_width: int,
    image_height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    opacities: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Public-facing entry point for the Metal projection."""
    return _FullyFusedProjection3DGSMetal.apply(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        float(eps2d),
        float(near_plane),
        float(far_plane),
        float(radius_clip),
        opacities,
    )


__all__ = [
    "_projection_ewa_3dgs_fused_fwd_metal",
    "fully_fused_projection_3dgs_metal",
]
