# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-backed `projection_2dgs_fused` (Stage 11, partial).

Forward kernel only — backward routes through the existing torch-reference
autograd path. The 2DGS rasterizer kernels and the 3DGUT eval3d path remain
torch-only for now; see plan file for the full Stage 11 scope.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ._kernels import _load


def _projection_2dgs_fused_fwd_metal(
    means: Tensor,  # [B, N, 3]
    quats: Tensor,  # [B, N, 4]
    scales: Tensor,  # [B, N, 3]
    viewmats: Tensor,  # [B, C, 4, 4]
    Ks: Tensor,  # [B, C, 3, 3]
    image_width: int,
    image_height: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Returns `(radii, means2d, depths, ray_transforms, normals)`."""
    assert means.device.type == "mps"
    B, N = means.shape[0], means.shape[1]
    C = viewmats.shape[1]
    device = means.device

    means_flat = means.contiguous().reshape(B * N, 3)
    quats_flat = quats.contiguous().reshape(B * N, 4)
    scales_flat = scales.contiguous().reshape(B * N, 3)
    viewmats_flat = viewmats.contiguous().reshape(B * C, 16)
    Ks_flat = Ks.contiguous().reshape(B * C, 9)

    total = B * C * N
    radii = torch.zeros(total, 2, device=device, dtype=torch.int32)
    means2d = torch.zeros(total, 2, device=device, dtype=torch.float32)
    depths = torch.zeros(total, device=device, dtype=torch.float32)
    ray_transforms = torch.zeros(total, 3, 3, device=device, dtype=torch.float32)
    normals = torch.zeros(total, 3, device=device, dtype=torch.float32)

    lib = _load("projection_2dgs_fused_fwd")
    lib.projection_2dgs_fused_fwd(
        radii,
        means2d,
        depths,
        ray_transforms,
        normals,
        means_flat,
        quats_flat,
        scales_flat,
        viewmats_flat,
        Ks_flat,
        B,
        C,
        N,
        image_width,
        image_height,
        float(near_plane),
        float(far_plane),
        float(radius_clip),
        threads=(total,),
        group_size=(min(total, 256),),
    )
    return (
        radii.reshape(B, C, N, 2),
        means2d.reshape(B, C, N, 2),
        depths.reshape(B, C, N),
        ray_transforms.reshape(B, C, N, 3, 3),
        normals.reshape(B, C, N, 3),
    )


class _FullyFusedProjection2DGSMetal(torch.autograd.Function):
    """Metal forward + torch-reference backward (autograd)."""

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
        near_plane: float,
        far_plane: float,
        radius_clip: float,
    ):
        radii, means2d, depths, M, normals = _projection_2dgs_fused_fwd_metal(
            means, quats, scales, viewmats, Ks,
            image_width, image_height, near_plane, far_plane, radius_clip,
        )
        ctx.save_for_backward(means, quats, scales, viewmats, Ks)
        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        return radii, means2d, depths, M, normals

    @staticmethod
    def backward(ctx, v_radii, v_m2d, v_depths, v_M, v_normals):
        from ..cuda._torch_impl_2dgs import _fully_fused_projection_2dgs

        means, quats, scales, viewmats, Ks = ctx.saved_tensors
        m = means.detach().requires_grad_(True)
        q = quats.detach().requires_grad_(True)
        s = scales.detach().requires_grad_(True)
        with torch.enable_grad():
            radii, m2d, d, M, n = _fully_fused_projection_2dgs(
                m, q, s, viewmats, Ks, ctx.image_width, ctx.image_height,
                ctx.near_plane, ctx.far_plane,
            )
            loss = torch.zeros((), device=m.device, dtype=m2d.dtype)
            if v_m2d is not None: loss = loss + (m2d * v_m2d).sum()
            if v_depths is not None: loss = loss + (d * v_depths).sum()
            if v_M is not None: loss = loss + (M * v_M).sum()
            if v_normals is not None: loss = loss + (n * v_normals).sum()
        gm, gq, gs = torch.autograd.grad(loss, [m, q, s], allow_unused=True)
        return gm, gq, gs, None, None, None, None, None, None, None


def fully_fused_projection_2dgs_metal(
    means, quats, scales, viewmats, Ks,
    image_width, image_height, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
):
    return _FullyFusedProjection2DGSMetal.apply(
        means, quats, scales, viewmats, Ks,
        int(image_width), int(image_height),
        float(near_plane), float(far_plane), float(radius_clip),
    )


__all__ = [
    "_projection_2dgs_fused_fwd_metal",
    "fully_fused_projection_2dgs_metal",
]
