# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-backed `projection_ut_3dgs_fused` (forward).

Supports the global-shutter UT path with three camera models:
  * pinhole (cm_id=0)
  * OpenCV-pinhole + radial/tangential/thin-prism distortion (cm_id=1)
  * OpenCV-fisheye + 4 radial coefficients (cm_id=2)

Rolling shutter and ftheta still fall back to the torch reference at the
host wrapper layer (`fully_fused_projection_with_ut`).

`fully_fused_projection_with_ut` is documented as non-differentiable, so we
ship only a forward kernel here.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ._kernels import _load


def _projection_ut_3dgs_fused_fwd_metal(
    means: Tensor,  # [B, N, 3]
    quats: Tensor,  # [B, N, 4]
    scales: Tensor,  # [B, N, 3]
    opacities: Optional[Tensor],  # [B, N] or None
    viewmats: Tensor,  # [B, C, 4, 4]
    Ks: Tensor,  # [B, C, 3, 3]
    image_width: int,
    image_height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    ut_alpha: float,
    ut_beta: float,
    ut_kappa: float,
    margin_factor: float,
    calc_compensations: bool,
    cm_id: int = 0,
    radial_coeffs: Optional[Tensor] = None,    # [B, C, 6] (or [B, C, 4] for fisheye)
    tangential_coeffs: Optional[Tensor] = None, # [B, C, 2]
    thin_prism_coeffs: Optional[Tensor] = None, # [B, C, 4]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Returns (radii, means2d, depths, conics, compensations or None)."""
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
        op_flat = torch.zeros(1, device=device, dtype=torch.float32)
        has_opacities = 0
    else:
        op_flat = opacities.contiguous().reshape(B * N).to(torch.float32)
        has_opacities = 1

    # Pad / zero-fill distortion buffers so the kernel can index them
    # unconditionally (cm_id=0 reads but ignores).
    if radial_coeffs is not None:
        # Accept both [B, C, 4] (fisheye) and [B, C, 6] (pinhole-distorted).
        rad_full = torch.zeros(B, C, 6, device=device, dtype=torch.float32)
        rad_full[..., : radial_coeffs.shape[-1]] = radial_coeffs.to(torch.float32)
        rad_flat = rad_full.contiguous().reshape(B * C, 6)
    else:
        rad_flat = torch.zeros(B * C, 6, device=device, dtype=torch.float32)
    if tangential_coeffs is not None:
        tan_flat = tangential_coeffs.contiguous().reshape(B * C, 2).to(torch.float32)
    else:
        tan_flat = torch.zeros(B * C, 2, device=device, dtype=torch.float32)
    if thin_prism_coeffs is not None:
        prism_flat = thin_prism_coeffs.contiguous().reshape(B * C, 4).to(torch.float32)
    else:
        prism_flat = torch.zeros(B * C, 4, device=device, dtype=torch.float32)

    total = B * C * N
    radii = torch.zeros(total, 2, device=device, dtype=torch.int32)
    means2d = torch.zeros(total, 2, device=device, dtype=torch.float32)
    depths = torch.zeros(total, device=device, dtype=torch.float32)
    conics = torch.zeros(total, 3, device=device, dtype=torch.float32)
    if calc_compensations:
        compensations = torch.zeros(total, device=device, dtype=torch.float32)
        comp_buf = compensations
        has_comp = 1
    else:
        compensations = None
        # Pass a 1-element sentinel; not read when has_compensations==0.
        comp_buf = torch.zeros(1, device=device, dtype=torch.float32)
        has_comp = 0

    lib = _load("projection_ut_3dgs_fused_fwd")
    lib.projection_ut_3dgs_fused_fwd(
        radii,
        means2d,
        depths,
        conics,
        comp_buf,
        means_flat,
        quats_flat,
        scales_flat,
        op_flat,
        viewmats_flat,
        Ks_flat,
        B,
        C,
        N,
        image_width,
        image_height,
        float(eps2d),
        float(near_plane),
        float(far_plane),
        float(radius_clip),
        float(ut_alpha),
        float(ut_beta),
        float(ut_kappa),
        float(margin_factor),
        has_opacities,
        has_comp,
        int(cm_id),
        rad_flat,
        tan_flat,
        prism_flat,
        threads=(total,),
        group_size=(min(total, 256),),
    )
    return (
        radii.reshape(B, C, N, 2),
        means2d.reshape(B, C, N, 2),
        depths.reshape(B, C, N),
        conics.reshape(B, C, N, 3),
        compensations.reshape(B, C, N) if compensations is not None else None,
    )


__all__ = ["_projection_ut_3dgs_fused_fwd_metal"]
