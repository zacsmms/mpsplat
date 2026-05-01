# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-backed 3DGUT eval3d rasterizer (forward + backward).

Pinhole + global-shutter + no-distortion + CDIM=3 fast path. Other paths
fall back to the torch reference at the host wrapper layer.

Both fwd and bwd are native Metal kernels. The bwd uses the same atomic-
scatter pattern as the 3DGS rasterizer bwd (Stage 7), with the world-space
ray-Gaussian gradient chain (cross-product VJP, L2-normalize VJP, quat-to-
rotmat VJP, etc.). This eliminates the runtime nerfacc dependency that the
torch reference required.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ._kernels import _load


_TILE_SIZE = 16


def _build_radial_buf(I: int, device, radial_coeffs: Optional[Tensor]) -> Tensor:
    """Pad-or-zero a [I, 4] radial buffer for the kernel signature."""
    if radial_coeffs is None:
        return torch.zeros(I, 4, device=device, dtype=torch.float32)
    rc = radial_coeffs.contiguous().reshape(I, -1).to(torch.float32)
    if rc.shape[-1] >= 4:
        return rc[:, :4].contiguous()
    out = torch.zeros(I, 4, device=device, dtype=torch.float32)
    out[:, : rc.shape[-1]] = rc
    return out


def _rasterize_to_pixels_eval3d_fwd_metal(
    means: Tensor,  # [B, N, 3]    shared across cameras
    quats: Tensor,  # [B, N, 4]
    scales: Tensor,  # [B, N, 3]
    colors: Tensor,  # [I, N, 3]   per-image (I = B * C)
    opacities: Tensor,  # [I, N]
    viewmats: Tensor,  # [I, 4, 4]
    Ks: Tensor,  # [I, 3, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: Tensor,  # [I, tile_h, tile_w]
    flatten_ids: Tensor,  # [n_isects]
    C: int,
    cm_id: int = 0,
    radial_coeffs: Optional["Tensor"] = None,  # [I, 4] for fisheye
    newton_iters: int = 20,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns (render_colors, render_alphas, last_ids)."""
    assert tile_size == _TILE_SIZE
    assert colors.shape[-1] == 3
    B, N = means.shape[0], means.shape[1]
    I = colors.shape[0]
    assert I == B * C
    H, W = image_height, image_width
    tile_h, tile_w = tile_offsets.shape[-2:]

    means_flat = means.contiguous().reshape(B * N, 3)
    quats_flat = quats.contiguous().reshape(B * N, 4)
    scales_flat = scales.contiguous().reshape(B * N, 3)
    colors_flat = colors.contiguous().reshape(I * N, 3)
    op_flat = opacities.contiguous().reshape(I * N)
    viewmats_flat = viewmats.contiguous().reshape(I, 16)
    Ks_flat = Ks.contiguous().reshape(I, 9)
    tile_offsets_i32 = tile_offsets.to(torch.int32).contiguous()
    flatten_ids_i32 = flatten_ids.to(torch.int32).contiguous()

    device = means.device
    render_colors = torch.zeros(I, H, W, 3, device=device, dtype=torch.float32)
    render_alphas = torch.zeros(I, H, W, 1, device=device, dtype=torch.float32)
    last_ids = torch.zeros(I, H, W, device=device, dtype=torch.int32)

    n_isects = int(flatten_ids.numel())
    rad_buf = _build_radial_buf(I, device, radial_coeffs)
    lib = _load("rasterize_to_pixels_eval3d_fwd")
    lib.rasterize_to_pixels_eval3d_fwd(
        render_colors,
        render_alphas,
        last_ids,
        means_flat,
        quats_flat,
        scales_flat,
        colors_flat,
        op_flat,
        viewmats_flat,
        Ks_flat,
        tile_offsets_i32,
        flatten_ids_i32,
        W,
        H,
        tile_w,
        tile_h,
        n_isects,
        I,
        N,
        C,
        int(cm_id),
        rad_buf,
        int(newton_iters),
        threads=(tile_w * _TILE_SIZE, tile_h * _TILE_SIZE, I),
        group_size=(_TILE_SIZE, _TILE_SIZE, 1),
    )
    return render_colors, render_alphas, last_ids


def _rasterize_to_pixels_eval3d_bwd_metal(
    means: Tensor,  # [B, N, 3]
    quats: Tensor,  # [B, N, 4]
    scales: Tensor,  # [B, N, 3]
    colors: Tensor,  # [I, N, 3]
    opacities: Tensor,  # [I, N]
    viewmats: Tensor,  # [I, 4, 4]
    Ks: Tensor,  # [I, 3, 3]
    tile_offsets: Tensor,  # [I, tile_h, tile_w]
    flatten_ids: Tensor,
    render_alphas: Tensor,  # [I, H, W, 1]
    last_ids: Tensor,  # [I, H, W]
    v_render_colors: Tensor,  # [I, H, W, 3]
    v_render_alphas: Tensor,  # [I, H, W, 1]
    image_width: int,
    image_height: int,
    tile_size: int,
    C: int,
    cm_id: int = 0,
    radial_coeffs: Optional[Tensor] = None,
    newton_iters: int = 20,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Returns (v_means, v_quats, v_scales, v_colors, v_opacities)."""
    assert tile_size == _TILE_SIZE
    assert colors.shape[-1] == 3
    B, N = means.shape[0], means.shape[1]
    I = colors.shape[0]
    assert I == B * C
    H, W = image_height, image_width
    tile_h, tile_w = tile_offsets.shape[-2:]
    device = means.device

    means_flat = means.contiguous().reshape(B * N, 3)
    quats_flat = quats.contiguous().reshape(B * N, 4)
    scales_flat = scales.contiguous().reshape(B * N, 3)
    colors_flat = colors.contiguous().reshape(I * N, 3)
    op_flat = opacities.contiguous().reshape(I * N)
    viewmats_flat = viewmats.contiguous().reshape(I, 16)
    Ks_flat = Ks.contiguous().reshape(I, 9)
    tile_offsets_i32 = tile_offsets.to(torch.int32).contiguous()
    flatten_ids_i32 = flatten_ids.to(torch.int32).contiguous()
    render_alphas_flat = render_alphas.contiguous().reshape(I * H * W)
    last_ids_flat = last_ids.to(torch.int32).contiguous().reshape(I * H * W)
    v_render_colors_flat = v_render_colors.contiguous().reshape(I * H * W * 3)
    v_render_alphas_flat = v_render_alphas.contiguous().reshape(I * H * W)

    v_means_out = torch.zeros(B * N, 3, device=device, dtype=torch.float32)
    v_quats_out = torch.zeros(B * N, 4, device=device, dtype=torch.float32)
    v_scales_out = torch.zeros(B * N, 3, device=device, dtype=torch.float32)
    v_colors_out = torch.zeros(I * N, 3, device=device, dtype=torch.float32)
    v_opacities_out = torch.zeros(I * N, device=device, dtype=torch.float32)

    n_isects = int(flatten_ids.numel())
    rad_buf = _build_radial_buf(I, device, radial_coeffs)
    lib = _load("rasterize_to_pixels_eval3d_bwd")
    lib.rasterize_to_pixels_eval3d_bwd(
        v_means_out,
        v_quats_out,
        v_scales_out,
        v_colors_out,
        v_opacities_out,
        means_flat,
        quats_flat,
        scales_flat,
        colors_flat,
        op_flat,
        viewmats_flat,
        Ks_flat,
        render_alphas_flat,
        last_ids_flat,
        v_render_colors_flat,
        v_render_alphas_flat,
        tile_offsets_i32,
        flatten_ids_i32,
        W,
        H,
        tile_w,
        tile_h,
        n_isects,
        I,
        N,
        C,
        int(cm_id),
        rad_buf,
        int(newton_iters),
        threads=(tile_w * _TILE_SIZE, tile_h * _TILE_SIZE, I),
        group_size=(_TILE_SIZE, _TILE_SIZE, 1),
    )
    return (
        v_means_out.reshape(B, N, 3),
        v_quats_out.reshape(B, N, 4),
        v_scales_out.reshape(B, N, 3),
        v_colors_out.reshape(I, N, 3),
        v_opacities_out.reshape(I, N),
    )


class _RasterizeToPixelsEval3dMetal(torch.autograd.Function):
    """Metal forward + Metal backward for 3DGUT eval3d (pinhole + CDIM=3)."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        colors: Tensor,
        opacities: Tensor,
        viewmats: Tensor,
        Ks: Tensor,
        image_width: int,
        image_height: int,
        tile_size: int,
        tile_offsets: Tensor,
        flatten_ids: Tensor,
        C: int,
        cm_id: int,
        radial_coeffs: Optional[Tensor],
        newton_iters: int,
    ) -> Tuple[Tensor, Tensor]:
        rc, ra, last_ids = _rasterize_to_pixels_eval3d_fwd_metal(
            means, quats, scales, colors, opacities, viewmats, Ks,
            image_width, image_height, tile_size,
            tile_offsets, flatten_ids, C,
            cm_id=cm_id, radial_coeffs=radial_coeffs, newton_iters=newton_iters,
        )
        ctx.save_for_backward(
            means, quats, scales, colors, opacities, viewmats, Ks,
            tile_offsets, flatten_ids, ra, last_ids,
            radial_coeffs if radial_coeffs is not None else torch.empty(0),
        )
        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.tile_size = tile_size
        ctx.C = C
        ctx.cm_id = cm_id
        ctx.has_radial = radial_coeffs is not None
        ctx.newton_iters = newton_iters
        return rc, ra

    @staticmethod
    def backward(ctx, grad_colors: Tensor, grad_alphas: Tensor):
        (
            means, quats, scales, colors, opacities, viewmats, Ks,
            tile_offsets, flatten_ids, render_alphas, last_ids, radial_buf,
        ) = ctx.saved_tensors
        rad = radial_buf if ctx.has_radial else None
        v_m, v_q, v_s, v_col, v_op = _rasterize_to_pixels_eval3d_bwd_metal(
            means, quats, scales, colors, opacities, viewmats, Ks,
            tile_offsets, flatten_ids,
            render_alphas, last_ids, grad_colors, grad_alphas,
            ctx.image_width, ctx.image_height, ctx.tile_size, ctx.C,
            cm_id=ctx.cm_id, radial_coeffs=rad, newton_iters=ctx.newton_iters,
        )
        return (
            v_m, v_q, v_s, v_col, v_op,
            None,  # viewmats
            None,  # Ks
            None,  # image_width
            None,  # image_height
            None,  # tile_size
            None,  # tile_offsets
            None,  # flatten_ids
            None,  # C
            None,  # cm_id
            None,  # radial_coeffs
            None,  # newton_iters
        )


def rasterize_to_pixels_eval3d_metal(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    colors: Tensor,
    opacities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: Tensor,
    flatten_ids: Tensor,
    C: int,
    cm_id: int = 0,
    radial_coeffs: Optional[Tensor] = None,
    newton_iters: int = 20,
) -> Tuple[Tensor, Tensor]:
    return _RasterizeToPixelsEval3dMetal.apply(
        means, quats, scales, colors, opacities, viewmats, Ks,
        image_width, image_height, tile_size, tile_offsets, flatten_ids, C,
        cm_id, radial_coeffs, newton_iters,
    )


__all__ = ["rasterize_to_pixels_eval3d_metal", "_rasterize_to_pixels_eval3d_fwd_metal"]
