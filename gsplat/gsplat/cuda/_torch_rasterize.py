# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch rasterizer fallback used by the MPS port.

The CUDA-vintage `_torch_impl._rasterize_to_pixels` depended on nerfacc and on
the `rasterize_to_indices_3dgs` CUDA op for index generation; neither runs on
MPS. This module ships a self-contained tile-based rasterizer that uses only
stock PyTorch ops, so it works on MPS, CPU, and CUDA. It is much slower than
the CUDA fused kernels — Phase B replaces the hot path with a Metal kernel.
"""

from typing import Optional, Tuple

import math
import torch
from torch import Tensor

from ._constants import MAX_ALPHA


def _rasterize_to_pixels_torch(
    means2d: Tensor,  # [..., N, 2]
    conics: Tensor,  # [..., N, 3] (upper-triangle of inverse 2D covariance)
    colors: Tensor,  # [..., N, C]
    opacities: Tensor,  # [..., N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_h, tile_w]  int32
    flatten_ids: Tensor,  # [n_isects]  int32, sorted front-to-back per tile
    backgrounds: Optional[Tensor] = None,  # [..., C]
    masks: Optional[Tensor] = None,  # [..., tile_h, tile_w]  bool
) -> Tuple[Tensor, Tensor]:
    """Tile-based alpha compositing of 2D Gaussians.

    Iterates over (image, tile) pairs in Python and vectorizes the inner
    Gaussian × pixel work with stock PyTorch ops. Front-to-back order is taken
    from `flatten_ids`, which the caller sorts via `isect_tiles(sort=True)`.
    """
    image_dims = means2d.shape[:-2]
    N, C_channels = means2d.shape[-2], colors.shape[-1]
    tile_h, tile_w = isect_offsets.shape[-2:]

    I = math.prod(image_dims) if image_dims else 1
    means2d_f = means2d.reshape(I, N, 2)
    conics_f = conics.reshape(I, N, 3)
    colors_f = colors.reshape(I, N, C_channels)
    opacities_f = opacities.reshape(I, N)
    isect_offsets_f = isect_offsets.reshape(I, tile_h, tile_w)
    masks_f = masks.reshape(I, tile_h, tile_w) if masks is not None else None

    n_isects = int(flatten_ids.numel())
    device = means2d.device
    dtype = means2d.dtype

    render_colors = torch.zeros(I, image_height, image_width, C_channels, device=device, dtype=dtype)
    render_alphas = torch.zeros(I, image_height, image_width, 1, device=device, dtype=dtype)

    # Per-tile pixel grids (cached once; reused across tiles).
    py_full = torch.arange(image_height, device=device, dtype=dtype) + 0.5
    px_full = torch.arange(image_width, device=device, dtype=dtype) + 0.5

    # `flatten_ids` indexes into a flat (image * N) buffer; the per-image row
    # is recoverable as flatten_ids % N.
    flatten_ids_long = flatten_ids.to(torch.long)

    for img_idx in range(I):
        offs = isect_offsets_f[img_idx]  # [tile_h, tile_w]
        # Build a sentinel offset = n_isects for the (last+1) tile to derive ranges.
        flat_offs = offs.reshape(-1)
        next_offs = torch.cat([flat_offs[1:], torch.tensor([n_isects], device=device, dtype=flat_offs.dtype)])
        starts = flat_offs.tolist()
        ends = next_offs.tolist()

        for tile_lin_idx in range(tile_h * tile_w):
            start = int(starts[tile_lin_idx])
            end = int(ends[tile_lin_idx])
            if end <= start:
                continue
            ty, tx = divmod(tile_lin_idx, tile_w)
            if masks_f is not None and not bool(masks_f[img_idx, ty, tx]):
                continue
            y0, y1 = ty * tile_size, min((ty + 1) * tile_size, image_height)
            x0, x1 = tx * tile_size, min((tx + 1) * tile_size, image_width)
            if y0 >= y1 or x0 >= x1:
                continue

            ph, pw = y1 - y0, x1 - x0
            # Pixel grid for this tile.
            py = py_full[y0:y1]
            px = px_full[x0:x1]
            grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
            pix = torch.stack([grid_x, grid_y], dim=-1)  # [ph, pw, 2]

            # Gaussian indices touching this tile (sorted front-to-back).
            gs_idx = flatten_ids_long[start:end] % N  # [k]
            mu = means2d_f[img_idx, gs_idx]  # [k, 2]
            cn = conics_f[img_idx, gs_idx]  # [k, 3]
            op = opacities_f[img_idx, gs_idx]  # [k]
            col = colors_f[img_idx, gs_idx]  # [k, C]

            # Δ = pixel - μ : [k, ph, pw, 2]
            delta = pix.unsqueeze(0) - mu[:, None, None, :]
            sigma = 0.5 * (
                cn[:, 0, None, None] * delta[..., 0] ** 2
                + cn[:, 2, None, None] * delta[..., 1] ** 2
            ) + cn[:, 1, None, None] * delta[..., 0] * delta[..., 1]  # [k, ph, pw]
            alpha = (op[:, None, None] * torch.exp(-sigma)).clamp(max=MAX_ALPHA)
            alpha = torch.where(sigma < 0, torch.zeros_like(alpha), alpha)

            # Front-to-back compositing along the leading "k" axis.
            # T_i = ∏_{j<i} (1 - α_j)  →  use cumulative product of (1 - α).
            one_minus = (1.0 - alpha).clamp(min=1e-7)
            # Pre-shift so trans[0] = 1.
            ones = torch.ones_like(one_minus[:1])
            trans = torch.cat([ones, one_minus[:-1]], dim=0).cumprod(dim=0)  # [k, ph, pw]
            weights = alpha * trans  # [k, ph, pw]

            tile_color = (weights.unsqueeze(-1) * col[:, None, None, :]).sum(dim=0)  # [ph, pw, C]
            tile_alpha = weights.sum(dim=0, keepdim=True).permute(1, 2, 0)  # [ph, pw, 1]

            render_colors[img_idx, y0:y1, x0:x1] = tile_color
            render_alphas[img_idx, y0:y1, x0:x1] = tile_alpha

    if backgrounds is not None:
        bg = backgrounds.reshape(I, 1, 1, C_channels)
        render_colors = render_colors + bg * (1.0 - render_alphas)

    out_shape_color = image_dims + (image_height, image_width, C_channels)
    out_shape_alpha = image_dims + (image_height, image_width, 1)
    return render_colors.reshape(out_shape_color), render_alphas.reshape(out_shape_alpha)


def _rasterize_to_pixels_2dgs_torch(
    means2d: Tensor,  # [..., N, 2]
    ray_transforms: Tensor,  # [..., N, 3, 3]
    colors: Tensor,  # [..., N, C]
    opacities: Tensor,  # [..., N]
    normals: Tensor,  # [..., N, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_h, tile_w]  int32
    flatten_ids: Tensor,  # [n_isects]  int32
    backgrounds: Optional[Tensor] = None,  # [..., C]
    masks: Optional[Tensor] = None,  # [..., tile_h, tile_w]  bool
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Tile-based 2DGS rasterizer (colors / alphas / normals; distort & median = 0).

    Mirrors `_rasterize_to_pixels_torch` but uses the 2DGS ray-transform-based
    alpha formulation from `accumulate_2dgs`. The distortion and median-depth
    outputs are returned as zeros — they are only consumed by 2DGS-specific
    surface-reconstruction losses, which are out of scope until a real Metal
    kernel lands in Stage 11.
    """
    image_dims = means2d.shape[:-2]
    N, C_channels = means2d.shape[-2], colors.shape[-1]
    tile_h, tile_w = isect_offsets.shape[-2:]

    I = math.prod(image_dims) if image_dims else 1
    means2d_f = means2d.reshape(I, N, 2)
    rt_f = ray_transforms.reshape(I, N, 3, 3)
    colors_f = colors.reshape(I, N, C_channels)
    opacities_f = opacities.reshape(I, N)
    normals_f = normals.reshape(I, N, 3)
    isect_offsets_f = isect_offsets.reshape(I, tile_h, tile_w)
    masks_f = masks.reshape(I, tile_h, tile_w) if masks is not None else None

    n_isects = int(flatten_ids.numel())
    device = means2d.device
    dtype = means2d.dtype

    render_colors = torch.zeros(I, image_height, image_width, C_channels, device=device, dtype=dtype)
    render_alphas = torch.zeros(I, image_height, image_width, 1, device=device, dtype=dtype)
    render_normals = torch.zeros(I, image_height, image_width, 3, device=device, dtype=dtype)

    py_full = torch.arange(image_height, device=device, dtype=dtype) + 0.5
    px_full = torch.arange(image_width, device=device, dtype=dtype) + 0.5

    flatten_ids_long = flatten_ids.to(torch.long)

    for img_idx in range(I):
        flat_offs = isect_offsets_f[img_idx].reshape(-1)
        next_offs = torch.cat([flat_offs[1:], torch.tensor([n_isects], device=device, dtype=flat_offs.dtype)])
        starts = flat_offs.tolist()
        ends = next_offs.tolist()

        for tile_lin_idx in range(tile_h * tile_w):
            start = int(starts[tile_lin_idx])
            end = int(ends[tile_lin_idx])
            if end <= start:
                continue
            ty, tx = divmod(tile_lin_idx, tile_w)
            if masks_f is not None and not bool(masks_f[img_idx, ty, tx]):
                continue
            y0, y1 = ty * tile_size, min((ty + 1) * tile_size, image_height)
            x0, x1 = tx * tile_size, min((tx + 1) * tile_size, image_width)
            if y0 >= y1 or x0 >= x1:
                continue

            py = py_full[y0:y1]  # [ph]
            px = px_full[x0:x1]  # [pw]
            grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")  # [ph, pw]

            gs_idx = flatten_ids_long[start:end] % N  # [k]
            mu = means2d_f[img_idx, gs_idx]  # [k, 2]
            M = rt_f[img_idx, gs_idx]  # [k, 3, 3]
            op = opacities_f[img_idx, gs_idx]  # [k]
            col = colors_f[img_idx, gs_idx]  # [k, C]
            nrm = normals_f[img_idx, gs_idx]  # [k, 3]

            # 3D ray-splat intersection (eq. 9 in 2DGS paper).
            # h_u = -M[:, 0, :] + M[:, 2, :] * px ; h_v = -M[:, 1, :] + M[:, 2, :] * py
            M0 = M[:, 0, :]  # [k, 3]
            M1 = M[:, 1, :]  # [k, 3]
            M2 = M[:, 2, :]  # [k, 3]
            # broadcast: [k, ph, pw, 3]
            px_bc = grid_x[None, ..., None]  # [1, ph, pw, 1]
            py_bc = grid_y[None, ..., None]
            h_u = -M0[:, None, None, :] + M2[:, None, None, :] * px_bc
            h_v = -M1[:, None, None, :] + M2[:, None, None, :] * py_bc
            tmp = torch.cross(h_u, h_v, dim=-1)  # [k, ph, pw, 3]
            denom = torch.where(tmp[..., 2].abs() > 1e-12, tmp[..., 2], torch.ones_like(tmp[..., 2]))
            us = tmp[..., 0] / denom
            vs = tmp[..., 1] / denom
            sigmas_3d = us ** 2 + vs ** 2
            # 2D fallback: 2 * |delta|^2
            delta = torch.stack([grid_x, grid_y], dim=-1)[None] - mu[:, None, None, :]
            sigmas_2d = 2.0 * (delta[..., 0] ** 2 + delta[..., 1] ** 2)
            sigmas = 0.5 * torch.minimum(sigmas_3d, sigmas_2d)

            alpha = (op[:, None, None] * torch.exp(-sigmas)).clamp(max=MAX_ALPHA)
            alpha = torch.where(sigmas < 0, torch.zeros_like(alpha), alpha)

            one_minus = (1.0 - alpha).clamp(min=1e-7)
            ones = torch.ones_like(one_minus[:1])
            trans = torch.cat([ones, one_minus[:-1]], dim=0).cumprod(dim=0)
            weights = alpha * trans

            tile_color = (weights.unsqueeze(-1) * col[:, None, None, :]).sum(dim=0)
            tile_alpha = weights.sum(dim=0, keepdim=True).permute(1, 2, 0)
            tile_normal = (weights.unsqueeze(-1) * nrm[:, None, None, :]).sum(dim=0)

            render_colors[img_idx, y0:y1, x0:x1] = tile_color
            render_alphas[img_idx, y0:y1, x0:x1] = tile_alpha
            render_normals[img_idx, y0:y1, x0:x1] = tile_normal

    if backgrounds is not None:
        bg = backgrounds.reshape(I, 1, 1, C_channels)
        render_colors = render_colors + bg * (1.0 - render_alphas)

    out_color = render_colors.reshape(image_dims + (image_height, image_width, C_channels))
    out_alpha = render_alphas.reshape(image_dims + (image_height, image_width, 1))
    out_normal = render_normals.reshape(image_dims + (image_height, image_width, 3))
    # distort & median: zeros for now (only used by 2DGS surface losses).
    zero_dm_shape = image_dims + (image_height, image_width, 1)
    out_distort = torch.zeros(zero_dm_shape, device=device, dtype=dtype)
    out_median = torch.zeros(zero_dm_shape, device=device, dtype=dtype)
    return out_color, out_alpha, out_normal, out_distort, out_median


__all__ = ["_rasterize_to_pixels_torch", "_rasterize_to_pixels_2dgs_torch"]
