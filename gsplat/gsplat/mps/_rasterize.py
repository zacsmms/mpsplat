# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-backed 3DGS forward rasterizer + torch-fallback backward.

Stage 6 lights up the forward path only. The backward saves the inputs
needed for `_torch_impl._rasterize_to_pixels_torch` and reruns the rasterizer
under autograd to compute gradients. Stage 7 replaces the backward with a
native Metal kernel.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ._kernels import _load


_TILE_SIZE = 16  # must match TILE_SIZE in rasterize_to_pixels_3dgs_fwd.metal


def _rasterize_to_pixels_3dgs_fwd_metal(
    means2d: Tensor,  # [I, N, 2]  fp32, mps
    conics: Tensor,  # [I, N, 3]
    colors: Tensor,  # [I, N, 3] (CDIM=3 only for now)
    opacities: Tensor,  # [I, N]
    backgrounds: Optional[Tensor],  # [I, 3] or None
    masks: Optional[Tensor],  # [I, tile_h, tile_w] bool or None
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: Tensor,  # [I, tile_h, tile_w] int32
    flatten_ids: Tensor,  # [n_isects] int32
) -> Tuple[Tensor, Tensor, Tensor]:
    """Launch the Metal forward kernel. Returns (colors, alphas, last_ids).

    All inputs must be MPS tensors. The kernel is hard-coded to `tile_size=16`
    and `CDIM=3`; callers fall back to the torch path otherwise.
    """
    assert tile_size == _TILE_SIZE, f"only tile_size={_TILE_SIZE} is supported"
    assert colors.shape[-1] == 3, "Stage-6 Metal kernel hard-codes CDIM=3"
    assert means2d.dim() == 3, "packed mode not supported in Metal forward yet"

    I, N = means2d.shape[0], means2d.shape[1]
    H, W = image_height, image_width
    tile_h, tile_w = tile_offsets.shape[-2:]
    assert tile_offsets.shape == (I, tile_h, tile_w)

    if backgrounds is not None or (masks is not None and not masks.all()):
        # Background blending and tile masks aren't wired into the kernel yet;
        # caller should fall back to the torch rasterizer for these.
        raise NotImplementedError(
            "Metal forward kernel does not handle backgrounds / masks yet"
        )

    # The kernel reads `means2d`, `conics`, `colors`, `opacities` as flat
    # [I*N]-length buffers indexed by `flatten_ids`. The torch wrapper passes
    # them as [I, N, ...] which is contiguous in (I, N) row-major, matching
    # what we need.
    means2d_flat = means2d.contiguous().reshape(I * N, 2)
    conics_flat = conics.contiguous().reshape(I * N, 3)
    colors_flat = colors.contiguous().reshape(I * N, 3)
    opacities_flat = opacities.contiguous().reshape(I * N)

    tile_offsets_i32 = tile_offsets.to(torch.int32).contiguous()
    flatten_ids_i32 = flatten_ids.to(torch.int32).contiguous()

    render_colors = torch.zeros(I, H, W, 3, device=means2d.device, dtype=torch.float32)
    render_alphas = torch.zeros(I, H, W, 1, device=means2d.device, dtype=torch.float32)
    last_ids = torch.zeros(I, H, W, device=means2d.device, dtype=torch.int32)

    n_isects = int(flatten_ids.numel())
    lib = _load("rasterize_to_pixels_3dgs_fwd")
    lib.rasterize_to_pixels_3dgs_fwd(
        render_colors,
        render_alphas,
        last_ids,
        means2d_flat,
        conics_flat,
        colors_flat,
        opacities_flat,
        tile_offsets_i32,
        flatten_ids_i32,
        W,
        H,
        tile_w,
        tile_h,
        n_isects,
        I,
        threads=(tile_w * _TILE_SIZE, tile_h * _TILE_SIZE, I),
        group_size=(_TILE_SIZE, _TILE_SIZE, 1),
    )
    return render_colors, render_alphas, last_ids


def _rasterize_to_pixels_3dgs_bwd_metal(
    means2d: Tensor,  # [I, N, 2]
    conics: Tensor,  # [I, N, 3]
    colors: Tensor,  # [I, N, 3]
    opacities: Tensor,  # [I, N]
    tile_offsets: Tensor,  # [I, tile_h, tile_w]
    flatten_ids: Tensor,  # [n_isects]
    render_alphas: Tensor,  # [I, H, W, 1]
    last_ids: Tensor,  # [I, H, W]
    v_render_colors: Tensor,  # [I, H, W, 3]
    v_render_alphas: Tensor,  # [I, H, W, 1]
    image_width: int,
    image_height: int,
    tile_size: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Launch the Metal backward kernel.

    Returns dense `[I, N, ...]` gradients for `means2d`, `conics`, `colors`,
    and `opacities`. The kernel uses `atomic<float>` adds, so the buffers are
    zero-initialized then accumulated in place.
    """
    assert tile_size == _TILE_SIZE
    assert colors.shape[-1] == 3
    I, N = means2d.shape[0], means2d.shape[1]
    H, W = image_height, image_width
    tile_h, tile_w = tile_offsets.shape[-2:]

    means2d_flat = means2d.contiguous().reshape(I * N, 2)
    conics_flat = conics.contiguous().reshape(I * N, 3)
    colors_flat = colors.contiguous().reshape(I * N, 3)
    opacities_flat = opacities.contiguous().reshape(I * N)
    tile_offsets_i32 = tile_offsets.to(torch.int32).contiguous()
    flatten_ids_i32 = flatten_ids.to(torch.int32).contiguous()
    render_alphas_flat = render_alphas.contiguous().reshape(I * H * W)
    last_ids_flat = last_ids.to(torch.int32).contiguous().reshape(I * H * W)
    v_render_colors_flat = v_render_colors.contiguous().reshape(I * H * W * 3)
    v_render_alphas_flat = v_render_alphas.contiguous().reshape(I * H * W)

    v_means2d = torch.zeros(I * N, 2, device=means2d.device, dtype=torch.float32)
    v_conics = torch.zeros(I * N, 3, device=means2d.device, dtype=torch.float32)
    v_colors = torch.zeros(I * N, 3, device=means2d.device, dtype=torch.float32)
    v_opacities = torch.zeros(I * N, device=means2d.device, dtype=torch.float32)

    n_isects = int(flatten_ids.numel())
    lib = _load("rasterize_to_pixels_3dgs_bwd")
    lib.rasterize_to_pixels_3dgs_bwd(
        v_means2d,
        v_conics,
        v_colors,
        v_opacities,
        means2d_flat,
        conics_flat,
        colors_flat,
        opacities_flat,
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
        threads=(tile_w * _TILE_SIZE, tile_h * _TILE_SIZE, I),
        group_size=(_TILE_SIZE, _TILE_SIZE, 1),
    )

    return (
        v_means2d.reshape(I, N, 2),
        v_conics.reshape(I, N, 3),
        v_colors.reshape(I, N, 3),
        v_opacities.reshape(I, N),
    )


class _RasterizeToPixels3DGSMetal(torch.autograd.Function):
    """3DGS rasterizer with Metal forward + torch reference backward.

    `forward` saves the inputs and calls the Metal kernel. `backward` reruns
    the rasterizer under autograd via the torch reference path
    (`_torch_rasterize._rasterize_to_pixels_torch`) and returns its gradients.

    Stage 7 will replace `backward` with a native Metal bwd kernel.
    """

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        conics: Tensor,
        colors: Tensor,
        opacities: Tensor,
        image_width: int,
        image_height: int,
        tile_size: int,
        tile_offsets: Tensor,
        flatten_ids: Tensor,
        backgrounds: Optional[Tensor],
        masks: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        render_colors, render_alphas, last_ids = _rasterize_to_pixels_3dgs_fwd_metal(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            image_width,
            image_height,
            tile_size,
            tile_offsets,
            flatten_ids,
        )
        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            tile_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.tile_size = tile_size
        ctx.has_backgrounds = backgrounds is not None
        ctx.has_masks = masks is not None
        return render_colors, render_alphas

    @staticmethod
    def backward(ctx, grad_colors: Tensor, grad_alphas: Tensor):
        (
            means2d,
            conics,
            colors,
            opacities,
            tile_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        v_means2d, v_conics, v_colors, v_opacities = (
            _rasterize_to_pixels_3dgs_bwd_metal(
                means2d,
                conics,
                colors,
                opacities,
                tile_offsets,
                flatten_ids,
                render_alphas,
                last_ids,
                grad_colors,
                grad_alphas,
                ctx.image_width,
                ctx.image_height,
                ctx.tile_size,
            )
        )
        # forward signature has 11 inputs; return None for the non-tensor ones
        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            None,  # image_width
            None,  # image_height
            None,  # tile_size
            None,  # tile_offsets
            None,  # flatten_ids
            None,  # backgrounds
            None,  # masks
        )


def rasterize_to_pixels_3dgs_metal(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Public-facing entry point used by `_wrapper.rasterize_to_pixels`."""
    return _RasterizeToPixels3DGSMetal.apply(
        means2d,
        conics,
        colors,
        opacities,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
        backgrounds,
        masks,
    )


__all__ = ["rasterize_to_pixels_3dgs_metal"]
