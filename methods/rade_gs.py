"""
RaDe-GS — Rasterizing Depth in Gaussian Splatting (Zhang et al., 2026).

Drop-in companion to ``gsplat.rasterization``: produces per-pixel depth and
surface-normal maps for a 3D Gaussian scene by rasterizing the spatially
varying depth derived from each splat's ray-space Jacobian, instead of the
flat per-splat depth used in vanilla 3DGS.

Backend selection is automatic. ``gsplat.rasterization`` dispatches on the
device of the input tensors, so this module runs unchanged on:

  * MPS  — native Metal rasterizer (``torch.mps.compile_shader``).
  * CUDA — upstream gsplat CUDA kernels (when the wheel is built with CUDA).
  * CPU  — torch reference implementation.

Math (all per-Gaussian, all done with batched torch ops):

    Σ_cam = R · Σ_world · Rᵀ                       (world → camera)
    J     = ∂(u, v, t) / ∂(x, y, z) |_{x_cam}        (3×3 ray-space Jacobian)
    Σ'    = J · Σ_cam · Jᵀ                          (ray-space covariance)
    p̂    = (v'ᵀ Σ'⁻¹) / (v'ᵀ Σ'⁻¹ v'),  v' = (0,0,1)
    p     = p̂[:2]                                   (depth-plane vector, 1×2)
    n'    = −(p_x, p_y, 1)ᵀ                          (ray-space plane normal)
    n     = Jᵀ · n', normalized                     (camera-space normal)

For each pixel (u, v) covered by Gaussian i, the paper's Eq. (3) gives the
along-ray distance::

    t*(u, v) = t_c + p · (u_c − u, v_c − v)

which is *affine in (u, v)*. Expanded into pixel-grid form::

    t*(u, v) = A + B · u + C · v
    where  A = t_c + p_x · u_c + p_y · v_c
           B = −p_x
           C = −p_y

Because (A, B, C) are constants per Gaussian, they alpha-blend like any
other feature. We pack them — plus the three components of n — into six
``extra_signals`` channels, let ``gsplat.rasterization`` do the
front-to-back compositing, then reconstruct the depth analytically::

    d(u, v) = cos θ(u, v) · t̄*(u, v)

where t̄* is the alpha-blended affine combination and cos θ is the angle
between the pixel ray and the camera's principal axis (a function of K
only). This is bit-equivalent to a custom rasterizer that evaluates the
spatially varying depth inside the inner tile loop, because alpha blending
is linear in the per-Gaussian feature.
"""

# SPDX-FileCopyrightText: Copyright 2026 Zachary Scott-Murphy
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor

import inspect as _inspect

from gsplat import quat_scale_to_covar_preci, rasterization


# Channel layout for the six extra-signal channels we splat.
_FEAT_A, _FEAT_B, _FEAT_C = 0, 1, 2  # affine depth coefficients
_FEAT_NX, _FEAT_NZ = 3, 5  # camera-space normal slice [NX:NZ+1]

# The ``extra_signals`` parameter exists in the mpsplat fork (and other forks)
# but not in upstream gsplat. When it's missing we fall back to a two-pass
# rasterization in ``rasterization_rade_gs``.
_HAS_EXTRA_SIGNALS = "extra_signals" in _inspect.signature(rasterization).parameters


def _world_covars(
    quats: Optional[Tensor],
    scales: Optional[Tensor],
    covars: Optional[Tensor],
) -> Tensor:
    """Resolve world-space (N, 3, 3) covariances from either parameterization."""
    if covars is not None:
        return covars
    covars, _ = quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False, triu=False
    )
    return covars


def _ray_space_jacobian(
    x_cam: Tensor,  # (..., N, 3) — Gaussian centers in camera space
    Ks: Tensor,  # (..., 3, 3) — broadcastable to (..., 1, 3, 3) per-N
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Local affine Jacobian J = ∂(u, v, t) / ∂(x, y, z) at each Gaussian center.

    Returns:
        J:    (..., N, 3, 3) — the full 3×3 ray-space Jacobian.
        t_c:  (..., N)       — Euclidean distance from camera origin to mean.
        z_c:  (..., N)       — camera-space z of each Gaussian.
    """
    x, y, z = x_cam.unbind(dim=-1)  # each (..., N)
    z_safe = z.clamp(min=eps)
    t_c = torch.sqrt(x * x + y * y + z * z).clamp(min=eps)

    fx = Ks[..., 0, 0].unsqueeze(-1)  # (..., 1)
    fy = Ks[..., 1, 1].unsqueeze(-1)  # (..., 1)

    inv_z = 1.0 / z_safe  # (..., N)
    inv_z2 = inv_z * inv_z
    inv_t = 1.0 / t_c

    zero = torch.zeros_like(z_safe)

    # Row 0: ∂u/∂(x,y,z), Row 1: ∂v/∂(x,y,z), Row 2: ∂t/∂(x,y,z).
    j00 = fx * inv_z
    j02 = -fx * x * inv_z2
    j11 = fy * inv_z
    j12 = -fy * y * inv_z2
    j20 = x * inv_t
    j21 = y * inv_t
    j22 = z * inv_t

    # Stack into (..., N, 3, 3). One big stack is faster than repeated cats on MPS.
    J = torch.stack(
        [j00, zero, j02, zero, j11, j12, j20, j21, j22], dim=-1
    ).reshape(x_cam.shape[:-1] + (3, 3))
    return J, t_c, z_safe


def _depth_vector_p_from_cov(cov_ray: Tensor, eps: float = 1e-8) -> Tensor:
    """Extract the 1×2 depth-plane vector p from ray-space covariance Σ'.

    Closed-form: with v' = (0, 0, 1),

        p̂ = (v'ᵀ Σ'⁻¹) / (v'ᵀ Σ'⁻¹ v')

    The numerator is the third row of Σ'⁻¹; the denominator is the (2, 2)
    element of Σ'⁻¹. Using cofactors of Σ' (symmetric), this works out to::

        p_x = (Σ'[0,1] · Σ'[1,2] − Σ'[0,2] · Σ'[1,1]) / (Σ'[0,0] · Σ'[1,1] − Σ'[0,1]²)
        p_y = (Σ'[0,1] · Σ'[0,2] − Σ'[0,0] · Σ'[1,2]) / (Σ'[0,0] · Σ'[1,1] − Σ'[0,1]²)

    Computing it this way avoids batched 3×3 inversion (``torch.linalg.inv``
    is ~10× slower on MPS than these scalar ops at N ≈ 10⁶, and triggers
    fallback on some pre-MPS-3 builds).
    """
    s00 = cov_ray[..., 0, 0]
    s01 = cov_ray[..., 0, 1]
    s02 = cov_ray[..., 0, 2]
    s11 = cov_ray[..., 1, 1]
    s12 = cov_ray[..., 1, 2]

    denom = (s00 * s11 - s01 * s01).clamp(min=eps)  # cofactor C_22 of Σ'
    p_x = (s01 * s12 - s02 * s11) / denom
    p_y = (s01 * s02 - s00 * s12) / denom
    return torch.stack([p_x, p_y], dim=-1)  # (..., N, 2)


def _camera_space_normal(J: Tensor, p: Tensor, eps: float = 1e-8) -> Tensor:
    """Per-Gaussian unit normal in camera space: n = Jᵀ · n', with n' = −(p_x, p_y, 1)ᵀ."""
    n_ray = torch.cat([-p, -torch.ones_like(p[..., :1])], dim=-1)  # (..., N, 3)
    # J is (..., N, 3, 3); multiply by Jᵀ via einsum so we don't materialize a transpose.
    n = torch.einsum("...nji,...nj->...ni", J, n_ray)  # Jᵀ · n_ray, per Gaussian
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=eps)
    # Flip so the normal faces the camera (camera looks down +z in this codebase's
    # convention, so the side of the splat facing the camera has n_z < 0).
    flip = torch.where(n[..., 2:3] > 0, -torch.ones_like(n[..., :1]), torch.ones_like(n[..., :1]))
    return n * flip


def compute_rade_gs_features(
    means: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    quats: Optional[Tensor] = None,
    scales: Optional[Tensor] = None,
    covars: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Compute the per-Gaussian (A, B, C, n_x, n_y, n_z) feature pack.

    Output shape: ``(C, N, 6)``. The first three channels are the affine
    coefficients of the spatially varying along-ray distance::

        t*(u, v) = A + B · u + C · v

    so that alpha-blending them gives a per-pixel depth that's bit-equivalent
    to a per-pixel evaluation of the paper's Eq. (3). The last three channels
    are the camera-space surface normal, which is per-splat constant and so
    alpha-blends directly.

    Args:
        means:    (N, 3) world-space Gaussian centers.
        viewmats: (C, 4, 4) world-to-camera transforms.
        Ks:       (C, 3, 3) pinhole intrinsics.
        quats / scales: alternative parameterization of world covariance.
        covars:   (N, 3, 3) explicit world-space covariance (overrides quats/scales).
    """
    cov_world = _world_covars(quats, scales, covars)  # (N, 3, 3)

    R = viewmats[..., :3, :3]  # (C, 3, 3)
    t_v = viewmats[..., :3, 3]  # (C, 3)

    # Transform Gaussian means and covariances into each camera's frame.
    x_cam = torch.einsum("cij,nj->cni", R, means) + t_v[:, None, :]  # (C, N, 3)
    cov_cam = torch.einsum(
        "cij,njk,clk->cnil", R, cov_world, R
    )  # (C, N, 3, 3)

    # Ray-space Jacobian (per camera, per Gaussian) and its derived quantities.
    J, t_c, _ = _ray_space_jacobian(x_cam, Ks, eps=eps)  # (C, N, 3, 3), (C, N), (C, N)

    # Σ' = J · Σ_cam · Jᵀ (3×3 ray-space covariance).
    cov_ray = torch.einsum("cnij,cnjk,cnlk->cnil", J, cov_cam, J)  # (C, N, 3, 3)

    # Depth-plane vector p (1×2) and camera-space unit normal.
    p = _depth_vector_p_from_cov(cov_ray, eps=eps)  # (C, N, 2)
    n_cam = _camera_space_normal(J, p, eps=eps)  # (C, N, 3)

    # Projected center (u_c, v_c) in pixel coordinates. We compute this here so
    # callers don't need to redo the projection — and it matches the projection
    # `gsplat.rasterization` uses internally for the actual splat placement.
    fx = Ks[..., 0, 0, None]
    fy = Ks[..., 1, 1, None]
    cx = Ks[..., 0, 2, None]
    cy = Ks[..., 1, 2, None]
    inv_z = 1.0 / x_cam[..., 2].clamp(min=eps)
    u_c = fx * x_cam[..., 0] * inv_z + cx  # (C, N)
    v_c = fy * x_cam[..., 1] * inv_z + cy  # (C, N)

    p_x = p[..., 0]
    p_y = p[..., 1]

    # Affine depth: t*(u, v) = t_c + p · (u_c − u, v_c − v) = A + B u + C v.
    A = t_c + p_x * u_c + p_y * v_c  # (C, N)
    B = -p_x
    C_ = -p_y

    # Pack: (A, B, C, n_x, n_y, n_z) → (C, N, 6).
    return torch.stack([A, B, C_, n_cam[..., 0], n_cam[..., 1], n_cam[..., 2]], dim=-1)


def _build_pixel_grid(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> Tuple[Tensor, Tensor]:
    """Pixel-center grid (u, v), each of shape (H, W). 0.5 offset → pixel centers."""
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    return uu, vv


def _cos_theta_map(
    Ks: Tensor, height: int, width: int, dtype: torch.dtype, eps: float = 1e-8
) -> Tensor:
    """cos θ between each pixel's ray and the principal axis. Shape (C, H, W, 1)."""
    device = Ks.device
    uu, vv = _build_pixel_grid(height, width, device, dtype)  # (H, W), (H, W)

    fx = Ks[..., 0, 0].view(-1, 1, 1)  # (C, 1, 1)
    fy = Ks[..., 1, 1].view(-1, 1, 1)
    cx = Ks[..., 0, 2].view(-1, 1, 1)
    cy = Ks[..., 1, 2].view(-1, 1, 1)

    dx = (uu - cx) / fx  # (C, H, W)
    dy = (vv - cy) / fy
    inv_norm = torch.rsqrt(dx * dx + dy * dy + 1.0 + eps)
    return inv_norm.unsqueeze(-1)  # (C, H, W, 1)


def rasterization_rade_gs(
    means: Tensor,  # (N, 3)
    quats: Tensor,  # (N, 4)
    scales: Tensor,  # (N, 3)
    opacities: Tensor,  # (N,)
    colors: Tensor,  # (N, D) or (N, K, 3) when sh_degree is set
    viewmats: Tensor,  # (C, 4, 4)
    Ks: Tensor,  # (C, 3, 3)
    width: int,
    height: int,
    *,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    radius_clip: float = 0.0,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    tile_size: int = 16,
    normalize_normals: bool = True,
    flip_normals_to_face_camera: bool = True,
    covars: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Render color, depth, and surface-normal maps in the RaDe-GS formulation.

    All input tensors must live on the same device. Pick the backend implicitly
    by choosing that device:

        * ``means.to("mps")`` → native Metal rasterizer.
        * ``means.to("cuda")`` → CUDA rasterizer (if mpsplat was built with it).
        * ``means.to("cpu")``  → torch reference path.

    Returns a dict with:

        * ``render_colors``: (C, H, W, D) — RGB (or N-D features).
        * ``render_alphas``: (C, H, W, 1) — accumulated opacity.
        * ``render_depths``: (C, H, W, 1) — camera-space z-depth, alpha-normalized.
        * ``render_normals``: (C, H, W, 3) — unit-length camera-space normals.
        * ``meta``: the meta dict from ``gsplat.rasterization`` (for debugging).
    """
    assert means.dim() == 2 and means.shape[-1] == 3, means.shape
    assert viewmats.dim() == 3 and viewmats.shape[-2:] == (4, 4), viewmats.shape
    assert Ks.dim() == 3 and Ks.shape[-2:] == (3, 3), Ks.shape

    C = viewmats.shape[0]
    N = means.shape[0]
    dtype = means.dtype

    # 1. Pre-compute the six per-Gaussian features (depth coefs + normal).
    #    Shape: (C, N, 6). The depth coefs depend on the camera, so this can't
    #    collapse to (N, 6) — see the J · Σ · Jᵀ step in compute_rade_gs_features.
    extras = compute_rade_gs_features(
        means=means,
        viewmats=viewmats,
        Ks=Ks,
        quats=quats if covars is None else None,
        scales=scales if covars is None else None,
        covars=covars,
    )  # (C, N, 6)

    # rasterization() concatenates colors and extra_signals along the last axis,
    # so they must agree in rank. extras is per-camera (C, N, 6); broadcast
    # colors to (C, N, ...) to match. SH coeffs (..., N, K, 3) get the same
    # treatment along the leading dims.
    if sh_degree is None:
        if colors.dim() == 2:
            colors = colors.unsqueeze(0).expand(C, N, colors.shape[-1])
    else:
        if colors.dim() == 3:
            colors = colors.unsqueeze(0).expand(C, N, colors.shape[-2], 3)

    # 2. Rasterize RGB + the six extra channels.
    #
    # Forks that expose ``extra_signals=`` (the mpsplat fork, some others) can
    # do this in a single pass — the extras come back as Σᵢ wᵢ · featᵢ on the
    # ``meta`` dict. Upstream gsplat doesn't have that path, so we issue a
    # second rasterization with the per-Gaussian feature pack treated as
    # colors. With a zero background, the standard ``RGB`` render mode is
    # also Σᵢ wᵢ · featᵢ, so the downstream math (dividing by α) is identical.
    common_kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        eps2d=eps2d,
        tile_size=tile_size,
        render_mode="RGB",
        rasterize_mode=rasterize_mode,
        covars=covars,
    )

    if _HAS_EXTRA_SIGNALS:
        render_colors, render_alphas, meta = rasterization(
            colors=colors,
            sh_degree=sh_degree,
            backgrounds=backgrounds,
            extra_signals=extras,
            **common_kwargs,
        )
        # render_colors: (C, H, W, D); meta["render_extra_signals"]: (C, H, W, 6).
        blended = meta["render_extra_signals"]  # alpha-accumulated, NOT divided by α.
    else:
        # Pass 1: RGB rendering with the user's colors/SH and backgrounds.
        render_colors, render_alphas, meta = rasterization(
            colors=colors,
            sh_degree=sh_degree,
            backgrounds=backgrounds,
            **common_kwargs,
        )
        # Pass 2: feature rendering with no background. ``rasterization``
        # returns Σᵢ wᵢ · featᵢ + (1 − Σᵢ wᵢ) · bg; with ``backgrounds=None``
        # gsplat defaults to zero, matching the ``extra_signals`` semantics.
        # (Don't construct a zero tensor ourselves — upstream gsplat versions
        # disagree on the exact shape expected here.)
        blended, _, _ = rasterization(
            colors=extras,
            sh_degree=None,
            backgrounds=None,
            **common_kwargs,
        )

    # 3. Reconstruct the per-pixel along-ray distance t* from the blended affine
    #    coefficients, then collapse to camera-space z-depth via cos θ.
    #
    #    Blended values are Σ_i w_i · feat_i where w_i = α_i · T_i. To get the
    #    expected depth conditional on a hit, divide by the accumulated alpha
    #    Σ_i w_i = render_alphas (clamped to avoid 0/0 on empty pixels).
    alpha = render_alphas.clamp(min=1e-10)  # (C, H, W, 1)
    a_blend = blended[..., _FEAT_A : _FEAT_A + 1] / alpha
    b_blend = blended[..., _FEAT_B : _FEAT_B + 1] / alpha
    c_blend = blended[..., _FEAT_C : _FEAT_C + 1] / alpha

    uu, vv = _build_pixel_grid(height, width, means.device, dtype)
    uu = uu.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1) — broadcasts over C
    vv = vv.unsqueeze(0).unsqueeze(-1)

    t_star = a_blend + b_blend * uu + c_blend * vv  # (C, H, W, 1) Euclidean ray distance
    cos_theta = _cos_theta_map(Ks, height, width, dtype)  # (C, H, W, 1)
    render_depths = cos_theta * t_star  # camera-space z-depth

    # Pixels with no Gaussian coverage get α≈0 → depth is meaningless. Zero them
    # so downstream losses (and visualization) don't pick up garbage.
    render_depths = render_depths * (render_alphas > 1e-6)

    # 4. Recover the surface normal map. Normals were splatted as per-Gaussian
    #    constants, so alpha-blending them is the same as alpha-blending colors;
    #    we just need to renormalize the result to unit length.
    n = blended[..., _FEAT_NX : _FEAT_NZ + 1]  # (C, H, W, 3)
    if normalize_normals:
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    if flip_normals_to_face_camera:
        # n_z > 0 means the normal points away from the camera (in this codebase
        # the camera looks down +z). Flip those splat contributions so the
        # output map is consistently camera-facing.
        flip = torch.where(n[..., 2:3] > 0, -1.0, 1.0)
        n = n * flip

    render_normals = n
    return {
        "render_colors": render_colors,
        "render_alphas": render_alphas,
        "render_depths": render_depths,
        "render_normals": render_normals,
        "meta": meta,
    }


__all__ = [
    "compute_rade_gs_features",
    "rasterization_rade_gs",
]
