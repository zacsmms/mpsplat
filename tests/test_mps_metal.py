# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-5 smoke: torch.mps.compile_shader compiles + dispatches a kernel."""

import pytest
import torch

from gsplat.cuda._dispatch import (
    Backend,
    has_metal,
    register_metal_kernel,
    select_backend,
)
from gsplat.mps import is_metal_available
from gsplat.mps._kernels import _load, elementwise_add


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_elementwise_add_correct():
    a = torch.rand(1024, device="mps")
    b = torch.rand(1024, device="mps")
    out = elementwise_add(a, b)
    assert out.device.type == "mps"
    assert out.shape == a.shape
    assert (out - (a + b)).abs().max().item() < 1e-6


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_shader_compile_is_cached():
    # Calling `_load` twice should hit the lru_cache and return the same
    # library object both times.
    lib1 = _load("elementwise_add")
    lib2 = _load("elementwise_add")
    assert lib1 is lib2


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_rasterize_to_pixels_3dgs_fwd_metal_matches_torch():
    """Stage 6: Metal forward output matches torch reference within 1e-4."""
    from gsplat.cuda._torch_rasterize import _rasterize_to_pixels_torch
    from gsplat.mps._rasterize import _rasterize_to_pixels_3dgs_fwd_metal

    device = "mps"
    N, H, W = 64, 128, 128
    tile_size, tile_h, tile_w = 16, H // 16, W // 16
    g = torch.Generator(device="cpu").manual_seed(0)
    m2d = (torch.rand(1, N, 2, generator=g) * torch.tensor([W, H])).to(device)
    conics = torch.full((1, N, 3), 0.05, device=device)
    conics[..., 1] = 0.0
    colors = torch.rand(1, N, 3, generator=g).to(device)
    opacities = torch.full((1, N), 0.5, device=device)

    # Synthesize a balanced isect: gaussians round-robin over tiles.
    flatten_ids = torch.arange(N, device=device, dtype=torch.int32)
    n_tiles = tile_h * tile_w
    n_per = max(1, N // n_tiles)
    flatten_ids = flatten_ids[: n_per * n_tiles]
    isect_offsets = torch.arange(
        0, n_per * n_tiles, n_per, device=device, dtype=torch.int32
    ).reshape(1, tile_h, tile_w)

    img_t, a_t = _rasterize_to_pixels_torch(
        m2d, conics, colors, opacities, W, H, tile_size, isect_offsets, flatten_ids
    )
    img_m, a_m, _ = _rasterize_to_pixels_3dgs_fwd_metal(
        m2d,
        conics,
        colors,
        opacities,
        None,
        None,
        W,
        H,
        tile_size,
        isect_offsets,
        flatten_ids,
    )
    assert (img_t - img_m).abs().max().item() < 1e-4
    assert (a_t - a_m).abs().max().item() < 1e-4


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_rasterize_to_pixels_3dgs_bwd_metal_matches_torch():
    """Stage 7: Metal backward grads match torch-autograd reference within tol."""
    from gsplat.cuda._torch_rasterize import _rasterize_to_pixels_torch
    from gsplat.mps._rasterize import _RasterizeToPixels3DGSMetal

    device = "mps"
    N, H, W = 32, 64, 64
    tile_size, tile_h, tile_w = 16, H // 16, W // 16
    g = torch.Generator(device="cpu").manual_seed(0)
    m2d = (torch.rand(1, N, 2, generator=g) * torch.tensor([W, H])).to(device)
    conics = torch.full((1, N, 3), 0.05, device=device)
    conics[..., 1] = 0.0
    colors = torch.rand(1, N, 3, generator=g).to(device)
    opacities = torch.full((1, N), 0.5, device=device)

    n_tiles = tile_h * tile_w
    n_per = max(1, N // n_tiles)
    flatten_ids = torch.arange(N, device=device, dtype=torch.int32)[: n_per * n_tiles]
    isect_offsets = torch.arange(
        0, n_per * n_tiles, n_per, device=device, dtype=torch.int32
    ).reshape(1, tile_h, tile_w)

    # Torch reference (autograd through the pure-PyTorch rasterizer).
    m_t = m2d.clone().requires_grad_(True)
    c_t = conics.clone().requires_grad_(True)
    col_t = colors.clone().requires_grad_(True)
    op_t = opacities.clone().requires_grad_(True)
    img_t, a_t = _rasterize_to_pixels_torch(
        m_t, c_t, col_t, op_t, W, H, tile_size, isect_offsets, flatten_ids
    )
    (img_t.sum() + a_t.sum()).backward()

    # Metal version — autograd.Function uses our Stage-7 Metal bwd kernel.
    m_m = m2d.clone().requires_grad_(True)
    c_m = conics.clone().requires_grad_(True)
    col_m = colors.clone().requires_grad_(True)
    op_m = opacities.clone().requires_grad_(True)
    img_m, a_m = _RasterizeToPixels3DGSMetal.apply(
        m_m,
        c_m,
        col_m,
        op_m,
        W,
        H,
        tile_size,
        isect_offsets,
        flatten_ids,
        None,
        None,
    )
    (img_m.sum() + a_m.sum()).backward()

    # Plan tolerance: 5e-4. Conics see slightly more drift due to
    # atomic-reordering (each gauss receives many add-from-pixel ops).
    assert (m_t.grad - m_m.grad).abs().max().item() < 5e-4, "means2d grad"
    assert (c_t.grad - c_m.grad).abs().max().item() < 1e-3, "conics grad"
    assert (col_t.grad - col_m.grad).abs().max().item() < 5e-4, "colors grad"
    assert (op_t.grad - op_m.grad).abs().max().item() < 5e-4, "opacities grad"


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_intersect_tile_metal_matches_torch():
    """Stage 8: Metal `intersect_tile` is bit-exact with the torch reference."""
    from gsplat.cuda._torch_impl import _isect_tiles
    from gsplat.mps._intersect import _intersect_tile_metal

    device = "mps"
    N, H, W = 64, 128, 128
    tile_size, tile_h, tile_w = 16, H // 16, W // 16
    g = torch.Generator(device="cpu").manual_seed(0)
    m2d = (torch.rand(1, N, 2, generator=g) * torch.tensor([W, H])).to(device)
    radii = torch.randint(2, 10, (1, N, 2), generator=g).to(device).int()
    depths = torch.rand(1, N, generator=g).to(device)

    tpg_t, iid_t, fid_t = _isect_tiles(
        m2d, radii, depths, tile_size, tile_w, tile_h, sort=True
    )
    tpg_m, iid_m, fid_m = _intersect_tile_metal(
        m2d, radii, depths, tile_size, tile_w, tile_h, sort=True
    )
    assert torch.equal(tpg_t, tpg_m), "tiles_per_gauss"
    assert torch.equal(iid_t, iid_m), "isect_ids"
    assert torch.equal(fid_t, fid_m), "flatten_ids"


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_projection_ewa_3dgs_fused_fwd_metal_matches_torch():
    """Stage 9: Metal `projection_ewa_3dgs_fused` matches torch reference
    for valid (non-culled) gaussians."""
    from gsplat.cuda._math import _quat_scale_to_covar_preci
    from gsplat.cuda._torch_impl import _fully_fused_projection
    from gsplat.mps._projection import _projection_ewa_3dgs_fused_fwd_metal

    device = "mps"
    B, C, N, H, W = 1, 1, 64, 128, 128
    g = torch.Generator(device="cpu").manual_seed(0)
    means = (torch.randn(B, N, 3, generator=g) * 0.5).to(device)
    raw_q = torch.randn(B, N, 4, generator=g).to(device)
    quats = raw_q / raw_q.norm(dim=-1, keepdim=True)
    scales = (torch.rand(B, N, 3, generator=g) * 0.1 + 0.02).to(device)
    viewmats = torch.eye(4)[None, None].clone().to(device)
    viewmats[..., :3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)
    Ks = torch.tensor(
        [[100.0, 0, 64.0], [0, 100.0, 64.0], [0, 0, 1.0]], device=device
    ).reshape(1, 1, 3, 3)

    covars3, _ = _quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False, triu=False
    )
    radii_t, m2d_t, depths_t, conics_t, _ = _fully_fused_projection(
        means, covars3, viewmats, Ks, W, H, eps2d=0.3, calc_compensations=False
    )
    radii_m, m2d_m, depths_m, conics_m = _projection_ewa_3dgs_fused_fwd_metal(
        means, quats, scales, viewmats, Ks, W, H, 0.3, 0.01, 1e10, 0.0, None
    )

    valid = (radii_m > 0).all(dim=-1)
    # radii bit-exact across the board.
    assert torch.equal(radii_t.int(), radii_m)
    # means2d / depths / conics match within fp32 noise on valid entries.
    assert (m2d_t[valid] - m2d_m[valid]).abs().max().item() < 1e-4
    assert (depths_t[valid] - depths_m[valid]).abs().max().item() < 1e-4
    assert (conics_t[valid] - conics_m[valid]).abs().max().item() < 1e-4


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
@pytest.mark.parametrize("degrees", [0, 1, 2, 3, 4])
def test_spherical_harmonics_metal_matches_torch(degrees):
    """Stage 10: Metal SH evaluation matches torch reference within 1e-5."""
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.mps._sh import _spherical_harmonics_fwd_metal

    K = (degrees + 1) ** 2
    M = 256
    g = torch.Generator(device="cpu").manual_seed(0)
    dirs = torch.randn(M, 3, generator=g).to("mps")
    coeffs = (torch.randn(M, K, 3, generator=g) * 0.3).to("mps")
    out_t = _spherical_harmonics(degrees, dirs, coeffs)
    out_m = _spherical_harmonics_fwd_metal(degrees, dirs, coeffs)
    assert (out_t - out_m).abs().max().item() < 1e-5


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_projection_2dgs_fused_fwd_metal_matches_torch():
    """Stage 11: Metal 2DGS projection matches torch reference for valid gaussians."""
    from gsplat.cuda._torch_impl_2dgs import _fully_fused_projection_2dgs
    from gsplat.mps._projection_2dgs import _projection_2dgs_fused_fwd_metal

    device = "mps"
    B, C, N, H, W = 1, 1, 64, 128, 128
    g = torch.Generator(device="cpu").manual_seed(0)
    means = (torch.randn(B, N, 3, generator=g) * 0.5).to(device)
    raw_q = torch.randn(B, N, 4, generator=g).to(device)
    quats = raw_q / raw_q.norm(dim=-1, keepdim=True)
    scales = (torch.rand(B, N, 3, generator=g) * 0.1 + 0.02).to(device)
    viewmats = torch.eye(4)[None, None].clone().to(device)
    viewmats[..., :3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)
    Ks = torch.tensor(
        [[100.0, 0, 64.0], [0, 100.0, 64.0], [0, 0, 1.0]], device=device
    ).reshape(1, 1, 3, 3)

    radii_t, m2d_t, d_t, M_t, n_t = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, W, H
    )
    radii_m, m2d_m, d_m, M_m, n_m = _projection_2dgs_fused_fwd_metal(
        means, quats, scales, viewmats, Ks, W, H, 0.01, 1e10, 0.0
    )
    valid = (radii_m > 0).all(dim=-1)
    assert torch.equal(radii_t.int(), radii_m)
    assert (m2d_t[valid] - m2d_m[valid]).abs().max().item() < 1e-3
    assert (d_t[valid] - d_m[valid]).abs().max().item() < 1e-4
    assert (M_t[valid] - M_m[valid]).abs().max().item() < 1e-4
    assert (n_t[valid] - n_m[valid]).abs().max().item() < 1e-4


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_rasterize_to_pixels_2dgs_fwd_metal_matches_torch():
    """Deferred-stage 1: Metal 2DGS rasterizer fwd matches torch reference."""
    import math

    from gsplat.cuda._torch_impl_2dgs import _fully_fused_projection_2dgs
    from gsplat.cuda._torch_rasterize import _rasterize_to_pixels_2dgs_torch
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles
    from gsplat.mps._rasterize_2dgs import _rasterize_to_pixels_2dgs_fwd_metal

    device = "mps"
    torch.manual_seed(0)
    B, C, N, H, W = 1, 1, 64, 128, 128
    means = (torch.randn(B, N, 3) * 0.5).to(device)
    raw_q = torch.randn(B, N, 4).to(device)
    quats = raw_q / raw_q.norm(dim=-1, keepdim=True)
    scales = (torch.rand(B, N, 3) * 0.1 + 0.02).to(device)
    viewmats = torch.eye(4)[None, None].clone().to(device)
    viewmats[..., :3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)
    Ks = torch.tensor(
        [[100.0, 0, 64.0], [0, 100.0, 64.0], [0, 0, 1.0]], device=device
    ).reshape(1, 1, 3, 3)
    radii, m2d, depths, M, normals = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, W, H
    )
    colors = torch.rand(B, C, N, 3, device=device)
    opacities = torch.rand(B, C, N, device=device) * 0.5 + 0.3
    I = B * C
    tile_size = 16
    tile_w = math.ceil(W / tile_size)
    tile_h = math.ceil(H / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        m2d, radii, depths, tile_size, tile_w, tile_h
    )
    iso = isect_offset_encode(isect_ids, I, tile_w, tile_h).reshape(I, tile_h, tile_w)

    rc_t, ra_t, rn_t, _, _ = _rasterize_to_pixels_2dgs_torch(
        m2d.reshape(I, N, 2), M.reshape(I, N, 3, 3),
        colors.reshape(I, N, 3), opacities.reshape(I, N), normals.reshape(I, N, 3),
        W, H, tile_size, iso, flatten_ids,
    )
    rc_m, ra_m, rn_m, _ = _rasterize_to_pixels_2dgs_fwd_metal(
        m2d.reshape(I, N, 2), M.reshape(I, N, 3, 3),
        colors.reshape(I, N, 3), opacities.reshape(I, N), normals.reshape(I, N, 3),
        W, H, tile_size, iso, flatten_ids,
    )
    assert (rc_t - rc_m).abs().max().item() < 5e-5
    assert (ra_t - ra_m).abs().max().item() < 5e-5
    assert (rn_t - rn_m).abs().max().item() < 5e-5


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_rasterize_to_pixels_2dgs_bwd_metal_matches_torch():
    """Deferred-stage 2: Metal 2DGS rasterizer bwd matches torch ref autograd."""
    import math

    from gsplat.cuda._torch_impl_2dgs import _fully_fused_projection_2dgs
    from gsplat.cuda._torch_rasterize import _rasterize_to_pixels_2dgs_torch
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles
    from gsplat.mps._rasterize_2dgs import rasterize_to_pixels_2dgs_metal

    device = "mps"
    torch.manual_seed(7)
    B, C, N, H, W = 1, 1, 64, 64, 64
    means = (torch.randn(B, N, 3) * 0.5).to(device)
    raw_q = torch.randn(B, N, 4).to(device)
    quats = (raw_q / raw_q.norm(dim=-1, keepdim=True))
    scales = (torch.rand(B, N, 3) * 0.1 + 0.02).to(device)
    viewmats = torch.eye(4)[None, None].clone().to(device)
    viewmats[..., :3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)
    Ks = torch.tensor(
        [[60.0, 0, 32.0], [0, 60.0, 32.0], [0, 0, 1.0]], device=device
    ).reshape(1, 1, 3, 3)
    radii, m2d, depths, rt, normals = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, W, H
    )
    m2d = m2d.detach().requires_grad_()
    rt = rt.detach().requires_grad_()
    normals = normals.detach().requires_grad_()
    colors = torch.rand(B, C, N, 3, device=device).requires_grad_()
    op = (torch.rand(B, C, N, device=device) * 0.5 + 0.3).requires_grad_()
    I = B * C
    tile_size = 16
    tile_w = math.ceil(W / tile_size)
    tile_h = math.ceil(H / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        m2d.reshape(B, C, N, 2).detach(), radii, depths, tile_size, tile_w, tile_h
    )
    iso = isect_offset_encode(isect_ids, I, tile_w, tile_h).reshape(I, tile_h, tile_w)

    m2d_v = m2d.reshape(I, N, 2)
    rt_v = rt.reshape(I, N, 3, 3)
    col_v = colors.reshape(I, N, 3)
    op_v = op.reshape(I, N)
    nrm_v = normals.reshape(I, N, 3)

    rc_t, ra_t, rn_t, _, _ = _rasterize_to_pixels_2dgs_torch(
        m2d_v, rt_v, col_v, op_v, nrm_v, W, H, tile_size, iso, flatten_ids,
    )
    rc_m, ra_m, rn_m = rasterize_to_pixels_2dgs_metal(
        m2d_v, rt_v, col_v, op_v, nrm_v, W, H, tile_size, iso, flatten_ids,
    )

    g_rc = torch.randn_like(rc_t)
    g_ra = torch.randn_like(ra_t)
    g_rn = torch.randn_like(rn_t)
    leafs = [m2d, rt, colors, op, normals]
    grads_t = torch.autograd.grad(
        (rc_t * g_rc).sum() + (ra_t * g_ra).sum() + (rn_t * g_rn).sum(),
        leafs, retain_graph=True, allow_unused=True,
    )
    grads_m = torch.autograd.grad(
        (rc_m * g_rc).sum() + (ra_m * g_ra).sum() + (rn_m * g_rn).sum(),
        leafs, retain_graph=True, allow_unused=True,
    )
    # Per-leaf relative tolerance — atomic-add ordering + fp32 amplifies
    # absolute error proportionally to leaf magnitude. ray_transforms grads
    # are O(100) for this scene, so we use a 1e-3 relative threshold.
    for name, gt, gm in zip(
        ["means2d", "ray_transforms", "colors", "opacities", "normals"],
        grads_t, grads_m,
    ):
        assert gt is not None and gm is not None, name
        rel = (gt - gm).abs().max().item() / (gt.abs().max().item() + 1e-12)
        assert rel < 1e-3, f"{name}: rel diff {rel:.3e}"


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
@pytest.mark.parametrize("deg", [1, 2, 3, 4])
def test_spherical_harmonics_bwd_metal_matches_torch(deg):
    """Deferred-stage 6: Metal SH bwd matches torch ref autograd."""
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.mps._sh import _spherical_harmonics_bwd_metal

    K = (deg + 1) ** 2
    M = 1024
    g = torch.Generator(device="cpu").manual_seed(deg)
    d_raw = torch.randn(M, 3, generator=g).to("mps")
    coeffs = (torch.randn(M, K, 3, generator=g) * 0.3).to("mps").requires_grad_()
    dirs_t = d_raw.clone().requires_grad_()
    out_t = _spherical_harmonics(deg, dirs_t, coeffs)
    v_out = torch.randn_like(out_t)
    gd_t, gc_t = torch.autograd.grad(
        (out_t * v_out).sum(), [dirs_t, coeffs], retain_graph=False, allow_unused=True,
    )
    if gd_t is None:
        gd_t = torch.zeros_like(d_raw)
    gd_m, gc_m = _spherical_harmonics_bwd_metal(deg, d_raw, coeffs.detach(), v_out)
    assert (gd_t - gd_m).abs().max().item() < 1e-4
    assert (gc_t - gc_m).abs().max().item() < 1e-5


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_projection_ewa_3dgs_fused_bwd_metal_matches_torch():
    """Deferred-stage 5: Metal projection_ewa bwd matches torch ref autograd."""
    from gsplat.cuda._math import _quat_scale_to_covar_preci
    from gsplat.cuda._torch_impl import _fully_fused_projection
    from gsplat.mps._projection import (
        _projection_ewa_3dgs_fused_bwd_metal,
        _projection_ewa_3dgs_fused_fwd_metal,
    )

    device = "mps"
    g = torch.Generator(device="cpu").manual_seed(7)
    B, C, N, H, W = 1, 1, 32, 128, 128
    means = (torch.randn(B, N, 3, generator=g) * 0.5).to(device).requires_grad_()
    raw_q = torch.randn(B, N, 4, generator=g).to(device).requires_grad_()
    scales = (torch.rand(B, N, 3, generator=g) * 0.1 + 0.02).to(device).requires_grad_()
    viewmats = torch.eye(4)[None, None].clone().to(device)
    viewmats[..., :3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)
    Ks = torch.tensor(
        [[100.0, 0, 64.0], [0, 100.0, 64.0], [0, 0, 1.0]], device=device
    ).reshape(1, 1, 3, 3)
    eps2d = 0.3

    qn = raw_q / raw_q.norm(dim=-1, keepdim=True)
    covars3, _ = _quat_scale_to_covar_preci(
        qn, scales, compute_covar=True, compute_preci=False, triu=False,
    )
    _, m2d_t, d_t, conics_t, _ = _fully_fused_projection(
        means, covars3, viewmats, Ks, W, H,
        eps2d=eps2d, near_plane=0.01, far_plane=1e10, calc_compensations=False,
    )
    g_m2d = torch.randn_like(m2d_t)
    g_d = torch.randn_like(d_t)
    g_c = torch.randn_like(conics_t)
    grads_t = torch.autograd.grad(
        (m2d_t * g_m2d).sum() + (d_t * g_d).sum() + (conics_t * g_c).sum(),
        [means, raw_q, scales], retain_graph=False, allow_unused=True,
    )

    radii_m, _, _, conics_m = _projection_ewa_3dgs_fused_fwd_metal(
        means.detach(), raw_q.detach(), scales.detach(), viewmats, Ks,
        W, H, eps2d, 0.01, 1e10, 0.0, None,
    )
    v_means, v_quats, v_scales = _projection_ewa_3dgs_fused_bwd_metal(
        means.detach(), raw_q.detach(), scales.detach(), viewmats, Ks,
        radii_m, conics_m, g_m2d, g_d, g_c, W, H,
    )

    grads_m = [v_means, v_quats, v_scales]
    names = ["v_means", "v_quats", "v_scales"]
    # Atomic-add ordering + fp32 amplifies absolute error proportionally to
    # leaf magnitude. Worst grad is v_means O(150); 1e-3 absolute = 7e-6 rel.
    for name, gt, gm in zip(names, grads_t, grads_m):
        assert gt is not None and gm is not None, name
        rel = (gt - gm).abs().max().item() / (gt.abs().max().item() + 1e-12)
        assert rel < 5e-4, f"{name}: rel diff {rel:.3e}"


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_projection_ut_3dgs_fused_fwd_metal_matches_torch():
    """Deferred-stage 3 (pinhole-only): Metal UT projection matches torch ref."""
    from gsplat.cuda._torch_impl_ut import _fully_fused_projection_with_ut
    from gsplat.cuda._wrapper import UnscentedTransformParameters
    from gsplat.mps._projection_ut import _projection_ut_3dgs_fused_fwd_metal

    device = "mps"
    g = torch.Generator(device="cpu").manual_seed(0)
    B, C, N, H, W = 1, 1, 32, 128, 128
    means = (torch.randn(B, N, 3, generator=g) * 0.5).to(device)
    raw_q = torch.randn(B, N, 4, generator=g).to(device)
    quats = raw_q / raw_q.norm(dim=-1, keepdim=True)
    scales = (torch.rand(B, N, 3, generator=g) * 0.1 + 0.02).to(device)
    opacities = (torch.rand(B, N, generator=g) * 0.5 + 0.3).to(device)
    viewmats = torch.eye(4)[None, None].clone().to(device)
    viewmats[..., :3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)
    Ks = torch.tensor(
        [[100.0, 0, 64.0], [0, 100.0, 64.0], [0, 0, 1.0]], device=device
    ).reshape(1, 1, 3, 3)
    eps2d = 0.3
    ut_params = UnscentedTransformParameters()  # default

    radii_t, m2d_t, d_t, conics_t, comp_t = _fully_fused_projection_with_ut(
        means, quats, scales, opacities, viewmats, Ks, W, H,
        eps2d=eps2d, calc_compensations=True,
        camera_model="pinhole", ut_params=ut_params,
    )
    radii_m, m2d_m, d_m, conics_m, comp_m = _projection_ut_3dgs_fused_fwd_metal(
        means, quats, scales, opacities, viewmats, Ks,
        W, H, eps2d, 0.01, 1e10, 0.0,
        ut_params.alpha, ut_params.beta, ut_params.kappa,
        ut_params.in_image_margin_factor,
        calc_compensations=True,
    )
    assert torch.equal(radii_t.int(), radii_m)
    valid = (radii_m > 0).all(dim=-1)
    assert (m2d_t[valid] - m2d_m[valid]).abs().max().item() < 1e-4
    assert (d_t[valid] - d_m[valid]).abs().max().item() < 1e-5
    assert (conics_t[valid] - conics_m[valid]).abs().max().item() < 1e-4
    assert (comp_t[valid] - comp_m[valid]).abs().max().item() < 1e-5


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_rasterize_to_pixels_eval3d_fwd_metal_matches_torch_inline():
    """Deferred-stage 4 (pinhole-only): Metal eval3d matches an inline torch
    reference. The shipped torch ref pulls in the deleted lidar module and
    requires nerfacc, so this test uses an inline pure-torch oracle that
    matches the Metal kernel's pinhole + global-shutter + CDIM=3 semantics."""
    import math

    import torch.nn.functional as F

    from gsplat.cuda._math import _quat_to_rotmat
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
    )
    from gsplat.mps._eval3d import _rasterize_to_pixels_eval3d_fwd_metal

    device = "mps"
    g = torch.Generator(device="cpu").manual_seed(0)
    B, C, N, H, W = 1, 1, 24, 32, 32
    means = (torch.randn(B, N, 3, generator=g) * 0.4).to(device)
    raw_q = torch.randn(B, N, 4, generator=g).to(device)
    quats = raw_q / raw_q.norm(dim=-1, keepdim=True)
    scales = (torch.rand(B, N, 3, generator=g) * 0.05 + 0.02).to(device)
    opacities = (torch.rand(B, C, N, generator=g) * 0.5 + 0.3).to(device)
    colors = torch.rand(B, C, N, 3, generator=g).to(device)
    viewmats = torch.eye(4)[None, None].clone().to(device)
    viewmats[..., :3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)
    Ks = torch.tensor(
        [[40.0, 0, 16.0], [0, 40.0, 16.0], [0, 0, 1.0]], device=device
    ).reshape(1, 1, 3, 3)
    radii, m2d, depths, _, _ = fully_fused_projection(
        means, None, quats, scales, viewmats, Ks, W, H, packed=False,
    )
    tile_size = 16
    tile_w = math.ceil(W / tile_size)
    tile_h = math.ceil(H / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        m2d, radii, depths, tile_size, tile_w, tile_h
    )
    iso = isect_offset_encode(isect_ids, B * C, tile_w, tile_h).reshape(
        B, C, tile_h, tile_w
    )
    I = B * C

    rc_m, ra_m, _ = _rasterize_to_pixels_eval3d_fwd_metal(
        means, quats, scales,
        colors.reshape(I, N, 3), opacities.reshape(I, N),
        viewmats.reshape(I, 4, 4), Ks.reshape(I, 3, 3),
        W, H, tile_size, iso.reshape(I, tile_h, tile_w), flatten_ids, C,
    )

    # Inline torch reference (pinhole + global shutter + CDIM=3).
    MAX_ALPHA = 1.0 - math.sqrt(1e-3)
    TRANS_THRESH = 1e-3
    R_q = _quat_to_rotmat(quats / quats.norm(dim=-1, keepdim=True))
    inv_s = 1.0 / scales
    rc_t = torch.zeros_like(rc_m)
    ra_t = torch.zeros_like(ra_m)
    iso_flat = iso.reshape(I, tile_h * tile_w)
    for img_id in range(I):
        bid = img_id // C
        R_view = viewmats.reshape(I, 4, 4)[img_id, :3, :3]
        t_view = viewmats.reshape(I, 4, 4)[img_id, :3, 3]
        ray_o_world = -(R_view.T @ t_view)
        Kf = Ks.reshape(I, 3, 3)[img_id]
        fx, fy, cxv, cyv = Kf[0, 0], Kf[1, 1], Kf[0, 2], Kf[1, 2]
        for ti in range(tile_h * tile_w):
            start = int(iso_flat[img_id, ti])
            ty, tx = divmod(ti, tile_w)
            last_tile = (img_id == I - 1) and (ti == tile_h * tile_w - 1)
            if last_tile:
                end = int(flatten_ids.numel())
            elif ti + 1 < tile_h * tile_w:
                end = int(iso_flat[img_id, ti + 1])
            else:
                end = (
                    int(iso_flat[img_id + 1, 0]) if img_id + 1 < I else int(flatten_ids.numel())
                )
            if end <= start:
                continue
            y0, y1 = ty * tile_size, min((ty + 1) * tile_size, H)
            x0, x1 = tx * tile_size, min((tx + 1) * tile_size, W)
            for py in range(y0, y1):
                for px in range(x0, x1):
                    d_cam = torch.tensor(
                        [(px + 0.5 - cxv) / fx, (py + 0.5 - cyv) / fy, 1.0],
                        device=device,
                    )
                    d_cam = F.normalize(d_cam, dim=-1)
                    ray_d_world = R_view.T @ d_cam
                    Tt = 1.0
                    col_acc = torch.zeros(3, device=device)
                    for idx in range(start, end):
                        gflat = int(flatten_ids[idx])
                        g_local = gflat - img_id * N
                        mw = means[bid, g_local]
                        d_origin = ray_o_world - mw
                        rqT_d = R_q[bid, g_local].T @ d_origin
                        gro = rqT_d * inv_s[bid, g_local]
                        rqT_dir = R_q[bid, g_local].T @ ray_d_world
                        grd_un = rqT_dir * inv_s[bid, g_local]
                        grd = F.normalize(grd_un, dim=-1, eps=1e-12)
                        gxg = torch.linalg.cross(grd, gro)
                        grayDist = (gxg * gxg).sum()
                        alpha = (
                            opacities[bid, 0, g_local] * torch.exp(-0.5 * grayDist)
                        ).clamp(max=MAX_ALPHA)
                        if alpha <= 0:
                            continue
                        next_T = Tt * (1.0 - alpha)
                        col_acc = col_acc + alpha * Tt * colors[bid, 0, g_local]
                        Tt = float(next_T)
                        if Tt < TRANS_THRESH:
                            break
                    rc_t[img_id, py, px] = col_acc
                    ra_t[img_id, py, px, 0] = 1.0 - Tt
    assert (rc_t - rc_m).abs().max().item() < 1e-4
    assert (ra_t - ra_m).abs().max().item() < 1e-4


@pytest.mark.skipif(
    not is_metal_available(),
    reason="torch.mps.compile_shader unavailable on this build",
)
def test_rasterize_to_pixels_eval3d_bwd_metal_matches_inline_autograd():
    """eval3d Metal bwd matches torch.autograd through an inline forward."""
    import math

    import torch.nn.functional as F

    from gsplat.cuda._math import _quat_to_rotmat
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
    )
    from gsplat.mps._eval3d import (
        _rasterize_to_pixels_eval3d_bwd_metal,
        _rasterize_to_pixels_eval3d_fwd_metal,
    )

    device = "mps"
    g = torch.Generator(device="cpu").manual_seed(7)
    B, C, N, H, W = 1, 1, 8, 16, 16
    means = (torch.randn(B, N, 3, generator=g) * 0.4).to(device).requires_grad_()
    raw_q = torch.randn(B, N, 4, generator=g).to(device).requires_grad_()
    quats_n = raw_q / raw_q.norm(dim=-1, keepdim=True)
    scales = (torch.rand(B, N, 3, generator=g) * 0.05 + 0.02).to(device).requires_grad_()
    opacities = (torch.rand(B, C, N, generator=g) * 0.5 + 0.3).to(device).requires_grad_()
    colors = torch.rand(B, C, N, 3, generator=g).to(device).requires_grad_()
    viewmats = torch.eye(4)[None, None].clone().to(device)
    viewmats[..., :3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)
    Ks = torch.tensor(
        [[40.0, 0, 8.0], [0, 40.0, 8.0], [0, 0, 1.0]], device=device
    ).reshape(1, 1, 3, 3)
    radii, m2d, depths, _, _ = fully_fused_projection(
        means.detach(), None, quats_n.detach(), scales.detach(),
        viewmats, Ks, W, H, packed=False,
    )
    tile_size = 16
    tile_w = math.ceil(W / tile_size)
    tile_h = math.ceil(H / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        m2d, radii, depths, tile_size, tile_w, tile_h
    )
    iso = isect_offset_encode(isect_ids, B * C, tile_w, tile_h).reshape(
        B, C, tile_h, tile_w
    )
    I = B * C

    # Inline torch reference (autograd-able, slow Python loop).
    MAX_ALPHA = 1.0 - math.sqrt(1e-3)
    TRANS_THRESH = 1e-3
    R_q = _quat_to_rotmat(quats_n)
    inv_s = 1.0 / scales
    rc_t = torch.zeros(I, H, W, 3, device=device)
    ra_t = torch.zeros(I, H, W, 1, device=device)
    iso_flat = iso.reshape(I, tile_h * tile_w)
    for img_id in range(I):
        bid = img_id // C
        R_view = viewmats.reshape(I, 4, 4)[img_id, :3, :3]
        t_view = viewmats.reshape(I, 4, 4)[img_id, :3, 3]
        ray_o_world = -(R_view.T @ t_view)
        Kf = Ks.reshape(I, 3, 3)[img_id]
        fxv, fyv, cxv, cyv = Kf[0, 0], Kf[1, 1], Kf[0, 2], Kf[1, 2]
        for ti in range(tile_h * tile_w):
            start = int(iso_flat[img_id, ti])
            ty, tx = divmod(ti, tile_w)
            last_tile = (img_id == I - 1) and (ti == tile_h * tile_w - 1)
            if last_tile:
                end = int(flatten_ids.numel())
            else:
                end = (
                    int(iso_flat[img_id, ti + 1])
                    if ti + 1 < tile_h * tile_w
                    else (
                        int(iso_flat[img_id + 1, 0])
                        if img_id + 1 < I
                        else int(flatten_ids.numel())
                    )
                )
            if end <= start:
                continue
            y0, y1 = ty * tile_size, min((ty + 1) * tile_size, H)
            x0, x1 = tx * tile_size, min((tx + 1) * tile_size, W)
            for py in range(y0, y1):
                for px in range(x0, x1):
                    d_cam = torch.stack([
                        (px + 0.5 - cxv) / fxv, (py + 0.5 - cyv) / fyv,
                        torch.tensor(1.0, device=device),
                    ])
                    d_cam = F.normalize(d_cam, dim=-1)
                    ray_d_world = R_view.T @ d_cam
                    Tt = torch.tensor(1.0, device=device)
                    col_acc = torch.zeros(3, device=device)
                    for idx in range(start, end):
                        gflat = int(flatten_ids[idx])
                        g_local = gflat - img_id * N
                        mw = means[bid, g_local]
                        d_origin = ray_o_world - mw
                        rqT_d = R_q[bid, g_local].T @ d_origin
                        gro = rqT_d * inv_s[bid, g_local]
                        rqT_dir = R_q[bid, g_local].T @ ray_d_world
                        grd_un = rqT_dir * inv_s[bid, g_local]
                        grd = F.normalize(grd_un, dim=-1, eps=1e-12)
                        gxg = torch.linalg.cross(grd, gro)
                        grayDist = (gxg * gxg).sum()
                        alpha = (
                            opacities[bid, 0, g_local] * torch.exp(-0.5 * grayDist)
                        ).clamp(max=MAX_ALPHA)
                        if alpha.item() <= 0:
                            continue
                        col_acc = col_acc + alpha * Tt * colors[bid, 0, g_local]
                        Tt = Tt * (1.0 - alpha)
                        if Tt.item() < TRANS_THRESH:
                            break
                    rc_t[img_id, py, px] = col_acc
                    ra_t[img_id, py, px, 0] = 1.0 - Tt
    g_rc = torch.randn_like(rc_t)
    g_ra = torch.randn_like(ra_t)
    leafs = [means, raw_q, scales, colors, opacities]
    grads_t = torch.autograd.grad(
        (rc_t * g_rc).sum() + (ra_t * g_ra).sum(),
        leafs, retain_graph=False, allow_unused=True,
    )

    # Metal fwd + bwd.
    rc_m, ra_m, last_ids = _rasterize_to_pixels_eval3d_fwd_metal(
        means.detach(), quats_n.detach(), scales.detach(),
        colors.detach().reshape(I, N, 3), opacities.detach().reshape(I, N),
        viewmats.reshape(I, 4, 4), Ks.reshape(I, 3, 3),
        W, H, tile_size, iso.reshape(I, tile_h, tile_w), flatten_ids, C,
    )
    v_m, v_qn, v_s, v_c, v_o = _rasterize_to_pixels_eval3d_bwd_metal(
        means.detach(), quats_n.detach(), scales.detach(),
        colors.detach().reshape(I, N, 3), opacities.detach().reshape(I, N),
        viewmats.reshape(I, 4, 4), Ks.reshape(I, 3, 3),
        iso.reshape(I, tile_h, tile_w), flatten_ids,
        ra_m, last_ids, g_rc, g_ra,
        W, H, tile_size, C,
    )
    # Convert v_quats_norm → v_raw_q via L2-normalize VJP.
    raw_q_leaf = raw_q.detach().requires_grad_()
    qn_chain = raw_q_leaf / raw_q_leaf.norm(dim=-1, keepdim=True)
    (raw_grad,) = torch.autograd.grad(qn_chain, [raw_q_leaf], grad_outputs=v_qn)

    grads_m = [
        v_m.reshape(B, N, 3),
        raw_grad,
        v_s.reshape(B, N, 3),
        v_c.reshape(B, C, N, 3),
        v_o.reshape(B, C, N),
    ]
    names = ["v_means", "v_raw_q", "v_scales", "v_colors", "v_opac"]
    for n, gt, gm in zip(names, grads_t, grads_m):
        assert gt is not None and gm is not None, n
        rel = (gt - gm).abs().max().item() / (gt.abs().max().item() + 1e-12)
        assert rel < 5e-4, f"{n}: rel diff {rel:.3e}"


def test_dispatch_registry_round_trip():
    # `select_backend` should route MPS tensors to METAL only after the op
    # name has been registered. Use a unique-per-test name to avoid leaking
    # state into other tests.
    op = "stage5_smoke_kernel"
    assert not has_metal(op)
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    t = torch.zeros(1, device="mps")
    assert select_backend(t, op) is Backend.TORCH
    register_metal_kernel(op)
    try:
        assert has_metal(op)
        assert select_backend(t, op) is Backend.METAL
    finally:
        from gsplat.cuda._dispatch import _METAL_KERNELS_AVAILABLE

        _METAL_KERNELS_AVAILABLE.discard(op)
