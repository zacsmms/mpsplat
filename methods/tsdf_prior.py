"""
Primitive TSDF-prior for RaDe-GS — experiment scaffolding.

Idea (see EXPLORATION discussion): the rasterized depth map from any single
view is biased by the argmax-as-intersection assumption. Multi-view consensus
should reduce that bias. We approximate the consensus surface by fusing the
current per-view depth maps into a TSDF volume, freeze it for some N
iterations, and add a regularization loss that pulls per-view depth toward
the consensus surface.

Pipeline:

  1.  Every ``refresh_every`` training iters, call ``TSDFVolume.integrate``
      with each training-view (depth, viewmat, K) tuple. This produces a
      frozen, voxelized SDF representing the average opinion of all views.
  2.  In the per-iter training loss, call ``tsdf_prior_loss(...)`` with the
      current view's rendered depth map and pose. It projects each pixel
      into world space, trilinearly samples the frozen SDF, and returns
      mean |SDF(x_pred)|. Drives predicted depth toward the zero level set.

Everything is pure torch — runs unchanged on MPS, CUDA, or CPU. No Open3D
dependency, since the volume here is for *regularization*, not for the final
mesh extraction (that still wants the Open3D / Marching-Tetrahedra path).

Caveats:

  * Voxel grid is axis-aligned and pre-allocated. For ``resolution=256`` and
    bounds spanning a 4-unit cube, memory is ~3 × 256³ × 4 B ≈ 200 MB (SDF +
    weight + a one-time voxel-centre grid that we don't keep around).
  * "Random" per-view error gets averaged out by fusion; "systematic" bias
    that's correlated across views does not. This is the well-known limit
    of self-distillation. See the discussion notes for why we still expect
    a measurable gain on the easy half of the failure modes.
"""

# SPDX-FileCopyrightText: Copyright 2026 Zachary Scott-Murphy
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TSDFConfig:
    """Hyperparameters for the primitive TSDF-prior."""

    # Axis-aligned world-space bounds: (min_xyz, max_xyz), each shape (3,).
    bounds_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    bounds_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Per-axis voxel count. Cubic is fine for object-scale scenes.
    resolution: int = 128
    # Truncation distance in world units. Rule of thumb: 4-8 × voxel size.
    truncation: float = 0.04
    # Maximum accumulated weight per voxel — caps the influence of any single
    # refresh so newer views can still nudge old voxels.
    max_weight: float = 64.0
    # Skip integration for pixels with alpha < this (no Gaussian coverage).
    alpha_threshold: float = 1e-3


class TSDFVolume:
    """A primitive axis-aligned TSDF volume for self-distillation regularization."""

    def __init__(self, cfg: TSDFConfig, device: torch.device, dtype: torch.dtype = torch.float32):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        R = cfg.resolution
        self.sdf = torch.full((R, R, R), cfg.truncation, device=device, dtype=dtype)
        self.weight = torch.zeros((R, R, R), device=device, dtype=dtype)

        self.bounds_min = torch.tensor(cfg.bounds_min, device=device, dtype=dtype)
        self.bounds_max = torch.tensor(cfg.bounds_max, device=device, dtype=dtype)
        self.voxel_size = (self.bounds_max - self.bounds_min) / (R - 1)

    @torch.no_grad()
    def reset(self) -> None:
        """Clear the volume — call before each refresh so views vote fresh."""
        self.sdf.fill_(self.cfg.truncation)
        self.weight.zero_()

    @torch.no_grad()
    def _voxel_centres(self) -> Tensor:
        """(R, R, R, 3) grid of world-space voxel-centre coordinates."""
        R = self.cfg.resolution
        lin = [
            torch.linspace(
                self.bounds_min[i].item(),
                self.bounds_max[i].item(),
                R,
                device=self.device,
                dtype=self.dtype,
            )
            for i in range(3)
        ]
        xx, yy, zz = torch.meshgrid(lin[0], lin[1], lin[2], indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1)  # (R, R, R, 3)

    @torch.no_grad()
    def integrate(
        self,
        depth: Tensor,  # (H, W) camera-space z-depth in world units
        viewmat: Tensor,  # (4, 4) world → camera
        K: Tensor,  # (3, 3) pinhole intrinsics matched to (H, W)
        alpha: Optional[Tensor] = None,  # (H, W) coverage mask, optional
    ) -> None:
        """Fuse one view into the volume.

        Standard TSDF update: project every voxel into the camera, read the
        depth at the projected pixel, compute observed SDF = depth - z_voxel
        (positive ⇒ voxel is between camera and surface), clamp to ±truncation,
        running-weighted-average into the stored SDF/weight grids.
        """
        H, W = depth.shape
        trunc = self.cfg.truncation

        centres = self._voxel_centres().reshape(-1, 3)  # (V, 3)

        R = viewmat[:3, :3]
        t = viewmat[:3, 3]
        pts_cam = centres @ R.T + t  # (V, 3)
        z = pts_cam[..., 2]

        # Project to pixel coords.
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        in_front = z > 1e-6
        z_safe = z.clamp(min=1e-6)
        u = fx * pts_cam[..., 0] / z_safe + cx
        v = fy * pts_cam[..., 1] / z_safe + cy

        # F.grid_sample expects normalized coords in [-1, 1] with shape (1, 1, 1, V, 2).
        grid_u = (u / (W - 1)) * 2 - 1
        grid_v = (v / (H - 1)) * 2 - 1
        grid = torch.stack([grid_u, grid_v], dim=-1).view(1, 1, -1, 2)

        depth_in = depth.view(1, 1, H, W)
        depth_sampled = F.grid_sample(
            depth_in, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).view(-1)  # (V,)

        in_image = (grid_u.abs() <= 1) & (grid_v.abs() <= 1)
        has_depth = depth_sampled > 1e-6

        # Optional alpha mask (no-coverage pixels shouldn't vote).
        if alpha is not None:
            alpha_in = alpha.view(1, 1, H, W)
            alpha_sampled = F.grid_sample(
                alpha_in, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).view(-1)
            has_depth = has_depth & (alpha_sampled > self.cfg.alpha_threshold)

        sdf_obs = depth_sampled - z  # +ve: voxel in front of surface
        in_band = sdf_obs.abs() < trunc

        valid = in_front & in_image & has_depth & in_band
        sdf_obs = sdf_obs.clamp(-trunc, trunc)

        sdf_flat = self.sdf.view(-1)
        w_flat = self.weight.view(-1)

        w_old = w_flat
        w_new_capped = (w_old + 1.0).clamp(max=self.cfg.max_weight)
        sdf_new = (sdf_flat * w_old + sdf_obs) / w_new_capped.clamp(min=1.0)

        sdf_flat[:] = torch.where(valid, sdf_new, sdf_flat)
        w_flat[:] = torch.where(valid, w_new_capped, w_old)

    def query_sdf(self, world_pts: Tensor) -> Tuple[Tensor, Tensor]:
        """Trilinearly sample SDF and weight at world-space points.

        Args:
            world_pts: (..., 3) tensor of query positions.
        Returns:
            sdf: (...) interpolated SDF values, clamped outside the volume.
            valid: (...) boolean — true where the sample fell inside the volume
                AND the local weight is non-zero (i.e. some view voted there).
        """
        flat = world_pts.reshape(-1, 3)
        norm = (flat - self.bounds_min) / (self.bounds_max - self.bounds_min)
        # grid_sample expects (x, y, z) in [-1, 1] with shape (N, D, H, W, 3).
        grid = (norm * 2 - 1).view(1, 1, 1, -1, 3)
        # Our stored axes are (i_x, j_y, k_z) — same as world XYZ in the grid we built.
        # grid_sample treats the last input dim as W, so we permute (D, H, W) = (Z, Y, X).
        sdf_vol = self.sdf.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
        w_vol = self.weight.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
        sdf = F.grid_sample(sdf_vol, grid, mode="bilinear", padding_mode="border", align_corners=True).view(-1)
        w = F.grid_sample(w_vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True).view(-1)
        in_box = (norm > 0).all(dim=-1) & (norm < 1).all(dim=-1)
        valid = in_box & (w > 0.5)
        return sdf.view(world_pts.shape[:-1]), valid.view(world_pts.shape[:-1])


def _unproject_depth(
    depth: Tensor, viewmat: Tensor, K: Tensor, alpha: Optional[Tensor] = None, alpha_thresh: float = 1e-3
) -> Tuple[Tensor, Tensor]:
    """Lift a (H, W) depth map into world-space (H, W, 3) point cloud.

    Returns (world_pts, mask) where mask flags pixels with valid coverage.
    """
    H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    ys = torch.arange(H, device=device, dtype=dtype) + 0.5
    xs = torch.arange(W, device=device, dtype=dtype) + 0.5
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (uu - cx) * depth / fx
    y_cam = (vv - cy) * depth / fy
    z_cam = depth
    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H, W, 3)

    # World = R^T (cam - t).
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    pts_world = (pts_cam - t) @ R  # equivalent to R^T @ (cam - t) for column vecs

    mask = depth > 1e-6
    if alpha is not None:
        mask = mask & (alpha > alpha_thresh)
    return pts_world, mask


def tsdf_prior_loss(
    volume: TSDFVolume,
    depth: Tensor,  # (H, W) predicted z-depth, gradient flowing
    viewmat: Tensor,  # (4, 4)
    K: Tensor,  # (3, 3)
    alpha: Optional[Tensor] = None,  # (H, W)
    huber_delta: float = 0.0,  # 0 ⇒ pure |.|; >0 ⇒ Huber with this delta
) -> Tensor:
    """``L_tsdf = E_pixels[ρ(SDF(unproject(d, view))]``, with ρ = |.| or Huber.

    The TSDF is frozen between refreshes (see ``TSDFVolume.integrate``), so
    gradients flow only through ``depth`` — exactly the self-distillation
    pattern we want.
    """
    world_pts, mask = _unproject_depth(depth, viewmat, K, alpha=alpha)
    sdf, valid = volume.query_sdf(world_pts)
    mask = mask & valid
    if mask.sum() == 0:
        return depth.new_zeros(())

    residual = sdf[mask]
    if huber_delta > 0:
        absr = residual.abs()
        quad = 0.5 * residual.pow(2) / huber_delta
        lin = absr - 0.5 * huber_delta
        loss = torch.where(absr < huber_delta, quad, lin)
    else:
        loss = residual.abs()
    return loss.mean()


@torch.no_grad()
def refresh_tsdf_from_views(
    volume: TSDFVolume,
    depths: Iterable[Tensor],
    viewmats: Iterable[Tensor],
    Ks: Iterable[Tensor],
    alphas: Optional[Iterable[Optional[Tensor]]] = None,
) -> None:
    """Reset the volume and re-integrate the current depth maps from all views."""
    volume.reset()
    alphas_iter = alphas if alphas is not None else [None] * 10**9
    for d, vm, K, a in zip(depths, viewmats, Ks, alphas_iter):
        volume.integrate(d.detach(), vm.detach(), K.detach(), alpha=a.detach() if a is not None else None)


# ---------------------------------------------------------------------------
# Smoke test: fuse synthetic depth maps of a known plane, check the zero level
# set of the resulting SDF lands close to the plane's true location. Tests the
# integrate + query + loss pipeline without needing any 3DGS infrastructure.
# Run with: ``python -m methods.tsdf_prior``
# ---------------------------------------------------------------------------
def _render_plane_depth(
    plane_z: float, viewmat: Tensor, K: Tensor, H: int, W: int
) -> Tensor:
    """Depth map of the world-plane z = plane_z, rendered through (viewmat, K)."""
    device, dtype = viewmat.device, viewmat.dtype
    ys = torch.arange(H, device=device, dtype=dtype) + 0.5
    xs = torch.arange(W, device=device, dtype=dtype) + 0.5
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Ray direction in camera frame.
    dx = (uu - cx) / fx
    dy = (vv - cy) / fy
    dz = torch.ones_like(dx)
    dirs_cam = torch.stack([dx, dy, dz], dim=-1)
    # Express plane in camera frame: world point (X, Y, plane_z) → cam.
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    # Plane normal in world is (0, 0, 1); in camera frame: R · (0,0,1).
    n_cam = R[:, 2]  # (3,)
    cam_origin_world = -R.T @ t  # world position of camera centre
    # Distance from camera origin to plane along its rays: solve
    #   plane_z = (cam_origin_world + s · (R^T dirs_cam))_z.
    # Easier: plane equation in cam: n_cam · X_cam + d = 0 with d = -n_cam · t' where t' = plane_z in world.
    # Plane: z_world = plane_z  ⇒ in cam: n_cam · X_cam = plane_z - n_cam · cam_origin_world
    rhs = plane_z - n_cam @ cam_origin_world
    denom = (n_cam.view(1, 1, 3) * dirs_cam).sum(dim=-1)  # (H, W)
    t_ray = rhs / denom.clamp(min=1e-6)
    return t_ray * dirs_cam[..., 2]  # convert ray-distance to z-depth


def _smoke_test() -> None:
    print("[tsdf_prior smoke test]")
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"  device: {device}")

    H, W = 96, 128
    fx = fy = 100.0
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device=device)

    # Five views looking at a plane at z = 1.0 from camera positions around z = 0.
    plane_z = 1.0
    cam_centres = [
        (0.0, 0.0, 0.0),
        (0.15, 0.0, 0.0),
        (-0.15, 0.0, 0.0),
        (0.0, 0.15, 0.0),
        (0.0, -0.15, 0.0),
    ]
    viewmats = []
    depths = []
    for (cx_, cy_, cz_) in cam_centres:
        vm = torch.eye(4, device=device)
        vm[:3, 3] = torch.tensor([-cx_, -cy_, -cz_], device=device)  # world->cam translation
        viewmats.append(vm)
        depths.append(_render_plane_depth(plane_z, vm, K, H, W))

    # Build TSDF with bounds tight around the plane.
    cfg = TSDFConfig(
        bounds_min=(-0.5, -0.5, 0.6),
        bounds_max=(0.5, 0.5, 1.4),
        resolution=64,
        truncation=0.08,
    )
    vol = TSDFVolume(cfg, device=device)
    refresh_tsdf_from_views(vol, depths, viewmats, [K] * len(depths))

    # Sample SDF along the world-z axis at (0, 0, *) and check the zero crossing.
    zs = torch.linspace(0.7, 1.3, 61, device=device)
    pts = torch.stack(
        [torch.zeros_like(zs), torch.zeros_like(zs), zs], dim=-1
    )  # (61, 3)
    sdf, valid = vol.query_sdf(pts)
    sdf = sdf.cpu()
    valid = valid.cpu()
    zs_cpu = zs.cpu()

    crossings = []
    for i in range(len(zs_cpu) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        if sdf[i] * sdf[i + 1] <= 0:
            # Linear interpolate the zero.
            t_ = sdf[i] / (sdf[i] - sdf[i + 1])
            crossings.append(zs_cpu[i] + t_ * (zs_cpu[i + 1] - zs_cpu[i]))

    print(f"  valid samples along z: {valid.sum().item()}/{len(valid)}")
    print(f"  zero crossings found:  {[round(c.item(), 4) for c in crossings]}")
    print(f"  ground-truth plane z:  {plane_z}")
    if crossings:
        err = min(abs(c.item() - plane_z) for c in crossings)
        print(f"  best-crossing error:   {err:.5f}  (voxel size = {vol.voxel_size[2].item():.5f})")

    # Loss: predicted depth = true depth → loss ≈ 0.
    loss_perfect = tsdf_prior_loss(vol, depths[0], viewmats[0], K)
    # Perturb depth by +5 cm → loss should jump.
    loss_perturbed = tsdf_prior_loss(vol, depths[0] + 0.05, viewmats[0], K)
    print(f"  loss(true depth):      {loss_perfect.item():.5f}")
    print(f"  loss(+5cm depth):      {loss_perturbed.item():.5f}")
    assert loss_perturbed > loss_perfect, "TSDF prior should penalize biased depth."
    print("  ok.")


if __name__ == "__main__":
    _smoke_test()
