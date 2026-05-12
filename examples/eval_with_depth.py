"""Evaluate a saved checkpoint against the 2DGS DTU bundle's *sparse* depth.

The bundle stores per-image .pt files in ``depths/`` shaped as::

    {image_name, depth (N,), coord (N, 2), error (N,), weight (N,)}

where ``depth`` is a sparse set of MVS-verified depth samples at the
``coord`` pixel locations (originally COLMAP feature points with depth).
There is *no* dense GT depth map in this bundle, so the comparison is
pred-depth-sampled-at-the-sparse-points vs. those sparse values.

Outputs (under ``--out_dir``):

  metrics_with_depth.json
    {
      ...existing photometric metrics from metrics.json...,
      "depth_rmse":     sqrt(mean((pred - gt)^2)) over all sparse samples,
      "depth_rel_err":  mean(|pred - gt| / gt),
      "depth_rmse_w":   weighted RMSE using `weight` as importance,
      "depth_n_views":  number of holdout views with matching .pt,
      "depth_n_samples": total sparse samples summed across views,
    }

  depth_panels/cmp_<i>.png
    [rgb | predicted depth | sparse points scattered on the pred map,
     colored by signed error]

Usage::

    python examples/eval_with_depth.py \\
        --data_dir   data/DTU/scan24 \\
        --ckpt       results/.../baseline/ckpts/ckpt_30000.pt \\
        --out_dir    results/.../baseline \\
        --data_factor 2
"""

# SPDX-FileCopyrightText: Copyright 2026 Zachary Scott-Murphy
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "examples"))
for _k in list(sys.modules):
    if _k == "datasets" or _k.startswith("datasets."):
        del sys.modules[_k]

from datasets.colmap import Dataset, Parser  # noqa: E402
from methods.rade_gs import rasterization_rade_gs  # noqa: E402


@dataclass
class Config:
    data_dir: str
    ckpt: str
    out_dir: str
    data_factor: int = 2
    test_every: int = 8
    depths_subdir: str = "depths"
    sh_degree: int = 3
    near_plane: float = 0.01
    far_plane: float = 1e10
    n_visualize: int = 8


def _find_depth_file(depths_dir: Path, image_name: str) -> Optional[Path]:
    stem = Path(image_name).stem
    for ext in (".pt", ".png", ".exr", ".npy"):
        p = depths_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for p in depths_dir.iterdir():
        if p.stem == stem or p.stem.lstrip("0") == stem.lstrip("0"):
            return p
    return None


def _load_sparse_depth(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (depth_values, pixel_xy, weight) from a 2DGS-bundle .pt file."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"unexpected .pt content (expected dict): {type(obj)}")
    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    depth = _np(obj["depth"]).astype(np.float32).reshape(-1)
    coord = _np(obj["coord"]).astype(np.float32).reshape(-1, 2)
    weight = _np(obj["weight"]).astype(np.float32).reshape(-1) if "weight" in obj else np.ones_like(depth)
    return depth, coord, weight


def _scale_coords(coord: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, float]:
    """Rescale `coord` to match a rendered image of size (H, W).

    The .pt coords are stored in the *original* DTU image resolution
    (~1600×1200). When we render at --data_factor 2 the rendered map is
    half that, so coords must be scaled down. We detect the factor from
    the data rather than trusting an argument.
    """
    cx_max = float(coord[:, 0].max())
    cy_max = float(coord[:, 1].max())
    # Use the dimension with the larger ratio to set the scale.
    sx = (W - 1) / cx_max if cx_max > 0 else 1.0
    sy = (H - 1) / cy_max if cy_max > 0 else 1.0
    s = min(sx, sy)
    # Snap to 1.0 if we're already close — avoids fractional drift.
    if 0.98 <= s <= 1.02:
        s = 1.0
    return coord * s, s


def _sample_bilinear(depth_map: torch.Tensor, coord_xy: np.ndarray) -> np.ndarray:
    """Bilinearly sample (H, W) depth_map at pixel coords (N, 2) → (N,)."""
    H, W = depth_map.shape
    xn = torch.from_numpy(coord_xy[:, 0]).float()
    yn = torch.from_numpy(coord_xy[:, 1]).float()
    xn = (xn / (W - 1)) * 2 - 1
    yn = (yn / (H - 1)) * 2 - 1
    grid = torch.stack([xn, yn], dim=-1).view(1, 1, -1, 2).to(depth_map.device)
    d = depth_map.view(1, 1, H, W)
    return F.grid_sample(d, grid, mode="bilinear", padding_mode="zeros", align_corners=True) \
        .view(-1).cpu().numpy()


def _colorize(d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Cheap turbo-ish colormap. (H, W) → (H, W, 3) uint8."""
    x = np.clip((d - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    r = np.clip(1.5 - 4 * np.abs(x - 0.75), 0, 1)
    g = np.clip(1.5 - 4 * np.abs(x - 0.50), 0, 1)
    b = np.clip(1.5 - 4 * np.abs(x - 0.25), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _scatter_error(pred_vis: np.ndarray, coord: np.ndarray, err: np.ndarray, vmax: float) -> np.ndarray:
    """Draw filled circles colored by signed error on top of an RGB depth viz."""
    out = pred_vis.copy()
    H, W = out.shape[:2]
    # Color: red = pred too far, blue = pred too close, white-ish = good.
    e = np.clip(err / max(vmax, 1e-6), -1, 1)
    r = np.clip( 1.0 - np.minimum(e, 0) * -1.0, 0, 1)
    b = np.clip( 1.0 - np.maximum(e, 0) * 1.0, 0, 1)
    g = np.clip( 1.0 - np.abs(e),              0, 1)
    cols = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    radius = 2
    for (x, y), c in zip(coord.astype(np.int64), cols):
        if not (radius <= x < W - radius and radius <= y < H - radius):
            continue
        out[y - radius:y + radius + 1, x - radius:x + radius + 1] = c
    return out


@torch.no_grad()
def main(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    parser = Parser(
        data_dir=cfg.data_dir, factor=cfg.data_factor,
        normalize=True, test_every=cfg.test_every,
    )
    valset = Dataset(parser, split="val")
    print(f"holdout views: {len(valset)}")

    ckpt = torch.load(cfg.ckpt, map_location=device)
    splats = {k: v.to(device) for k, v in ckpt.items()}

    depths_dir = Path(cfg.data_dir) / cfg.depths_subdir
    if not depths_dir.exists():
        raise FileNotFoundError(f"depths folder missing: {depths_dir}")
    print(f"depths from: {depths_dir}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "depth_panels"
    panels_dir.mkdir(exist_ok=True)

    sq_errs: list[float] = []
    abs_errs: list[float] = []
    rel_errs: list[float] = []
    sq_w_errs: list[float] = []
    weights_sum = 0.0
    n_views = 0
    n_samples = 0
    coord_scale_used: Optional[float] = None

    for i in range(len(valset)):
        item = valset[i]
        image = item["image"].unsqueeze(0).to(device) / 255.0
        camtoworld = item["camtoworld"].unsqueeze(0).to(device)
        K = item["K"].unsqueeze(0).to(device)
        H, W = image.shape[1:3]
        viewmat = torch.linalg.inv(camtoworld)
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

        out = rasterization_rade_gs(
            means=splats["means"],
            quats=splats["quats"] / splats["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmat, Ks=K, width=W, height=H,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
            sh_degree=cfg.sh_degree,
        )
        d_pred = out["render_depths"][0, ..., 0]                        # (H, W) on device
        rgb_pred = out["render_colors"][0, ..., :3].clamp(0, 1).cpu().numpy()

        image_name = parser.image_names[valset.indices[i]]
        gt_path = _find_depth_file(depths_dir, image_name)
        if gt_path is None:
            print(f"  [{i}] no .pt for {image_name}")
            continue

        d_gt, coord, weight = _load_sparse_depth(gt_path)
        coord_scaled, scale = _scale_coords(coord, H, W)
        if coord_scale_used is None:
            coord_scale_used = scale
            print(f"  detected coord scale (orig→rendered): {scale:.4f}")
        # Drop samples outside the image (after scaling).
        in_bounds = (
            (coord_scaled[:, 0] >= 0) & (coord_scaled[:, 0] < W) &
            (coord_scaled[:, 1] >= 0) & (coord_scaled[:, 1] < H) &
            (d_gt > 0)
        )
        coord_scaled = coord_scaled[in_bounds]
        d_gt = d_gt[in_bounds]
        weight = weight[in_bounds]
        if len(d_gt) == 0:
            print(f"  [{i}] all samples out of bounds")
            continue

        pred_at = _sample_bilinear(d_pred, coord_scaled)                # (N,)
        valid = pred_at > 0
        if not valid.any():
            print(f"  [{i}] no pixels covered at any sample point")
            continue
        err = pred_at[valid] - d_gt[valid]
        w = weight[valid]

        sq_errs.append(float((err ** 2).mean()))
        sq_w_errs.append(float((w * err ** 2).sum()))
        weights_sum += float(w.sum())
        abs_errs.append(float(np.abs(err).mean()))
        rel_errs.append(float((np.abs(err) / np.maximum(d_gt[valid], 1e-6)).mean()))
        n_views += 1
        n_samples += int(valid.sum())

        if i < cfg.n_visualize:
            d_pred_np = d_pred.cpu().numpy()
            gt_min, gt_max = float(d_gt[valid].min()), float(d_gt[valid].max())
            vmin = max(0.1, gt_min * 0.9)
            vmax = gt_max * 1.1
            rgb_u8 = (rgb_pred * 255).astype(np.uint8)
            pred_vis = _colorize(d_pred_np, vmin, vmax)
            err_thr = np.percentile(np.abs(err), 95) if len(err) > 5 else max(np.abs(err).max(), 0.05)
            scatter_vis = _scatter_error(pred_vis, coord_scaled[valid], err, vmax=err_thr)
            panel = np.concatenate([rgb_u8, pred_vis, scatter_vis], axis=1)
            imageio.imwrite(panels_dir / f"cmp_{i:03d}.png", panel)

    metrics: dict = {}
    legacy = out_dir / "metrics.json"
    if legacy.exists():
        metrics.update(json.loads(legacy.read_text()))
    if n_views > 0:
        metrics["depth_rmse"] = float(np.sqrt(np.mean(sq_errs)))
        metrics["depth_rel_err"] = float(np.mean(rel_errs))
        metrics["depth_mae"] = float(np.mean(abs_errs))
        metrics["depth_rmse_w"] = float(np.sqrt(np.sum(sq_w_errs) / max(weights_sum, 1e-6)))
        metrics["depth_n_views"] = n_views
        metrics["depth_n_samples"] = n_samples
    else:
        metrics["depth_rmse"] = None
        metrics["depth_n_views"] = 0
    (out_dir / "metrics_with_depth.json").write_text(json.dumps(metrics, indent=2))

    print("\n  views with depth: ", n_views, " / total:", len(valset))
    print("  total samples:    ", n_samples)
    if n_views > 0:
        print(f"  depth_rmse       : {metrics['depth_rmse']:.4f} m")
        print(f"  depth_mae        : {metrics['depth_mae']:.4f} m")
        print(f"  depth_rel_err    : {metrics['depth_rel_err']:.4f}")
        print(f"  depth_rmse_w     : {metrics['depth_rmse_w']:.4f} m  (weight-weighted)")
    print(f"\n  → wrote {out_dir / 'metrics_with_depth.json'}")
    print(f"  → panels {panels_dir}/cmp_*.png  (rgb | pred-depth | sparse-error-overlay)")


if __name__ == "__main__":
    main(tyro.cli(Config))
