"""Re-evaluate a saved checkpoint with DTU GT depth, no retraining.

Usage::

    python examples/eval_with_depth.py \\
        --data_dir   data/DTU/scan24 \\
        --ckpt       results/.../baseline/ckpts/ckpt_30000.pt \\
        --out_dir    results/.../baseline \\
        --data_factor 2

Writes ``metrics_with_depth.json`` next to ``out_dir`` and, for the first
few holdout views, saves a side-by-side panel ``cmp_<i>.png`` containing
[rendered RGB | rendered depth | GT depth | abs error] all normalized to
GT's depth range for a fair visual diff.

DTU GT depth format auto-detection:

  * PNG 16-bit → values in mm → divide by 1000 (the 2DGS bundle's format).
  * EXR float  → meters as-is.
  * NPY        → read raw.
  * Override   → pass ``--depth_scale 256`` etc.

The depth file for image ``foo.jpg`` is expected at ``depths/foo.<ext>``.
"""

# SPDX-FileCopyrightText: Copyright 2026 Zachary Scott-Murphy
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
import torch
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
    depth_scale: Optional[float] = None      # None ⇒ auto-detect from file dtype
    sh_degree: int = 3
    near_plane: float = 0.01
    far_plane: float = 1e10
    n_visualize: int = 8                     # how many side-by-side panels to save


_DEPTH_EXTS = (".png", ".exr", ".tif", ".tiff", ".npy")


def _find_depth_file(depths_dir: Path, image_name: str) -> Optional[Path]:
    stem = Path(image_name).stem
    for ext in _DEPTH_EXTS:
        p = depths_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # Some bundles strip the leading zero or pad differently — try a fuzzy match.
    for p in depths_dir.iterdir():
        if p.stem == stem or p.stem.lstrip("0") == stem.lstrip("0"):
            return p
    return None


def _load_depth(path: Path, override_scale: Optional[float]) -> np.ndarray:
    """Return depth in metres as (H, W) float32."""
    if path.suffix.lower() == ".npy":
        d = np.load(path).astype(np.float32)
    else:
        d = imageio.imread(path)
        if d.ndim == 3:                        # some EXRs come back as (H, W, 1)
            d = d[..., 0]
        d = d.astype(np.float32)
    if override_scale is not None:
        d = d / override_scale
    elif d.dtype == np.uint16 or d.max() > 100:  # almost certainly mm
        d = d / 1000.0
    return d


def _resize_depth_to(d: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbor resize to (h, w). Avoids interpolating across depth discontinuities."""
    H, W = d.shape
    if (H, W) == (h, w):
        return d
    ys = (np.linspace(0, H - 1, h)).round().astype(np.int64)
    xs = (np.linspace(0, W - 1, w)).round().astype(np.int64)
    return d[ys[:, None], xs[None, :]]


def _colorize(d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map (H, W) → (H, W, 3) uint8 via the 'turbo' colormap (approx, no matplotlib dep)."""
    x = np.clip((d - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    # Cheap perceptual ramp: blue → cyan → yellow → red.
    r = np.clip(1.5 - 4 * np.abs(x - 0.75), 0, 1)
    g = np.clip(1.5 - 4 * np.abs(x - 0.50), 0, 1)
    b = np.clip(1.5 - 4 * np.abs(x - 0.25), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


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
        raise FileNotFoundError(f"GT depth folder missing: {depths_dir}")
    print(f"depths from: {depths_dir}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "depth_panels"
    panels_dir.mkdir(exist_ok=True)

    rmses: list[float] = []
    rel_errs: list[float] = []

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
        d_pred = out["render_depths"][0, ..., 0].cpu().numpy()      # (H, W)
        alpha = out["render_alphas"][0, ..., 0].cpu().numpy()
        rgb_pred = out["render_colors"][0, ..., :3].clamp(0, 1).cpu().numpy()

        # Pull the image filename from the parser (matches Dataset's __getitem__ index).
        image_name = parser.image_names[valset.indices[i]]
        gt_path = _find_depth_file(depths_dir, image_name)
        if gt_path is None:
            print(f"  [{i}] no GT depth for {image_name} — skipping rmse")
            continue
        d_gt = _load_depth(gt_path, cfg.depth_scale)
        d_gt = _resize_depth_to(d_gt, H, W)

        mask = (d_gt > 0) & (d_pred > 0) & (alpha > 1e-3)
        if not mask.any():
            print(f"  [{i}] mask empty")
            continue
        err = d_pred - d_gt
        rmse = float(np.sqrt((err[mask] ** 2).mean()))
        rel = float(np.abs(err[mask] / d_gt[mask]).mean())
        rmses.append(rmse)
        rel_errs.append(rel)

        if i < cfg.n_visualize:
            vmin, vmax = float(d_gt[mask].min()), float(d_gt[mask].max())
            rgb_u8 = (rgb_pred * 255).astype(np.uint8)
            d_pred_vis = _colorize(d_pred, vmin, vmax)
            d_gt_vis = _colorize(d_gt, vmin, vmax)
            err_vis = _colorize(np.abs(err), 0.0, max(vmax - vmin, 1e-3) * 0.25)
            panel = np.concatenate([rgb_u8, d_pred_vis, d_gt_vis, err_vis], axis=1)
            imageio.imwrite(panels_dir / f"cmp_{i:03d}.png", panel)

    metrics_path = out_dir / "metrics_with_depth.json"
    metrics_old: dict = {}
    legacy = out_dir / "metrics.json"
    if legacy.exists():
        metrics_old = json.loads(legacy.read_text())
    out_metrics = {
        **metrics_old,
        "depth_rmse": float(np.mean(rmses)) if rmses else None,
        "depth_rel_err": float(np.mean(rel_errs)) if rel_errs else None,
        "depth_n_views": len(rmses),
    }
    metrics_path.write_text(json.dumps(out_metrics, indent=2))
    print(f"\nholdout depth RMSE: "
          f"{out_metrics['depth_rmse']:.4f} m  (over {out_metrics['depth_n_views']} views)")
    print(f"holdout depth rel err: {out_metrics['depth_rel_err']:.4f}")
    print(f"wrote: {metrics_path}")
    print(f"panels: {panels_dir}/cmp_*.png  (rgb | pred | gt | err)")


if __name__ == "__main__":
    main(tyro.cli(Config))
