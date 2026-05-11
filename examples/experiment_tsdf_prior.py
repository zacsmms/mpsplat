"""Side-by-side experiment: RaDe-GS baseline vs. RaDe-GS + TSDF-prior.

Designed to be run on a CUDA box (RTX 4080-class) so the rasterizer matches
the paper's hardware. Trains both methods end-to-end with identical seed,
dataset split, optimizer schedule, and densification settings — the only
difference between the runs is whether ``L_tsdf`` is in the loss.

Usage::

    python examples/experiment_tsdf_prior.py \
        --data_dir data/dtu/scan24 \
        --result_dir results/tsdf_prior/scan24 \
        --data_factor 2 \
        --max_steps 30000

Outputs (under ``result_dir``):

    baseline/
        ckpts/ckpt_30000.pt
        renders/<view_id>.png        ← rendered RGB, depth, normal on holdout
        metrics.json                 ← psnr / ssim / lpips / depth_rmse
    tsdf_prior/
        (same layout)
    comparison.md                    ← side-by-side table

Notes
-----
* Defaults mirror RaDe-GS paper §4.1.1: 30k iters, normal-consistency loss
  ``L_n`` turned on at 15k. TSDF prior turns on at the same iter and
  refreshes every 2k iters.
* No camera/appearance/distortion optimization — keep the comparison clean.
* Mesh extraction (TSDF fusion + Marching Cubes for Chamfer) is intentionally
  left out: it depends on Open3D and on dataset-specific scale/units that the
  user should wire up to the official DTU/TNT eval scripts.
"""

# SPDX-FileCopyrightText: Copyright 2026 Zachary Scott-Murphy
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from torch import Tensor

# Make this script runnable from repo root: `python examples/experiment_tsdf_prior.py ...`
# We put examples/ first so that `from datasets.colmap import ...` resolves
# to our local package, not the HuggingFace `datasets` package that ships
# with Colab. (examples/datasets/__init__.py promotes it to a regular package
# so it beats HF's namespace lookup.)
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "examples"))
for _k in list(sys.modules):
    if _k == "datasets" or _k.startswith("datasets."):
        del sys.modules[_k]

from datasets.colmap import Dataset, Parser  # noqa: E402
from utils import knn, rgb_to_sh, set_random_seed  # noqa: E402
from gsplat.strategy import DefaultStrategy  # noqa: E402
from methods.rade_gs import rasterization_rade_gs  # noqa: E402
from methods.tsdf_prior import (  # noqa: E402
    TSDFConfig,
    TSDFVolume,
    refresh_tsdf_from_views,
    tsdf_prior_loss,
)

try:
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    _HAS_TORCHMETRICS = True
except Exception:
    _HAS_TORCHMETRICS = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # ---- I/O ----
    data_dir: str = "data/360_v2/garden"
    data_factor: int = 4
    result_dir: str = "results/tsdf_prior/garden"
    test_every: int = 8
    seed: int = 42

    # ---- Schedule (matches RaDe-GS §4.1.1) ----
    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [15_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [30_000])
    normal_start_iter: int = 15_000

    # ---- 3DGS init ----
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    init_type: Literal["sfm", "random"] = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0

    # ---- Densification (DefaultStrategy) ----
    densify: bool = True                       # set False if upstream gsplat strategy is incompatible
    prune_opa: float = 0.05
    grow_grad2d: float = 2e-4
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    absgrad: bool = False

    # ---- Rasterizer ----
    near_plane: float = 0.01
    far_plane: float = 1e10
    antialiased: bool = False

    # ---- Loss weights ----
    ssim_lambda: float = 0.2
    normal_lambda: float = 5e-2

    # ---- TSDF prior ----
    tsdf_lambda: float = 5e-2
    tsdf_resolution: int = 128
    tsdf_truncation_factor: float = 6.0     # × voxel_size
    tsdf_refresh_every: int = 2000
    tsdf_max_refresh_views: int = 24        # cap views per refresh — memory budget
    tsdf_bounds_padding: float = 0.2        # bbox of sfm points, scaled by 1 + this

    # ---- Methods to run ----
    methods: List[str] = field(default_factory=lambda: ["baseline", "tsdf_prior"])


# ---------------------------------------------------------------------------
# Splat init / optimizer
# ---------------------------------------------------------------------------
def _build_splats(parser: Parser, cfg: Config, device: torch.device):
    """Initialise Gaussians (means, quats, scales, opacities, sh_dc, sh_rest)."""
    if cfg.init_type == "sfm":
        points = torch.from_numpy(parser.points).float()           # (N, 3)
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float() # (N, 3)
    else:
        scene_extent = parser.scene_scale * cfg.init_extent
        points = (torch.rand((cfg.init_num_pts, 3)) - 0.5) * 2 * scene_extent
        rgbs = torch.rand((cfg.init_num_pts, 3))

    N = points.shape[0]
    # Initial scale = avg distance to 3 nearest neighbors.
    dist2 = knn(points, K=4)[:, 1:] ** 2          # (N, 3) squared distances
    avg_dist = dist2.mean(dim=-1).clamp_min(1e-8).sqrt()
    scales = torch.log(avg_dist * cfg.init_scale).unsqueeze(-1).repeat(1, 3)  # (N, 3)

    quats = torch.zeros((N, 4))
    quats[:, 0] = 1.0                              # identity rotation
    opacities = torch.logit(torch.full((N,), cfg.init_opa))

    sh_coeffs = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
    sh_coeffs[:, 0, :] = rgb_to_sh(rgbs)           # DC term carries init colour

    splats = torch.nn.ParameterDict({
        "means":     torch.nn.Parameter(points),
        "quats":     torch.nn.Parameter(quats),
        "scales":    torch.nn.Parameter(scales),
        "opacities": torch.nn.Parameter(opacities),
        "sh0":       torch.nn.Parameter(sh_coeffs[:, :1, :]),
        "shN":       torch.nn.Parameter(sh_coeffs[:, 1:, :]),
    }).to(device)

    # Per-param LRs matched to 3DGS / RaDe-GS defaults.
    scene_scale = parser.scene_scale * cfg.init_extent
    param_groups = [
        {"params": [splats["means"]],     "lr": 1.6e-4 * scene_scale, "name": "means"},
        {"params": [splats["scales"]],    "lr": 5e-3,                "name": "scales"},
        {"params": [splats["quats"]],     "lr": 1e-3,                "name": "quats"},
        {"params": [splats["opacities"]], "lr": 5e-2,                "name": "opacities"},
        {"params": [splats["sh0"]],       "lr": 2.5e-3,              "name": "sh0"},
        {"params": [splats["shN"]],       "lr": 2.5e-3 / 20,         "name": "shN"},
    ]
    optimizers = {g["name"]: torch.optim.Adam([g], lr=g["lr"], eps=1e-15) for g in param_groups}
    return splats, optimizers, scene_scale


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def _ssim(x: Tensor, y: Tensor) -> Tensor:
    """Lightweight SSIM (window 11, gaussian). x, y: (B, C, H, W) in [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    win = 11
    sigma = 1.5
    coords = torch.arange(win, device=x.device, dtype=x.dtype) - win // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, win, 1) * (g / g.sum()).view(1, 1, 1, win)
    k = g.expand(x.shape[1], 1, win, win)
    mu_x = F.conv2d(x, k, padding=win // 2, groups=x.shape[1])
    mu_y = F.conv2d(y, k, padding=win // 2, groups=x.shape[1])
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    var_x = F.conv2d(x * x, k, padding=win // 2, groups=x.shape[1]) - mu_x2
    var_y = F.conv2d(y * y, k, padding=win // 2, groups=x.shape[1]) - mu_y2
    cov_xy = F.conv2d(x * y, k, padding=win // 2, groups=x.shape[1]) - mu_xy
    s = ((2 * mu_xy + C1) * (2 * cov_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (var_x + var_y + C2))
    return s.mean()


def _normal_from_depth(depth: Tensor, K: Tensor, viewmat: Tensor) -> Tensor:
    """Compute a per-pixel surface normal from the depth map via spatial gradients.

    depth: (H, W) z-depth. Returns: (H, W, 3) unit normals in camera frame.
    """
    H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    ys = torch.arange(H, device=device, dtype=dtype) + 0.5
    xs = torch.arange(W, device=device, dtype=dtype) + 0.5
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (uu - cx) * depth / fx
    y_cam = (vv - cy) * depth / fy
    pts = torch.stack([x_cam, y_cam, depth], dim=-1)              # (H, W, 3)

    # Central differences. dx along W-axis, dy along H-axis.
    dx = torch.zeros_like(pts); dy = torch.zeros_like(pts)
    dx[:, 1:-1, :] = pts[:, 2:, :] - pts[:, :-2, :]
    dy[1:-1, :, :] = pts[2:, :, :] - pts[:-2, :, :]

    n = torch.cross(dx, dy, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # Flip to face camera (z < 0 in this codebase's frame).
    flip = torch.where(n[..., 2:3] > 0, -torch.ones_like(n[..., :1]), torch.ones_like(n[..., :1]))
    return n * flip


# ---------------------------------------------------------------------------
# Train a single method
# ---------------------------------------------------------------------------
def train_method(
    method: str,
    cfg: Config,
    device: torch.device,
    parser: Parser,
    trainset: Dataset,
    valset: Dataset,
    out_dir: Path,
) -> Dict[str, float]:
    print(f"\n=== {method} (out_dir={out_dir}) ===")
    set_random_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)
    (out_dir / "renders").mkdir(exist_ok=True)

    splats, optimizers, scene_scale = _build_splats(parser, cfg, device)
    if cfg.densify:
        strategy = DefaultStrategy(
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            absgrad=cfg.absgrad,
        )
        strategy.check_sanity(splats, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    else:
        strategy = None
        strategy_state = None
        print("  densification disabled (--no_densify)")

    # TSDF prior (only allocated for the tsdf_prior run).
    use_tsdf = method == "tsdf_prior"
    tsdf = None
    if use_tsdf:
        pts = torch.from_numpy(parser.points).float()
        bbox_min = pts.min(0).values - cfg.tsdf_bounds_padding * scene_scale
        bbox_max = pts.max(0).values + cfg.tsdf_bounds_padding * scene_scale
        voxel = ((bbox_max - bbox_min) / (cfg.tsdf_resolution - 1)).max().item()
        trunc = cfg.tsdf_truncation_factor * voxel
        tsdf = TSDFVolume(
            TSDFConfig(
                bounds_min=tuple(bbox_min.tolist()),
                bounds_max=tuple(bbox_max.tolist()),
                resolution=cfg.tsdf_resolution,
                truncation=trunc,
            ),
            device=device,
        )
        print(f"  TSDF: bounds={bbox_min.tolist()} → {bbox_max.tolist()}, "
              f"voxel={voxel:.4f}, truncation={trunc:.4f}")

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2, persistent_workers=True
    )
    train_iter = iter(train_loader)

    t0 = time.time()
    pbar = tqdm.trange(1, cfg.max_steps + 1, desc=method)
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        image = batch["image"].to(device) / 255.0          # (1, H, W, 3)
        camtoworld = batch["camtoworld"].to(device)        # (1, 4, 4)
        K = batch["K"].to(device)                          # (1, 3, 3)
        H, W = image.shape[1:3]
        viewmat = torch.linalg.inv(camtoworld)             # world → cam

        sh_deg_now = min(cfg.sh_degree, step // cfg.sh_degree_interval)
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # (N, K, 3)

        # ---- Forward ----
        out = rasterization_rade_gs(
            means=splats["means"],
            quats=splats["quats"] / splats["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W, height=H,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            sh_degree=sh_deg_now,
            rasterize_mode="antialiased" if cfg.antialiased else "classic",
        )
        rgb = out["render_colors"][..., :3]                # (1, H, W, 3)
        alpha = out["render_alphas"][..., 0]               # (1, H, W)
        depth = out["render_depths"][..., 0]               # (1, H, W)
        normals = out["render_normals"]                    # (1, H, W, 3)

        # Strategy hook: needs info dict with means2d gradients etc.
        if strategy is not None:
            strategy.step_pre_backward(splats, optimizers, strategy_state, step, out["meta"])

        # ---- Loss ----
        gt = image
        l_l1 = (rgb - gt).abs().mean()
        l_ssim = 1 - _ssim(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
        loss = (1 - cfg.ssim_lambda) * l_l1 + cfg.ssim_lambda * l_ssim

        l_n = torch.zeros((), device=device)
        l_tsdf = torch.zeros((), device=device)
        if step > cfg.normal_start_iter:
            n_depth = _normal_from_depth(depth[0], K[0], viewmat[0])  # (H, W, 3)
            mask = alpha[0].detach()                                  # (H, W)
            cos = (normals[0] * n_depth).sum(-1)                      # (H, W)
            l_n = ((1 - cos) * mask).mean()
            loss = loss + cfg.normal_lambda * l_n

            if use_tsdf:
                l_tsdf = tsdf_prior_loss(tsdf, depth[0], viewmat[0], K[0], alpha=alpha[0])
                loss = loss + cfg.tsdf_lambda * l_tsdf

        loss.backward()

        # ---- Strategy post-backward (densify / prune) ----
        if strategy is not None:
            strategy.step_post_backward(splats, optimizers, strategy_state, step, out["meta"])

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        # ---- TSDF refresh ----
        if use_tsdf and step > cfg.normal_start_iter and step % cfg.tsdf_refresh_every == 0:
            _refresh_tsdf(tsdf, splats, parser, trainset, cfg, device)

        # ---- Logging ----
        if step % 50 == 0:
            pbar.set_postfix(
                l1=f"{l_l1.item():.3f}",
                ssim=f"{l_ssim.item():.3f}",
                ln=f"{l_n.item():.3f}",
                ltsdf=f"{l_tsdf.item():.3f}",
                n=f"{splats['means'].shape[0]}",
            )

        # ---- Eval / save ----
        if step in cfg.save_steps:
            torch.save({k: v.detach().cpu() for k, v in splats.items()},
                       out_dir / f"ckpts/ckpt_{step}.pt")

    train_time = time.time() - t0
    print(f"  trained in {train_time/60:.1f} min, final N={splats['means'].shape[0]}")

    metrics = evaluate(splats, cfg, valset, device, out_dir)
    metrics["train_time_s"] = train_time
    metrics["final_num_gaussians"] = int(splats["means"].shape[0])
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


@torch.no_grad()
def _refresh_tsdf(tsdf, splats, parser, trainset, cfg: Config, device: torch.device):
    """Re-fuse a subset of training views into the TSDF volume."""
    n = min(cfg.tsdf_max_refresh_views, len(trainset))
    idxs = np.random.default_rng(0).choice(len(trainset), n, replace=False)
    depths, viewmats, Ks, alphas = [], [], [], []
    for i in idxs:
        item = trainset[int(i)]
        camtoworld = item["camtoworld"].unsqueeze(0).to(device)
        K = item["K"].unsqueeze(0).to(device)
        H, W = item["image"].shape[:2]
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
        depths.append(out["render_depths"][0, ..., 0])
        viewmats.append(viewmat[0])
        Ks.append(K[0])
        alphas.append(out["render_alphas"][0, ..., 0])
    refresh_tsdf_from_views(tsdf, depths, viewmats, Ks, alphas)


# ---------------------------------------------------------------------------
# Evaluation on held-out views
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(splats, cfg: Config, valset: Dataset, device, out_dir: Path) -> Dict[str, float]:
    if _HAS_TORCHMETRICS:
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    psnr_vals, ssim_vals, lpips_vals, depth_rmses = [], [], [], []

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
        rgb = out["render_colors"][..., :3].clamp(0, 1)            # (1, H, W, 3)
        depth = out["render_depths"][..., 0]                       # (1, H, W)
        normals = out["render_normals"]                            # (1, H, W, 3)

        if _HAS_TORCHMETRICS:
            psnr_vals.append(psnr(rgb.permute(0, 3, 1, 2), image.permute(0, 3, 1, 2)).item())
            ssim_vals.append(ssim(rgb.permute(0, 3, 1, 2), image.permute(0, 3, 1, 2)).item())
            lpips_vals.append(lpips(rgb.permute(0, 3, 1, 2), image.permute(0, 3, 1, 2)).item())
        else:
            mse = ((rgb - image) ** 2).mean().item()
            psnr_vals.append(-10 * math.log10(mse + 1e-12))

        # GT depth (if dataset exposes it under "depth"; many COLMAP scenes don't).
        if "depth" in item and item["depth"] is not None:
            d_gt = item["depth"].to(device)
            mask = (d_gt > 0) & (depth[0] > 0)
            if mask.any():
                depth_rmses.append(((depth[0] - d_gt)[mask] ** 2).mean().sqrt().item())

        # Save a quick visual.
        if i < 8:
            rgb_img = (rgb[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            d_img = depth[0].cpu().numpy()
            d_norm = (d_img - d_img.min()) / (d_img.max() - d_img.min() + 1e-8)
            d_img = (d_norm * 255).astype(np.uint8)
            n_img = ((normals[0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            imageio.imwrite(out_dir / f"renders/{i:03d}_rgb.png", rgb_img)
            imageio.imwrite(out_dir / f"renders/{i:03d}_depth.png", d_img)
            imageio.imwrite(out_dir / f"renders/{i:03d}_normal.png", n_img)

    metrics = {
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
        "lpips": float(np.mean(lpips_vals)) if lpips_vals else None,
        "depth_rmse": float(np.mean(depth_rmses)) if depth_rmses else None,
        "n_holdout": len(valset),
    }
    return metrics


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------
def write_comparison(results: Dict[str, Dict[str, float]], cfg: Config, out_dir: Path) -> None:
    md = ["# RaDe-GS vs. RaDe-GS + TSDF-prior\n",
          f"- dataset: `{cfg.data_dir}` (factor {cfg.data_factor})",
          f"- steps: {cfg.max_steps},  normal start: {cfg.normal_start_iter},  "
          f"tsdf refresh: every {cfg.tsdf_refresh_every} from step {cfg.normal_start_iter}",
          f"- seed: {cfg.seed}\n",
          "| metric | " + " | ".join(results.keys()) + " | Δ |",
          "|---|" + "---|" * (len(results) + 1)]
    keys = ["psnr", "ssim", "lpips", "depth_rmse", "train_time_s", "final_num_gaussians"]
    for k in keys:
        vals = []
        for m in results:
            v = results[m].get(k)
            vals.append(f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "—"))
        delta = ""
        if (
            len(results) >= 2
            and isinstance(results.get("baseline", {}).get(k), (int, float))
            and isinstance(results.get("tsdf_prior", {}).get(k), (int, float))
        ):
            d = results["tsdf_prior"][k] - results["baseline"][k]
            delta = f"{d:+.4f}"
        md.append(f"| {k} | " + " | ".join(vals) + f" | {delta} |")
    md.append("\n## config\n```json")
    md.append(json.dumps(asdict(cfg), indent=2))
    md.append("```")
    (out_dir / "comparison.md").write_text("\n".join(md))
    print(f"\n→ wrote {out_dir / 'comparison.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    set_random_seed(cfg.seed)

    out_root = Path(cfg.result_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    parser = Parser(
        data_dir=cfg.data_dir,
        factor=cfg.data_factor,
        normalize=True,
        test_every=cfg.test_every,
    )
    trainset = Dataset(parser, split="train")
    valset = Dataset(parser, split="val")
    print(f"  train views: {len(trainset)},  val views: {len(valset)}")

    results: Dict[str, Dict[str, float]] = {}
    for method in cfg.methods:
        results[method] = train_method(
            method, cfg, device, parser, trainset, valset, out_root / method
        )

    write_comparison(results, cfg, out_root)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
