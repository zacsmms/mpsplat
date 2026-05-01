# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
from pathlib import Path
from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer import GsplatViewer, GsplatRenderTabState


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if args.ckpt is None:
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
        ) = load_test_data(device=device, scene_grid=args.scene_grid)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()

        sh_degree = None
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)

        # batched render
        for _ in tqdm.trange(1):
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                width,
                height,
                render_mode="RGB+D",
                packed=False,
                distributed=world_size > 1,
            )
        C = render_colors.shape[0]
        assert render_colors.shape == (C, height, width, 4)
        assert render_alphas.shape == (C, height, width, 1)
        render_colors.sum().backward()

        render_rgbs = render_colors[..., 0:3]
        render_depths = render_colors[..., 3:4]
        render_depths = render_depths / render_depths.max()

        # dump batch images
        os.makedirs(args.output_dir, exist_ok=True)
        canvas = (
            torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        imageio.imsave(
            f"{args.output_dir}/render_rank{world_rank}.png",
            (canvas * 255).astype(np.uint8),
        )
    else:
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
            means.append(ckpt["means"])
            quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales.append(torch.exp(ckpt["scales"]))
            opacities.append(torch.sigmoid(ckpt["opacities"]))
            sh0.append(ckpt["sh0"])
            shN.append(ckpt["shN"])
        means = torch.cat(means, dim=0)
        quats = torch.cat(quats, dim=0)
        scales = torch.cat(scales, dim=0)
        opacities = torch.cat(opacities, dim=0)
        sh0 = torch.cat(sh0, dim=0)
        shN = torch.cat(shN, dim=0)
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
        print("Number of Gaussians:", len(means))

        # Optional: keep only the top-K highest-opacity splats. Drops the
        # cost of per-frame projection + rasterization roughly proportional
        # to the new gauss count, in exchange for some loss in fine detail.
        if args.decimate > 0 and args.decimate < len(means):
            keep = torch.topk(opacities, args.decimate).indices
            means = means[keep]
            quats = quats[keep]
            scales = scales[keep]
            opacities = opacities[keep]
            colors = colors[keep]
            print(f"Decimated to {len(means)} Gaussians (kept top {args.decimate} by opacity).")

    # Scene-transform state: applied to every camera's viewmat as a right-
    # multiply, so the splats are NEVER touched and there's no per-frame
    # tensor rewrite. Using closure-captured `scene_T` (4x4 on device).
    scene_T = torch.eye(4, device=device)
    scene_state = {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
                   "up_axis_idx": 0}

    UP_AXIS_PRESETS = [
        ("Y up (default)", torch.eye(3)),
        ("Z up", torch.tensor([[1., 0, 0], [0, 0, -1], [0, 1, 0]])),
        ("-Y up",          torch.tensor([[1., 0, 0], [0, -1, 0], [0, 0, -1]])),
        ("-Z up",          torch.tensor([[1., 0, 0], [0, 0, 1], [0, -1, 0]])),
    ]

    def _euler_to_R(yaw_deg, pitch_deg, roll_deg):
        # Apply yaw (around Y), then pitch (around X), then roll (around Z).
        ay, ap, ar = (math.radians(d) for d in (yaw_deg, pitch_deg, roll_deg))
        cy, sy = math.cos(ay), math.sin(ay)
        cp, sp = math.cos(ap), math.sin(ap)
        cr, sr = math.cos(ar), math.sin(ar)
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Rz = torch.tensor([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        return Rz @ Rx @ Ry

    def _refresh_scene_T():
        R_axis = UP_AXIS_PRESETS[scene_state["up_axis_idx"]][1]
        R_euler = _euler_to_R(scene_state["yaw_deg"], scene_state["pitch_deg"],
                              scene_state["roll_deg"])
        R = (R_euler @ R_axis).to(device)
        T = torch.eye(4, device=device)
        T[:3, :3] = R
        nonlocal scene_T
        scene_T = T

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()
        # Right-multiply by the scene transform so the splats appear as if
        # rotated into the user's chosen frame, without ever touching the
        # gauss tensors.  (R_world_to_cam @ T_scene^-1 implements
        # "render the scene as if it were rotated by T_scene".)
        viewmat = viewmat @ torch.linalg.inv(scene_T)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            with_ut=args.with_ut,
            with_eval3d=args.with_eval3d,
        )
        render_tab_state.total_gs_count = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders

    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )
    # Better defaults for M-series: cap idle render at 1024 (vs the nerfview
    # default of 2048, which is ~4x slower) and bump motion-render budget so
    # dragging looks like a real preview instead of a 60×60 mosaic.
    viewer.render_tab_state.viewer_res = args.max_res
    viewer.render_tab_state.num_view_rays_per_sec = args.rays_per_sec

    # Scene-transform GUI panel.  All controls re-rotate the scene (via
    # viewmat right-multiply) on update — splats are never touched.
    with server.gui.add_folder("Scene Transform"):
        up_axis_dd = server.gui.add_dropdown(
            "Up axis",
            options=[name for name, _ in UP_AXIS_PRESETS],
            initial_value=UP_AXIS_PRESETS[0][0],
            hint="Cycle the world's up direction. COLMAP scenes are usually Y-up.",
        )
        yaw_slider = server.gui.add_slider(
            "Yaw",  min=-180, max=180, step=1, initial_value=0,
            hint="Rotate scene around the up axis.",
        )
        pitch_slider = server.gui.add_slider(
            "Pitch", min=-180, max=180, step=1, initial_value=0,
            hint="Tilt scene forward / back.",
        )
        roll_slider = server.gui.add_slider(
            "Roll",  min=-180, max=180, step=1, initial_value=0,
            hint="Roll scene around the camera-look axis.",
        )
        reset_btn = server.gui.add_button("Reset transform")

        @up_axis_dd.on_update
        def _(_) -> None:
            scene_state["up_axis_idx"] = next(
                i for i, (n, _) in enumerate(UP_AXIS_PRESETS) if n == up_axis_dd.value
            )
            _refresh_scene_T()
            viewer.rerender(_)

        @yaw_slider.on_update
        def _(_) -> None:
            scene_state["yaw_deg"] = float(yaw_slider.value)
            _refresh_scene_T(); viewer.rerender(_)

        @pitch_slider.on_update
        def _(_) -> None:
            scene_state["pitch_deg"] = float(pitch_slider.value)
            _refresh_scene_T(); viewer.rerender(_)

        @roll_slider.on_update
        def _(_) -> None:
            scene_state["roll_deg"] = float(roll_slider.value)
            _refresh_scene_T(); viewer.rerender(_)

        @reset_btn.on_click
        def _(_) -> None:
            scene_state.update(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                               up_axis_idx=0)
            up_axis_dd.value = UP_AXIS_PRESETS[0][0]
            yaw_slider.value = 0; pitch_slider.value = 0; roll_slider.value = 0
            _refresh_scene_T(); viewer.rerender(_)

    print(
        f"Viewer running on http://localhost:{args.port}"
        f"  (max_res={args.max_res}, rays/s={args.rays_per_sec:_}, "
        f"decimate={args.decimate or 'off'})"
    )
    print("Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
        --output_dir results/garden/ \
        --port 8082
    
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --output_dir results/garden/ \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")
    parser.add_argument(
        "--max_res", type=int, default=1024,
        help="max render resolution (idle). Lower = smoother orbit on M-series.",
    )
    parser.add_argument(
        "--rays_per_sec", type=int, default=2_000_000,
        help="rays/sec budget for motion-mode render. Higher = sharper while dragging.",
    )
    parser.add_argument(
        "--decimate", type=int, default=0,
        help="Keep only top-K highest-opacity splats at load time. 0 = keep all. "
             "60k → 15k typically gives ~4x viewer speedup with mild quality loss.",
    )
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
