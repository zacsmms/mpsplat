# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a tiny synthetic COLMAP-format dataset for Stage-3 smoke tests.

The dataset is what `examples/datasets/colmap.py:Parser` expects: a folder
containing `images/`, `sparse/0/{cameras.txt,images.txt,points3D.txt}` in
COLMAP text format, plus a stub `images_<factor>` directory if you ask the
trainer for a downsampled factor.

Usage:
    python tests/make_tiny_colmap.py /tmp/tiny_colmap
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from PIL import Image


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j = (i + 1) % 3
        k = (i + 2) % 3
        s = 2.0 * math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        w_xyz = [0.0, 0.0, 0.0]
        w = (R[k, j] - R[j, k]) / s
        w_xyz[i] = 0.25 * s
        w_xyz[j] = (R[j, i] + R[i, j]) / s
        w_xyz[k] = (R[k, i] + R[i, k]) / s
        x, y, z = w_xyz
    return np.array([w, x, y, z], dtype=np.float64)


def _orbit_pose(theta: float, radius: float = 3.0) -> np.ndarray:
    """4x4 world->camera matrix; camera at radius, looking at origin."""
    eye = np.array([radius * math.sin(theta), 0.0, radius * math.cos(theta)])
    forward = -eye / np.linalg.norm(eye)
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R = np.stack([right, up, forward], axis=0)  # 3x3
    t = -R @ eye
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def main(out_dir: str, n_views: int = 8, side: int = 64) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, f"images_{1}"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sparse", "0"), exist_ok=True)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    # Camera intrinsics shared across all images.
    fx = fy = side * 0.8
    cx, cy = side / 2.0, side / 2.0

    # Generate ground-truth images: a coloured gradient plus a moving spot,
    # so each view differs and there's something for the trainer to learn.
    img_paths = []
    poses = []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        w2c = _orbit_pose(theta)
        poses.append(w2c)

        # Procedural image: angular hue + central highlight that shifts.
        ys, xs = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
        nx = (xs - cx) / cx
        ny = (ys - cy) / cy
        dist = np.sqrt(nx ** 2 + ny ** 2)
        hue = (theta / (2 * math.pi))
        r = (0.4 + 0.5 * np.cos(2 * math.pi * (hue + 0.0))) * np.exp(-dist)
        g = (0.4 + 0.5 * np.cos(2 * math.pi * (hue + 0.33))) * np.exp(-dist)
        b = (0.4 + 0.5 * np.cos(2 * math.pi * (hue + 0.66))) * np.exp(-dist)
        img = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
        img_u8 = (img * 255).astype(np.uint8)
        name = f"img_{i:03d}.png"
        Image.fromarray(img_u8).save(os.path.join(out_dir, "images", name))
        Image.fromarray(img_u8).save(os.path.join(out_dir, f"images_1", name))
        img_paths.append(name)

    # Write cameras.txt — one shared PINHOLE camera.
    with open(os.path.join(out_dir, "sparse", "0", "cameras.txt"), "w") as f:
        f.write(
            "# Camera list with one line per camera:\n"
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        )
        f.write(f"1 PINHOLE {side} {side} {fx} {fy} {cx} {cy}\n")

    # Write images.txt
    with open(os.path.join(out_dir, "sparse", "0", "images.txt"), "w") as f:
        f.write(
            "# Image list with two lines per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        )
        for i, w2c in enumerate(poses):
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qw, qx, qy, qz = _quat_from_R(R)
            f.write(
                f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {img_paths[i]}\n"
            )
            # POINTS2D line: pycolmap's iterator terminates on truly-empty
            # lines, so write a single dummy 2D point with an invalid 3D id.
            f.write("0 0 18446744073709551615\n")

    # Write points3D.txt
    n_pts = 200
    pts = rng.uniform(-0.5, 0.5, size=(n_pts, 3))
    cols = (rng.uniform(0, 255, size=(n_pts, 3))).astype(int)
    with open(os.path.join(out_dir, "sparse", "0", "points3D.txt"), "w") as f:
        f.write(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        for i in range(n_pts):
            x, y, z = pts[i]
            r, g, b = cols[i]
            f.write(f"{i + 1} {x} {y} {z} {r} {g} {b} 1.0\n")

    print(f"OK: wrote {n_views} views + {n_pts} points to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny_colmap")
