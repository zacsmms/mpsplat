# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests: 3DGS, 2DGS, and 3DGUT rasterization run on MPS / CPU."""

import warnings

import pytest
import torch

from gsplat import rasterization, rasterization_2dgs


def _make_inputs(dev, N=64):
    g = torch.Generator(device="cpu").manual_seed(0)
    means = (torch.randn(N, 3, generator=g) * 0.5).to(dev)
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(N, 4).contiguous().to(dev)
    scales = torch.full((N, 3), 0.05).to(dev)
    opacities = torch.full((N,), 0.7).to(dev)
    colors = torch.rand(N, 3, generator=g).to(dev)
    viewmats = torch.eye(4)[None].clone()
    viewmats[..., :3, 3] = torch.tensor([0.0, 0.0, 2.0])
    viewmats = viewmats.to(dev)
    Ks = torch.tensor([[100.0, 0, 64], [0, 100, 64], [0, 0, 1.0]])[None].to(dev)
    return means, quats, scales, opacities, colors, viewmats, Ks


def _assert_image(img, alpha, dev_name):
    H, W = 128, 128
    assert tuple(img.shape) == (1, H, W, 3), f"shape={tuple(img.shape)}"
    assert img.device.type == dev_name, f"device={img.device}"
    nz = (img > 0.01).sum().item()
    assert nz > 1000, f"only {nz} bright pixels"
    a_max = float(alpha.max())
    assert a_max > 0.5, f"alpha_max={a_max:.3f} too low"


def test_3dgs_smoke(device):
    args = _make_inputs(device)
    img, alpha, _ = rasterization(*args, 128, 128, packed=False)
    _assert_image(img, alpha, device)


def test_2dgs_smoke(device):
    args = _make_inputs(device)
    out = rasterization_2dgs(*args, 128, 128, packed=False)
    _assert_image(out[0], out[1], device)


def test_3dgut_smoke(device):
    args = _make_inputs(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img, alpha, _ = rasterization(*args, 128, 128, packed=False, with_ut=True)
    _assert_image(img, alpha, device)
