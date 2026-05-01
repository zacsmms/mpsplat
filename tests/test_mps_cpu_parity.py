# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests: identical inputs render identical images on CPU vs MPS.

These tests only run when MPS is available — on a CPU-only machine they're
self-trivial and would just compare CPU to itself.
"""

import warnings

import pytest
import torch

from gsplat import rasterization, rasterization_2dgs

_HAS_MPS = torch.backends.mps.is_available()


def _inputs(dev, N=64):
    g = torch.Generator(device="cpu").manual_seed(42)
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


def _r3d(dev):
    img, alpha, _ = rasterization(*_inputs(dev), 128, 128, packed=False)
    return img.detach().cpu(), alpha.detach().cpu()


def _r2d(dev):
    out = rasterization_2dgs(*_inputs(dev), 128, 128, packed=False)
    return out[0].detach().cpu(), out[1].detach().cpu()


def _rut(dev):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img, alpha, _ = rasterization(
            *_inputs(dev), 128, 128, packed=False, with_ut=True
        )
    return img.detach().cpu(), alpha.detach().cpu()


@pytest.mark.skipif(not _HAS_MPS, reason="MPS not available")
def test_parity_3dgs():
    # MPS path uses Metal projection + isect + rasterize; CPU path uses
    # the torch references. Small per-stage fp32 drift (1e-7 to 1e-5) at
    # the projection compounds through the rasterizer, putting the final
    # image diff in the low-1e-3 range. The absolute image values are
    # 0..1 so this is still visually identical.
    a_cpu = _r3d("cpu")
    a_mps = _r3d("mps")
    assert (a_cpu[0] - a_mps[0]).abs().max().item() < 5e-3
    assert (a_cpu[1] - a_mps[1]).abs().max().item() < 5e-3


@pytest.mark.skipif(not _HAS_MPS, reason="MPS not available")
def test_parity_2dgs():
    # 2DGS still uses the torch projection reference on MPS (no Metal
    # kernel yet for 2DGS — Stage 11), so this can stay tight.
    a_cpu = _r2d("cpu")
    a_mps = _r2d("mps")
    assert (a_cpu[0] - a_mps[0]).abs().max().item() < 1e-4
    assert (a_cpu[1] - a_mps[1]).abs().max().item() < 1e-4


@pytest.mark.skipif(not _HAS_MPS, reason="MPS not available")
def test_parity_3dgut():
    a_cpu = _rut("cpu")
    a_mps = _rut("mps")
    # UT uses torch.linalg.det / inv on MPS with slightly larger drift.
    assert (a_cpu[0] - a_mps[0]).abs().max().item() < 1e-3
    assert (a_cpu[1] - a_mps[1]).abs().max().item() < 1e-3
