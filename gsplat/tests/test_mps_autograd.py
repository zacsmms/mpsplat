# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autograd parity: gradients on MPS match CPU within fp32 noise.

Each public op is exercised with `requires_grad=True` inputs; we compare the
gradients produced by `.backward()` on CPU and MPS for identical inputs.

When MPS is unavailable the cross-device tests are skipped.
"""

import pytest
import torch

from gsplat import (
    fully_fused_projection,
    fully_fused_projection_2dgs,
    quat_scale_to_covar_preci,
    rasterization,
    rasterization_2dgs,
    spherical_harmonics,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="cross-device parity needs MPS"
)


def _seed_inputs(device, N=32, requires_grad=True):
    g = torch.Generator(device="cpu").manual_seed(7)
    means = (torch.randn(N, 3, generator=g) * 0.5).to(device)
    raw_quats = torch.randn(N, 4, generator=g).to(device)
    scales = torch.rand(N, 3, generator=g).to(device) * 0.1 + 0.02
    opacities = torch.full((N,), 0.7).to(device)
    colors = torch.rand(N, 3, generator=g).to(device)
    viewmats = torch.eye(4)[None].clone()
    viewmats[..., :3, 3] = torch.tensor([0.0, 0.0, 2.0])
    viewmats = viewmats.to(device)
    Ks = torch.tensor([[100.0, 0, 64], [0, 100, 64], [0, 0, 1.0]])[None].to(device)
    if requires_grad:
        for t in (means, raw_quats, scales, opacities, colors):
            t.requires_grad_(True)
    return means, raw_quats, scales, opacities, colors, viewmats, Ks


def _grad_dict(loss, **named_tensors):
    """Backprop and return {name: grad_tensor.cpu()} for inputs that asked for it."""
    leaves = [t for t in named_tensors.values() if t.requires_grad]
    grads = torch.autograd.grad(loss, leaves, retain_graph=False, allow_unused=True)
    out = {}
    i = 0
    for name, t in named_tensors.items():
        if not t.requires_grad:
            continue
        g = grads[i]
        out[name] = g.detach().cpu() if g is not None else None
        i += 1
    return out


def _compare(name, grads_cpu, grads_mps, tol):
    bad = []
    for k in grads_cpu:
        gc, gm = grads_cpu[k], grads_mps[k]
        if gc is None and gm is None:
            continue
        assert gc is not None and gm is not None, f"{name}/{k}: one-side-None grad"
        diff = (gc - gm).abs().max().item()
        if diff > tol:
            bad.append(f"{k}={diff:.2e}")
    assert not bad, f"FAIL {name}: tol={tol:.0e}  " + "  ".join(bad)


def test_quat_scale_to_covar_preci():
    def run(device):
        means, raw_q, scales, *_ = _seed_inputs(device)
        quats = raw_q / raw_q.norm(dim=-1, keepdim=True)
        covars, _ = quat_scale_to_covar_preci(
            quats, scales, compute_covar=True, compute_preci=False, triu=False
        )
        loss = covars.sum()
        return _grad_dict(loss, raw_q=raw_q, scales=scales)

    _compare("quat_scale_to_covar_preci", run("cpu"), run("mps"), tol=1e-4)


def test_fully_fused_projection():
    def run(device):
        means, raw_q, scales, _, _, viewmats, Ks = _seed_inputs(device)
        quats = raw_q / raw_q.norm(dim=-1, keepdim=True)
        radii, m2d, depths, conics, _ = fully_fused_projection(
            means, None, quats, scales, viewmats, Ks, 128, 128, packed=False
        )
        loss = m2d.sum() + conics.sum() + depths.sum()
        return _grad_dict(loss, means=means, raw_q=raw_q, scales=scales)

    _compare("fully_fused_projection", run("cpu"), run("mps"), tol=1e-3)


def test_fully_fused_projection_2dgs():
    def run(device):
        means, raw_q, scales, _, _, viewmats, Ks = _seed_inputs(device)
        quats = raw_q / raw_q.norm(dim=-1, keepdim=True)
        radii, m2d, depths, rt, normals = fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, 128, 128, packed=False
        )
        loss = m2d.sum() + rt.sum() + normals.sum()
        return _grad_dict(loss, means=means, raw_q=raw_q, scales=scales)

    _compare("fully_fused_projection_2dgs", run("cpu"), run("mps"), tol=1e-3)


def test_spherical_harmonics():
    def run(device):
        g = torch.Generator(device="cpu").manual_seed(11)
        N = 32
        dirs_raw = torch.randn(N, 3, generator=g).to(device)
        coeffs = torch.randn(N, 4, 3, generator=g).to(device) * 0.3
        dirs_raw.requires_grad_(True)
        coeffs.requires_grad_(True)
        dirs = dirs_raw / dirs_raw.norm(dim=-1, keepdim=True)
        out = spherical_harmonics(degrees_to_use=1, dirs=dirs, coeffs=coeffs)
        loss = out.sum()
        return _grad_dict(loss, dirs_raw=dirs_raw, coeffs=coeffs)

    _compare("spherical_harmonics", run("cpu"), run("mps"), tol=1e-4)


def test_rasterization_3dgs():
    def run(device):
        means, raw_q, scales, opacities, colors, viewmats, Ks = _seed_inputs(device, N=24)
        quats = raw_q / raw_q.norm(dim=-1, keepdim=True)
        ops = torch.sigmoid(opacities)
        cols = torch.sigmoid(colors)
        img, _, _ = rasterization(
            means, quats, scales, ops, cols, viewmats, Ks, 96, 96, packed=False
        )
        loss = img.mean()
        return _grad_dict(
            loss,
            means=means,
            raw_q=raw_q,
            scales=scales,
            opacities=opacities,
            colors=colors,
        )

    _compare("rasterization-3DGS", run("cpu"), run("mps"), tol=2e-3)


def test_rasterization_2dgs():
    def run(device):
        means, raw_q, scales, opacities, colors, viewmats, Ks = _seed_inputs(device, N=24)
        quats = raw_q / raw_q.norm(dim=-1, keepdim=True)
        ops = torch.sigmoid(opacities)
        cols = torch.sigmoid(colors)
        out = rasterization_2dgs(
            means, quats, scales, ops, cols, viewmats, Ks, 96, 96, packed=False
        )
        img = out[0]
        loss = img.mean()
        return _grad_dict(
            loss,
            means=means,
            raw_q=raw_q,
            scales=scales,
            opacities=opacities,
            colors=colors,
        )

    _compare("rasterization-2DGS", run("cpu"), run("mps"), tol=2e-3)


