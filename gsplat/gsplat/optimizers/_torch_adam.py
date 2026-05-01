# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch implementation of the fused Adam step used by `SelectiveAdam`.

The CUDA `adam` op was an in-place fused kernel with a per-row visibility
mask. This drop-in replacement preserves the same signature and semantics
using only stock torch ops, so it runs on MPS, CUDA (slower), and CPU.

CUDA reference: `gsplat/cuda/csrc/AdamCUDA.cu` (deleted in Stage 0; recoverable
via `git show pre-mps-cleanup:gsplat/cuda/csrc/AdamCUDA.cu`).
"""

from typing import Optional

import torch
from torch import Tensor


def torch_adam(
    param: Tensor,  # [N, *] in-place updated
    grad: Tensor,  # same shape as param
    exp_avg: Tensor,  # same shape, in-place updated
    exp_avg_sq: Tensor,  # same shape, in-place updated
    valid: Optional[Tensor],  # [N] mask (boolean / 0-1) or None
    lr: float,
    b1: float,
    b2: float,
    eps: float,
) -> None:
    """In-place selective Adam step.

    The first axis of `param`/`grad`/`exp_avg`/`exp_avg_sq` is the gaussian
    axis (size N). When `valid` is provided, only rows where `valid[i]` is
    truthy are updated; the remaining rows of `param` and the moment buffers
    are left untouched (matching the CUDA kernel's "visibility-mask" branch).
    """
    if valid is None:
        exp_avg.mul_(b1).add_(grad, alpha=1.0 - b1)
        exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
        denom = exp_avg_sq.sqrt().add_(eps)
        param.addcdiv_(exp_avg, denom, value=-lr)
        return

    mask = valid.to(torch.bool)
    if not mask.any():
        return

    # Slice the masked rows, run the standard Adam update, write back.
    p = param[mask]
    g = grad[mask]
    ea = exp_avg[mask]
    eas = exp_avg_sq[mask]

    ea.mul_(b1).add_(g, alpha=1.0 - b1)
    eas.mul_(b2).addcmul_(g, g, value=1.0 - b2)
    p.addcdiv_(ea, eas.sqrt().add_(eps), value=-lr)

    param[mask] = p
    exp_avg[mask] = ea
    exp_avg_sq[mask] = eas


__all__ = ["torch_adam"]
