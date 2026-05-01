# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base machinery for Metal-backed autograd functions (Phase B+).

Stages 6 onwards register `torch.autograd.Function` subclasses whose
`forward` calls a Metal kernel and whose `backward` calls either a Metal
kernel (preferred) or the existing torch reference path (transitional).

Stage 5 ships only the toy `elementwise_add` kernel, so this module is
intentionally minimal — the abstractions land alongside the kernels they
serve.
"""

from typing import Any

import torch


class MetalFunctionBase(torch.autograd.Function):
    """Base class for Metal-backed autograd functions.

    Subclasses implement `forward(ctx, *args)` and `backward(ctx, *grads)`.
    The base class doesn't add behaviour yet — it exists to make the
    inheritance chain explicit so future shared utilities (gradient checking,
    contiguous-input enforcement, debug logging) have a single home.
    """


__all__ = ["MetalFunctionBase"]
