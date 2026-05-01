# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled-shader cache + Python wrappers for Metal kernels.

Each kernel lives as a single `.metal` file under `_shaders/`. `_load(name)`
reads the file once, hands its source to `torch.mps.compile_shader`, and
caches the resulting library object so subsequent calls don't recompile.
"""

import functools
import os

import torch
from torch import Tensor

_SHADER_DIR = os.path.join(os.path.dirname(__file__), "_shaders")


@functools.lru_cache(maxsize=None)
def _load(name: str):
    """Compile and cache the shader library at `_shaders/<name>.metal`."""
    path = os.path.join(_SHADER_DIR, f"{name}.metal")
    with open(path) as f:
        source = f.read()
    return torch.mps.compile_shader(source)


def elementwise_add(a: Tensor, b: Tensor) -> Tensor:
    """Toy kernel: `out = a + b`. Sanity test for the build pipeline."""
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.device.type == "mps" and b.device.type == "mps"
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    out = torch.empty_like(a)
    lib = _load("elementwise_add")
    lib.elementwise_add(out, a.contiguous(), b.contiguous())
    return out


__all__ = ["elementwise_add"]
