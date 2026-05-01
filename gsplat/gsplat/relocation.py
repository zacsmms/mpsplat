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

import math
from typing import Tuple

import torch
from torch import Tensor


def compute_relocation(
    opacities: Tensor,  # [N]
    scales: Tensor,  # [N, 3]
    ratios: Tensor,  # [N]
    binoms: Tensor,  # [n_max, n_max]
) -> Tuple[Tensor, Tensor]:
    """Compute new Gaussians from a set of old Gaussians.

    This function interprets the Gaussians as samples from a likelihood distribution.
    It uses the old opacities and scales to compute the new opacities and scales.
    This is an implementation of the paper
    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/pdf/2404.09591>`_,

    Args:
        opacities: The opacities of the Gaussians. [N]
        scales: The scales of the Gaussians. [N, 3]
        ratios: The relative frequencies for each of the Gaussians. [N]
        binoms: Precomputed lookup table for binomial coefficients used in
          Equation 9 in the paper. [n_max, n_max]

    Returns:
        A tuple:

        **new_opacities**: The opacities of the new Gaussians. [N]
        **new_scales**: The scales of the Gaussians. [N, 3]
    """

    N = opacities.shape[0]
    n_max, _ = binoms.shape
    assert scales.shape == (N, 3), scales.shape
    assert ratios.shape == (N,), ratios.shape
    opacities = opacities.contiguous()
    scales = scales.contiguous()
    ratios.clamp_(min=1, max=n_max)
    ratios_int = ratios.long().contiguous()

    # Pure-PyTorch port of `gsplat/cuda/csrc/RelocationCUDA.cu`. The CUDA kernel
    # walks each Gaussian's split count `n_idx` and accumulates a normalising
    # `denom_sum` over an O(n^2) double sum. We vectorise across N and iterate
    # in Python over the (i, k) pairs up to `n_max`, masking out rows whose
    # `n_idx` hasn't reached `i` yet.
    new_opacities = 1.0 - (1.0 - opacities) ** (1.0 / ratios_int.to(opacities.dtype))
    denom_sum = torch.zeros_like(opacities)
    for i in range(1, n_max + 1):
        active = (ratios_int >= i).to(opacities.dtype)
        for k in range(i):
            bin_coeff = binoms[i - 1, k]
            sign = (-1.0) ** k
            term = sign / math.sqrt(k + 1) * new_opacities ** (k + 1)
            denom_sum = denom_sum + active * bin_coeff * term
    coeff = opacities / denom_sum.clamp(min=1e-20)
    new_scales = coeff[:, None] * scales
    return new_opacities, new_scales
