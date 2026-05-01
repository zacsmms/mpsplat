# SPDX-FileCopyrightText: Copyright 2023-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import platform
import warnings

from setuptools import find_packages, setup

__version__ = None
exec(open("gsplat/version.py", "r").read())

URL = "https://github.com/nerfstudio-project/gsplat"  # upstream
# mpsplat is the Apple Silicon port of gsplat. The Python package is still
# named `gsplat` so existing downstream code (`import gsplat; from gsplat
# import rasterization`) keeps working unchanged.

if platform.system() != "Darwin":
    warnings.warn(
        "mpsplat is an Apple Silicon / MPS-only fork of gsplat. "
        f"Installing on {platform.system()} will succeed but only the CPU torch "
        "fallback path will be available; there is no MPS device on this platform."
    )

setup(
    name="mpsplat",
    version=__version__,
    description=(
        "mpsplat — an Apple Silicon (MPS / Metal) port of gsplat. "
        "Differentiable 3D Gaussian Splatting that runs on Mac."
    ),
    keywords="gaussian, splatting, mps, metal, apple-silicon, gsplat-fork",
    url=URL,
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3",
        "numpy",
        "jaxtyping",
        "rich>=12",
    ],
    extras_require={
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.3",
            "pytest-env==0.8.1",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml>=6.0.1",
            "build",
            "twine",
            "imageio>=2.37.2",
        ],
    },
    packages=find_packages(),
    include_package_data=True,
)
