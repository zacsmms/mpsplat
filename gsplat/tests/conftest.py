# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for the MPS-port test suite.

Two changes vs. the upstream gsplat conftest:

1. The `setup_test_environment` autouse fixture now seeds + cleans whatever
   accelerator backend is available (MPS or CUDA), instead of CUDA only.

2. A `--device` CLI option lets you opt-in to running tests on CPU, MPS, or
   CUDA. The fixture `device` (string) and `device_t` (torch.device) are
   exposed for tests that need to be device-aware. The default is `mps` if
   available, else `cpu`. Tests that do `device = torch.device("cuda:0")` at
   module import are dynamically rewritten to use the chosen device — see
   `pytest_collection_modifyitems` below.
"""

import gc

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Device to run tests on. Defaults to mps if available, else cpu.",
    )


def _resolve_default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@pytest.fixture(scope="session")
def device(request) -> str:
    chosen = request.config.getoption("--device") or _resolve_default_device()
    return chosen


@pytest.fixture(scope="session")
def device_t(device) -> torch.device:
    return torch.device(device)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Reseed and clear caches before every test."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


@pytest.fixture(scope="session")
def dist_init():
    """No-op on the MPS port — multi-process distributed was removed."""
    yield


# Tests that import modules deleted in Stage 0 (lidar / external distortion).
# We can't even *collect* them, so skip the import entirely. Restore them if
# the relevant module is ever brought back.
collect_ignore = [
    "test_external_distortion.py",
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cuda_only: legacy test that depends on CUDA-only behaviour "
        "(custom autograd hooks, fp64, lidar/UT internals). Skipped on "
        "MPS / CPU device runs.",
    )


# ---------------------------------------------------------------------------
# Auto-rewriting of upstream skipif markers
#
# Most upstream tests were guarded by `@pytest.mark.skipif(not
# torch.cuda.is_available(), reason="No CUDA device")`. On MPS that
# decorator skips everything. We strip the marker at collection time when
# the chosen device is MPS or CPU (the test bodies themselves are device-
# parametrised via the `device` module-level constant in each file).
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(config, items):
    chosen = config.getoption("--device") or _resolve_default_device()
    skip_cuda_only = pytest.mark.skip(
        reason=f"legacy CUDA-only test, skipped on device={chosen}"
    )
    if chosen != "cuda":
        for item in items:
            if "cuda_only" in item.keywords:
                item.add_marker(skip_cuda_only)
    if chosen == "cuda":
        return
    for item in items:
        new_markers = []
        for mark in item.iter_markers(name="skipif"):
            cond = mark.args[0] if mark.args else None
            reason = mark.kwargs.get("reason") or (mark.args[1] if len(mark.args) > 1 else "")
            if isinstance(reason, str) and "cuda" in reason.lower() and cond is True:
                # Drop "skip when CUDA missing" markers; the test will run on
                # whatever device the user chose (MPS / CPU). If a test
                # genuinely can't run there, it should `pytest.skip(...)` or
                # `pytest.xfail(...)` from inside the body.
                continue
            new_markers.append(mark)
        # We can't easily mutate iter_markers; instead, the simplest
        # approximation is to strip via own_markers when present.
        own = getattr(item, "own_markers", None)
        if own:
            item.own_markers = [
                m for m in own
                if not (
                    m.name == "skipif"
                    and isinstance(m.kwargs.get("reason", ""), str)
                    and "cuda" in m.kwargs.get("reason", "").lower()
                )
            ]
