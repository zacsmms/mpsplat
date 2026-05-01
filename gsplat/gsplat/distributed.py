# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-process drop-in for the deleted distributed entry point.

The original module ran trainers under `torch.multiprocessing.spawn` for
multi-GPU NCCL collectives. The MPS port has no NCCL and runs single-process,
so `cli()` here just calls `fn(local_rank=0, world_rank=0, world_size=1, args)`
once. This keeps `from gsplat.distributed import cli` callers working.
"""

from typing import Any, Callable


def cli(fn: Callable, args: Any, verbose: bool = False) -> bool:
    """Run `fn` once with rank 0 / world size 1.

    Multi-process distributed training (NCCL / multi-GPU) was removed with
    the CUDA backend; this stub keeps the single-process path identical to
    the upstream API.
    """
    if verbose:
        print("[mpsplat] running single-process (distributed support removed)")
    fn(0, 0, 1, args)
    return True


__all__ = ["cli"]
