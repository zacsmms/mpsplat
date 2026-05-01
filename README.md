# mpsplat

**A Mac (Apple Silicon / MPS) port of [gsplat](https://github.com/nerfstudio-project/gsplat).**

![sample mpsplat render](media/orb_render.png)

A flythrough of the same scene, rendered after a real mpsplat training run on an M-series Mac:

https://github.com/zacsmms/mpsplat/raw/main/media/orb_flythrough.mp4


This repository is a fork of the excellent [gsplat](https://github.com/nerfstudio-project/gsplat)
library by **Kanazawa AI Research (KAIR)** at UC Berkeley and
collaborators (NVIDIA, ShanghaiTech University, Amazon, Meta, IIIT,
LumaAI, SpectacularAI, Aalto University, CMU, and others). The
original library is CUDA-only and does not run on Macs. **mpsplat
exists for one reason: to make gsplat work on Apple Silicon, where most
of us actually carry a GPU around.**

> mpsplat is **not** an attempt to replace gsplat or claim any of its
> design or research credit. All of the math, the algorithms, the
> training strategies, the densification heuristics, the tests, the
> examples — those are gsplat's. What this fork does is rip out the
> CUDA build pipeline and reimplement the hot kernels in Metal Shading
> Language so the same library runs on `device="mps"`. Every contributor
> to upstream gsplat is also a contributor here, by descent.

If you're on a CUDA box, **please use upstream
[gsplat](https://github.com/nerfstudio-project/gsplat)** — it's faster,
better tested at scale, and actively maintained. mpsplat is for the case
where the only GPU you have is the one in your MacBook.

## What changed from upstream

| | upstream `gsplat` | `mpsplat` |
|---|---|---|
| Target hardware | NVIDIA CUDA | Apple Silicon (M-series) via PyTorch MPS |
| Build step | nvcc + ninja JIT-compile of `csrc/*.cu` | `torch.mps.compile_shader` of `_shaders/*.metal` (no C++ build) |
| Distributed training | NCCL multi-GPU | single-process (NCCL doesn't exist on macOS) |
| Lidar / external windshield distortion | supported | removed (NCore-specific, out of scope here) |
| Public API surface | `import gsplat; rasterization(...)` etc. | **identical** — same calls, same shapes, same returns |

The kernel inventory we ported (or kept as fast pure-PyTorch fallbacks):

- 3DGS rasterizer (forward + backward) — native Metal
- 3DGS tile intersection — native Metal
- 3DGS EWA projection (forward + backward) — native Metal
- Spherical harmonics (forward + backward) — native Metal
- 2DGS rasterizer (forward + backward) — native Metal
- 2DGS projection (forward) — native Metal; backward via torch autograd
- 3DGUT UT projection (pinhole / pinhole+OpenCV-distortion / OpenCV-fisheye) — native Metal
- 3DGUT eval3d (forward + backward, pinhole / fisheye) — native Metal

The Python package is still imported as `gsplat`, so any code that
already targets the upstream API runs unchanged on a Mac after `pip
install -e .` of this fork. We renamed only the *project*, not the
*package*.

## Install

You'll need `uv` (or vanilla pip), Python 3.13, and PyTorch ≥ 2.3 with MPS:

```bash
brew install uv ffmpeg colmap            # colmap & ffmpeg are for capture pipeline
git clone https://github.com/zacsmms/mpsplat.git
cd mpsplat
uv venv --python 3.13
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -r examples/requirements.txt
uv pip install --python .venv/bin/python "imageio[ffmpeg]"
```

There is no compile step — the C++/CUDA extension was deleted in
favour of `torch.mps.compile_shader`, which JIT-compiles `.metal` files
the first time each kernel is invoked.

For the end-to-end "phone footage → splat → viewer" walk-through, see
the [`QUICKSTART.md`](QUICKSTART.md).

## Status

Working on M-series Macs (M1 / M2 / M3 / M4). The full pytest suite is
35 passing / 27 legacy CUDA tests cleanly skipped. End-to-end image
fitting on a small synthetic problem runs ~166× faster than the pure-
PyTorch baseline. Real-scene training is closer to 5–15× faster than
the pure-PyTorch baseline depending on the scene; the gap to upstream
gsplat on equivalent NVIDIA hardware is roughly 2–4× wall-clock.

## License & attribution

Apache 2.0 — same as upstream gsplat. The `LICENSE` file is unchanged.
The SPDX headers in modified files preserve upstream's copyright line
and add NVIDIA's (most of the kernel reimplementations were derived
from NVIDIA's CUDA reference) and the date of this port.

If you publish work that uses mpsplat, please cite the upstream gsplat
paper:

```bibtex
@article{ye2025gsplat,
  title={gsplat: An open-source library for Gaussian splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen,
          Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and
          Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and
          Angjoo Kanazawa},
  journal={Journal of Machine Learning Research},
  volume={26}, number={34}, pages={1--17}, year={2025}
}
```

## Upstream news (preserved here for context)

[Jan 2026] [PPISP](https://research.nvidia.com/labs/sil/projects/ppisp/) integrated upstream as an alternative to bilateral grid for compensating training views.

[May 2025] Arbitrary batching (multiple scenes × multiple viewpoints) supported upstream — see [docs/batch.md](docs/batch.md).

[April 2025] [NVIDIA 3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/) integrated upstream — [docs/3dgut.md](docs/3dgut.md).

mpsplat tracks features as the porting work allows; PPISP and arbitrary
batching are present, full ftheta + rolling-shutter routes through the
torch reference rather than Metal kernels.

## Contributing

Bug reports, additional Metal kernels, or upstream-tracking PRs welcome.
Please open issues on the upstream gsplat repo for algorithmic
questions; this fork only owns the MPS-specific code paths.
