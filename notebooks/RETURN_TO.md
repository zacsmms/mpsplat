# RETURN_TO — TSDF-prior experiment (paused 2026-05-12)

## TL;DR

One DTU scene tested (scan24, half-res, no densification, upstream gsplat 1.5.3 on Colab T4). Four training runs, sparse depth eval against the 2DGS bundle's `.pt` files.

**Headline finding: the primitive TSDF-prior *hurts* sparse depth RMSE on scan24, monotonically with λ:**

| run         | Δ vs baseline_s42 (m) | ratio to seed-noise floor |
|-------------|-----------------------|---------------------------|
| tsdf λ=0.025 | worse                 | **1.5×** — borderline harm |
| tsdf λ=0.05  | worse                 | **3.72×** — clear harm     |

Last commit at pause: `b64c9d6` (sparse depth eval rewrite).

## The hypothesis to pursue

TSDF-prior is *reinforcing* the argmax bias rather than correcting it.

- Single-view argmax is biased toward the Gaussian centre, not the true surface.
- If that bias is **correlated across views** (likely on tilted / specular surfaces — a splat tilted toward the camera looks similar from many viewpoints), TSDF averaging yields a *biased* consensus.
- L_tsdf then pulls per-view predictions toward that biased consensus, compounding the error rather than smoothing random noise.

A confirmed negative result would itself be a contribution ("naïve self-distillation hurts when argmax bias is view-correlated → motivates a confidence-weighted alternative"). But we have to rule out engineering causes first.

## Knobs to test (priority order)

### 1. L_n conflict — most likely confound
Normal-consistency (L_n) and L_tsdf both turn on at step 15k. They may be optimising toward different surfaces.
- **Run:** `--normal_lambda 0 --tsdf_lambda 0.025`.
- **If TSDF helps with L_n off:** the two losses were fighting, not a real prior failure.
- **Cost:** ~15 min on A100, one extra run.

### 2. TSDF resolution + truncation
Defaults are 128³ voxels, truncation = 6× voxel size ≈ 36 cm on this scene (~8 m span). Big "voting zone" — lots of room for noisy depths near a surface.
- **Run:** `--tsdf_resolution 256` + truncation 2–3× voxel.
- Knobs are in `methods/tsdf_prior.TSDFConfig` (`resolution`, `truncation`); experiment script's `--tsdf_truncation_factor` controls the factor.
- **If RMSE swings:** result is partly a parameterisation artifact.

### 3. Refresh cadence
First refresh at step 17k = first 15k iters of biased depths are what we freeze as the target.
- **Run:** refresh every 500 iters in a warmup phase rather than every 2k post-15k.
- Already a flag (`--tsdf_refresh_every`). Default 2000.

### 4. Refresh view count
Currently 24 random training views per refresh. More views = lower-variance consensus but slower.
- **Run:** bump to all 42 train views (`--tsdf_max_refresh_views 42`). ~3× slower per refresh but cleaner target.

### 5. Confidence weighting — the real fix candidate
Each splat currently votes equally in the TSDF. Weight each contribution by `v'ᵀ Σ'⁻¹ v'` (sharp peak = trustworthy argmax). Fat splats along the view direction get less say.
- **Code change** in `methods/tsdf_prior.py::TSDFVolume.integrate` — pass per-splat confidence into the running average.
- This is the variant that would go in the paper if the negative result holds.

### 6. Smaller λ
Monotonic harm at λ ∈ {0.025, 0.05}. Check the harm scales down all the way, or whether there's a non-zero crossover.
- **Run:** λ ∈ {0.001, 0.005, 0.010}.

### 7. With densification
Disabled because upstream gsplat 1.5.3's DefaultStrategy crashes on the meta dict. With densification on, splats can move/grow to satisfy the prior — prior might be bad on fixed splat set but helpful when splats are adaptive.
- **Prereq:** fix DefaultStrategy compatibility (pin to a gsplat version where shapes match, or write a minimal in-house densifier).
- Last try at pinning gsplat==1.4.0 silently kept 1.5.3 (no CUDA wheels for Colab's torch combo).

## DTU scenes to add

Currently scan24 only. Pattern across scenes is what makes the argument; one scene is not enough.

Recommended next set:

| scene    | character                                   | role |
|----------|---------------------------------------------|------|
| scan65   | clean, simple object, smooth surfaces       | easy case — prior should help if anywhere |
| scan83   | textured planar (clay bricks)               | textured planar stresses argmax bias |
| scan105  | bunny — mixed concave/convex                | typical detail test |
| scan114  | reflective + thin structures                | hard case — surface poorly defined |
| scan122  | jeweled stones, complex specular            | hard case — view-correlated bias maximal |

If the prior **hurts on medium (24)**, **helps on easy (65)**, **devastates on hard (114)** — that's a real story about *where* the bias is correlated.

## Code state (what's on `main`)

| file                                          | purpose                                                       |
|-----------------------------------------------|---------------------------------------------------------------|
| `methods/rade_gs.py`                          | RaDe-GS rasterizer; auto-detects upstream vs. fork            |
| `methods/tsdf_prior.py`                       | `TSDFVolume`, `tsdf_prior_loss`, `refresh_tsdf_from_views`    |
| `examples/experiment_tsdf_prior.py`           | trainer; flags: `--seed`, `--methods`, `--tsdf_lambda`, `--no-densify`, etc. |
| `examples/eval_with_depth.py`                 | sparse depth eval against `depths/*.pt`                       |
| `notebooks/experiment_tsdf_prior_colab.ipynb` | 4-run experiment with auto-verdict cell                       |

## Quick resume

**On Colab (alt account, fresh runtime):**

1. Open `https://colab.research.google.com/github/zacsmms/mpsplat/blob/main/notebooks/experiment_tsdf_prior_colab.ipynb?authuser=1`
2. Edit cell 2: `SCENE = "scan65"` (or whichever).
3. Run top to bottom. Existing checkpoints under `/content/drive/MyDrive/mpsplat_results/<scene>/` are skipped — iteration is cheap.

**For knob tests**, extend the `RUNS` list in cell 2. Example:

```python
RUNS += [
    {"name": "tsdf_l0.025_no_ln",      "method": "tsdf_prior", "seed": 42, "tsdf_lambda": 0.025, "extra": ["--normal_lambda", "0"]},
    {"name": "tsdf_l0.025_lowtrunc",   "method": "tsdf_prior", "seed": 42, "tsdf_lambda": 0.025, "extra": ["--tsdf_truncation_factor", "3.0"]},
    {"name": "tsdf_l0.025_refresh500", "method": "tsdf_prior", "seed": 42, "tsdf_lambda": 0.025, "extra": ["--tsdf_refresh_every", "500"]},
]
```

`train()` in cell 10 needs a one-liner edit to append `r.get("extra", [])` to its `cmd`. (Add this if the knob-test branch becomes the main path.)

## What to do tomorrow, in order

1. **Look at the depth panels for `tsdf_l0.05` vs `baseline_s42`** under `mpsplat_results/scan24/*/tsdf_prior/depth_panels/`. If the sparse error dots are red/blue in *correlated* locations across the two runs → bias is real and view-correlated → hypothesis confirmed. If errors are in *new* locations → it's engineering.
2. **Knob 1 (L_n ablation)** — one extra training run. Single biggest confound to rule out.
3. **Add scan65 + scan114** — broaden before publishing the finding either way.
4. Knobs 2–3 (resolution, refresh) if 1 didn't explain it.
5. Knob 5 (confidence weighting) — implement and re-run. The real "fix" candidate.
6. Knob 7 (densification) — only worth pursuing once knobs 1–5 are settled.

## Loose ends in the codebase

- `experiment_tsdf_prior.py::evaluate()` still tries `item["depth"]` — vestigial; the COLMAP `Dataset` doesn't expose it. Harmless but confusing. Remove or wire up to sparse depth.
- `tsdf_prior.py::__main__` smoke test passes on MPS but hasn't been run on CUDA recently.
- The two-pass fallback in `rade_gs.py` is ~2× the forward time. If we end up doing lots of runs, consider porting `extra_signals=` to upstream gsplat (small PR).

## Useful URLs (already public)

- Repo: https://github.com/zacsmms/mpsplat
- DTU+COLMAP Drive: https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9
- Notebook (account 0): https://colab.research.google.com/github/zacsmms/mpsplat/blob/main/notebooks/experiment_tsdf_prior_colab.ipynb
- Notebook (account 1): same URL with `?authuser=1`
