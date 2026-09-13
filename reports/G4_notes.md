
## Reading the table

- **Common val loss** is the comparable number: identical fixed 5-color validation batches (`scripts/evaluate.py`, seed 123, 16×512 examples). "Final val loss" uses each run's own batches and is not comparable across sampler configs. "Best val loss (step)" is the minimum over the run's own eval history — the early-stopped value, which matters for the 1% and 10% runs that overfit under matched 20k-step compute.
- All runs: single seed, 20k steps, batch 512, AdamW 3e-4 cosine, EMA 0.999, cosine VP schedule with T=1000, DDIM 100 steps at evaluation. Train set = Kuler + MTurk 2014 train split with mean rating ≥ 3 (45,924 palettes). Learning-curve runs subsample palettes at random (seed 0).

## Does geometry help, and at what data scale?

**No measurable gain at any scale tested.** Feeding the §1.5 pairwise rotation invariants into the absolute network (M-geomlite) changes common val loss by −0.0008 at 100% data, +0.020 at 10%, and (early-stopped) −0.024 at 1% — all single-seed, all within the run-to-run noise the second-seed runs are meant to quantify (see addendum). Probe metrics (diversity, gamut validity, duplicate rate, scorer mean, reconstruction) are identical to within 0.01, and the two models produce nearly the same completions from the same noise seed (`experiments/grid_geometry.png`). The data-efficiency claim does not show up in the 1%/10% learning curves.

**The exact-equivariant pathway alone (M-geom) is worse** on val loss (1.104 vs 1.059) and its raw gamut validity collapses to 76% (vs 90%): a hue-rotation-equivariant denoiser cannot represent the sRGB gamut's asymmetry in Oklab, exactly as R1/R4 predicted. It still scores well after gamut mapping (scorer mean 2.78) because the pipeline maps its out-of-gamut samples back — but that mapping is doing part of the work.

**M-combined posts the lowest single-seed number** (1.054 vs 1.059), but the second M-abs seed reached 1.053, so this is within seed noise — at 1.77× the parameters and 1.6× the training time; the absolute/geometric noise-norm ratio settled at ~0.9, so neither pathway dominated. A parameter-matched M-abs (dim 176 or depth 7) is the fair comparison and has not been run; a 0.5% loss improvement at this cost does not justify the added architectural constraints.

**Recommendation:** ship M-abs (with `subset_augment`, per G3) as the generator. Keep the geometric code as a tested ablation; drop it from the deployment path. The central hypothesis (§1.1) — relative geometry provides useful parameter sharing — is not supported at this data size for this task; the 45k-palette corpus is enough for the absolute model to learn the geometry itself.

## Caveats

- Single seed per cell until the addendum below; conclusions rest on effect sizes being small, not on significance tests.
- Parameter counts are close but not matched (630k–1.45M). Compute matched by steps, not FLOPs.
- All evaluation is on Kuler-derived data and the DeepSets scorer trained on the same rating population. The fresh human study (Phase 6 package) is the real test.

## Addendum: seed variance (seed 0 / seed 1)

| Model | Common val loss | seed |Δ| | Scorer mean | Diversity | Gamut rate raw |
| --- | --- | --- | --- | --- | --- |
| mabs | 1.0593 / 1.0532 | 0.0062 | 2.773 / 2.763 | 0.207 / 0.212 | 0.905 / 0.886 |
| geomlite | 1.0585 / 1.0566 | 0.0019 | 2.763 / 2.767 | 0.208 / 0.207 | 0.901 / 0.890 |

Seed-to-seed spread in common val loss is 0.0062; the M-abs vs M-geomlite gap at 100% data is 0.0013 (two-seed means 1.0562 vs 1.0575). The 10% and 1% gaps (0.020, 0.024, opposite signs) are of the same order as this spread. Conclusion unchanged: no measurable geometry effect.
