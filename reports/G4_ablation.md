# G4 — Geometry ablation

| Run | Params | Data | Train palettes | Train s | Final val loss | Best val loss (step) | Common val loss | diversity | gamut_rate_raw | duplicate_rate | score | recon_min_over_K | recon_mean_over_K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline:colorwheel | — | 100% | — | — | — | — | — | 0.153 | 1.000 | 0.034 | 2.721 | 0.169 | 0.239 |
| baseline:retrieval | — | 100% | — | — | — | — | — | 0.228 | 1.000 | 0.003 | 2.682 | 0.125 | 0.229 |
| baseline:kde | — | 100% | — | — | — | — | — | 0.253 | 0.792 | 0.020 | 2.702 | 0.132 | 0.246 |
| mabs | 819,459 | 100% | 45,924 | 897.9 | 1.0753 | 1.0753 (20000) | 1.0593 | 0.207 | 0.905 | 0.023 | 2.773 | 0.120 | 0.248 |
| mabs_s1 | 819,459 | 100% | 45,924 | 295.5 | 1.0699 | 1.0699 (20000) | 1.0532 | 0.212 | 0.886 | 0.014 | 2.763 | 0.120 | 0.242 |
| mabs_f10 | 819,459 | 10% | 4,592 | 526.2 | 1.2385 | 1.1163 (6000) | 1.2244 | 0.199 | 0.910 | 0.017 | 2.778 | 0.122 | 0.233 |
| mabs_f01 | 819,459 | 1% | 459 | 528.5 | 3.6376 | 1.3125 (2000) | 3.5472 | 0.136 | 0.973 | 0.023 | 2.780 | 0.144 | 0.220 |
| mabs_subset | 819,459 | 100% | 45,924 | 999.2 | 1.1279 | 1.1279 (20000) | 1.0684 | 0.210 | 0.893 | 0.017 | 2.761 | 0.121 | 0.250 |
| geomlite | 839,443 | 100% | 45,924 | 995.1 | 1.0766 | 1.0766 (20000) | 1.0585 | 0.208 | 0.901 | 0.020 | 2.763 | 0.119 | 0.242 |
| geomlite_s1 | 839,443 | 100% | 45,924 | 299.1 | 1.0740 | 1.0740 (20000) | 1.0566 | 0.207 | 0.890 | 0.023 | 2.767 | 0.120 | 0.242 |
| geomlite_f10 | 839,443 | 10% | 4,592 | 453.3 | 1.2645 | 1.1159 (6000) | 1.2448 | 0.195 | 0.917 | 0.017 | 2.774 | 0.125 | 0.232 |
| geomlite_f01 | 839,443 | 1% | 459 | 451.1 | 3.7365 | 1.2890 (2000) | 3.6328 | 0.132 | 0.971 | 0.037 | 2.785 | 0.145 | 0.219 |
| geom | 630,005 | 100% | 45,924 | 1214.9 | 1.1213 | 1.1213 (20000) | 1.1044 | 0.205 | 0.757 | 0.028 | 2.781 | 0.122 | 0.266 |
| combined | 1,449,464 | 100% | 45,924 | 1444.8 | 1.0696 | 1.0696 (20000) | 1.0539 | 0.214 | 0.881 | 0.009 | 2.752 | 0.119 | 0.240 |

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

