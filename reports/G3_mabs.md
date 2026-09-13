# G3 — Diffusion models vs baselines

| Run | Params | Train palettes | Train s | Final val loss | diversity | gamut_rate_raw | duplicate_rate | score | recon_min_over_K | recon_mean_over_K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline:colorwheel | — | — | — | — | 0.153 | 1.000 | 0.034 | 2.721 | 0.169 | 0.239 |
| baseline:retrieval | — | — | — | — | 0.228 | 1.000 | 0.003 | 2.682 | 0.125 | 0.229 |
| baseline:kde | — | — | — | — | 0.253 | 0.792 | 0.020 | 2.702 | 0.132 | 0.246 |
| mabs | 819,459 | 45,924 | 897.9 | 1.0753 | 0.207 | 0.905 | 0.023 | 2.773 | 0.120 | 0.248 |
| mabs_subset | 819,459 | 45,924 | 999.2 | 1.1279 | 0.210 | 0.893 | 0.017 | 2.761 | 0.121 | 0.250 |

## Notes

- **Final val loss** in the table is each run's own validation batches; `mabs_subset` draws subset-augmented (2–4 color) val examples, so it is not comparable. On *common* 5-color val batches (`scripts/evaluate.py`, seed 123): mabs 1.0593, mabs_subset 1.0684. Per target count m=1..5: mabs 1.007 / 1.036 / 1.058 / 1.099 / 1.158; mabs_subset 1.020 / 1.044 / 1.070 / 1.106 / 1.159. Subset augmentation costs ~0.01 in 5-color loss and buys explicit support for n+m ∈ {2,3,4}, which the 5-color corpus otherwise never shows the model.
- On the 22 probe contexts (which include n+m = 3 and 4), the two runs are within noise on every metric. Both beat retrieval on scorer mean (2.77 vs 2.68) and on held-out reconstruction (0.120 vs 0.125, diagnostic only) while producing ~10% fewer duplicates than colorwheel; diversity sits between colorwheel and retrieval.
- Raw gamut validity is ~90%: one in ten sampled palettes has at least one out-of-gamut color before mapping. The Phase 5 pipeline handles this by rejection (32 oversamples, 1 round sufficed on the sage+brown probe: 32/32 in gamut, 0 duplicates).
- Learning curves (1%, 10%) are in the G4 table.
- Recommendation for G3: proceed with M-abs as the reference; keep `subset_augment` on for the deployed generator (decision 4 in Part 5, pending owner confirmation), since the target use case is 3–4 color palettes.
- Qualitative grid: `experiments/grid_mabs.png` (sage+brown → tans, ochres, dark teal; navy → muted blues and warm accents; magenta+lime → cyan/orange accents rather than neutrals).
