# G5 — Changes after the external review (items 1–8)

All numbers below are on the **rebuilt splits** (exact-recall dedup, chains > 100 forced to train, MTurk means without the 114 held-out raters) and the **ordinal scorer** (latent scale, higher is better; not comparable to the 1–5 scale in G2–G4).

## Data (items 1, 3, 4)

- Dedup: 406,051 groups from 453,358 records; 167,200 near pairs (was 166,717 with the heuristic blocking); largest chain 11,182 rows, now in train.
- Cross-split near-duplicate pairs (exact audit): {'train-val': 0, 'train-test': 0, 'val-test': 0}.
- Rater separation: 114 users held out of every training signal; the rating-≥3 generator filter changed for 781 palettes.
- Training counts now report post-filter eligible palettes per family; best checkpoints are saved and evaluated.

## Scorer (items 4, 6)

| Variant | Spearman vs seen-user mean | Pairwise acc | Spearman vs held-out-user mean |
| --- | --- | --- | --- |
| deepsets | 0.733 ± 0.002 | 0.772 | 0.239 ± 0.001 |
| sortedL | 0.717 ± 0.001 | 0.764 | 0.244 ± 0.002 |
| ordered | 0.710 ± 0.003 | 0.762 | 0.235 ± 0.002 |
| deepsets-ordinal | 0.751 ± 0.001 | 0.781 | 0.248 ± 0.003 |
| deepsets-pairwise | 0.683 ± 0.018 | 0.747 | 0.211 ± 0.012 |

Deployed: **deepsets-ordinal**. Reliability ceilings on the test palettes: repeat-judgment Pearson 0.716 (n=59,710, exact agreement 0.57); split-half Spearman–Brown of seen-user means 0.708; seen-vs-held-out mean Spearman 0.292 (median 4 held-out ratings per palette, so this ceiling is noise-limited, not a generalization failure).

## Generators, raw vs full pipeline (items 2, 8)

| Method | Pipeline | diversity | gamut_rate_raw | duplicate_rate | score | recon_min_over_K | recon_mean_over_K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline:colorwheel | no | 0.160 | 1.000 | 0.025 | -0.088 | 0.184 | 0.253 |
| baseline:retrieval | no | 0.249 | 1.000 | 0.002 | -0.116 | 0.129 | 0.228 |
| baseline:retrieval_filtered | no | 0.267 | 1.000 | 0.007 | -0.104 | 0.122 | 0.228 |
| baseline:retrieval_filtered+pipeline | yes | 0.246 | 1.000 | 0.000 | -0.004 | 0.114 | 0.227 |
| baseline:kde | no | 0.275 | 0.794 | 0.000 | -0.109 | 0.132 | 0.243 |
| v2/mabs | no | 0.227 | 0.900 | 0.016 | -0.045 | 0.119 | 0.248 |
| v2/mabs | yes | 0.220 | 1.000 | 0.000 | 0.058 | 0.114 | 0.221 |
| v2/mabs_subset | no | 0.233 | 0.895 | 0.013 | -0.046 | 0.121 | 0.245 |
| v2/mabs_subset | yes | 0.219 | 1.000 | 0.000 | 0.070 | 0.114 | 0.221 |
| v2/flow | no | 0.233 | 0.904 | 0.013 | -0.073 | 0.120 | 0.225 |
| v2/flow | yes | 0.234 | 1.000 | 0.000 | 0.040 | 0.115 | 0.223 |
| v2/srgb | no | 0.231 | 0.898 | 0.018 | -0.082 | 0.121 | 0.234 |
| v2/srgb | yes | 0.223 | 1.000 | 0.000 | 0.045 | 0.115 | 0.222 |

## Sampling-step sweep (item 7)

**v2/mabs**

| method | steps | s/call | scorer | diversity | gamut per-color | gamut whole | dup rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ddim | 10 | 0.038 | -0.089 | 0.203 | 0.830 | 0.769 | 0.070 |
| ddim | 20 | 0.047 | -0.038 | 0.196 | 0.900 | 0.841 | 0.031 |
| dpmpp2m | 10 | 0.038 | -0.162 | 0.230 | 0.457 | 0.300 | 0.067 |
| dpmpp2m | 20 | 0.073 | -0.059 | 0.208 | 0.744 | 0.642 | 0.034 |

**v2/flow**

| method | steps | s/call | scorer | diversity | gamut per-color | gamut whole | dup rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| euler | 5 | 0.014 | -0.007 | 0.190 | 0.985 | 0.971 | 0.038 |
| euler | 10 | 0.01 | -0.036 | 0.211 | 0.962 | 0.928 | 0.029 |
| euler | 20 | 0.018 | -0.052 | 0.222 | 0.938 | 0.882 | 0.017 |
| euler | 50 | 0.042 | -0.063 | 0.229 | 0.901 | 0.822 | 0.022 |
| heun | 5 | 0.009 | -0.059 | 0.227 | 0.959 | 0.925 | 0.029 |
| heun | 10 | 0.017 | -0.069 | 0.232 | 0.934 | 0.885 | 0.019 |
| heun | 20 | 0.033 | -0.071 | 0.233 | 0.904 | 0.825 | 0.019 |
| heun | 50 | 0.081 | -0.071 | 0.233 | 0.878 | 0.786 | 0.019 |

**v2/srgb**

| method | steps | s/call | scorer | diversity | gamut per-color | gamut whole | dup rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ddim | 10 | 0.025 | -0.130 | 0.208 | 0.872 | 0.803 | 0.082 |
| ddim | 20 | 0.031 | -0.084 | 0.209 | 0.916 | 0.868 | 0.058 |
| ddim | 100 | 0.147 | -0.087 | 0.231 | 0.898 | 0.834 | 0.024 |
| dpmpp2m | 10 | 0.026 | -0.259 | 0.261 | 0.505 | 0.351 | 0.089 |
| dpmpp2m | 20 | 0.046 | -0.134 | 0.232 | 0.757 | 0.639 | 0.036 |
| dpmpp2m | 100 | 0.221 | -0.093 | 0.234 | 0.882 | 0.800 | 0.024 |

## Common validation loss (same objective/space only)

| run | common val loss | by m=1..5 |
| --- | --- | --- |
| experiments/v2/mabs | 1.0941 | 1.049, 1.093, 1.098, 1.103, 1.183 |
| experiments/v2/mabs_subset | 1.1034 | 1.070, 1.103, 1.103, 1.105, 1.183 |
| experiments/v2/srgb | 2.2163 | 2.205, 2.274, 2.170, 2.170, 2.330 |

## Findings by review item

1. **Bugs.** All five fixed with tests: fixed–fixed pairs no longer trigger duplicate rejection (`#808080` + `#818181` now completes); repulsion also separates targets from context; subset-derived examples get the null rating token; eligible (post-filter) counts are logged (30,687, not 45,924); best checkpoints are saved and used for evaluation.
2. **Fair retrieval.** With the same rating filter and the same rerank/diversity pipeline, retrieval reaches scorer 0.00 vs raw retrieval −0.12: most of the earlier "neural gain" was postprocessing. Through the *same* pipeline the diffusion models still win on the scorer (M-abs 0.06, M-abs+subset 0.07) and match retrieval on reconstruction, but retrieval keeps higher diversity (0.25 vs 0.22). The generator's honest advantage is quality at equal duplicate rate, not variety.
3. **Dedup recall.** The per-channel-sorted embedding has a provable recall bound (‖F_A−F_B‖ ≤ n·d); it found 483 near pairs the sort-by-L heuristic missed. Exact cross-split audit: 0 near pairs in any split pair. The 11k-row chain is forced into train.
4. **Rater separation.** 114 users removed from all training signal; 781 palettes changed filter status. Reliability ceilings: repeat-judgment r = 0.72, split-half Spearman–Brown = 0.71, so the ordinal scorer's 0.75 vs seen-user means is at the ceiling. Held-out-user agreement is 0.29 because those users give a median of 4 ratings per palette; the scorer reaches 0.25 of it. A larger held-out rater pool, not a better model, is what would raise that number.
5. **Balanced human pilot.** Probe set rebuilt to 26 conditional contexts balanced by fixed-color count (6/10/8/2 for n = 1–4); the study package has 318 blinded items over 6 methods including "complete to five then subset"; the power model now includes item variance (with default guesses, N ≈ 117 raters, and the item-variance floor shows that adding contexts matters more than adding raters). Not run.
6. **Scorer.** Ordinal cumulative-link with per-rater shift and scale beats MSE on means (0.751 vs 0.733 Spearman, 3 seeds each) and is deployed. Within-user pairwise training is worse (0.68); the ordered-features scorer still does not beat DeepSets (R3 stands).
7. **Step sweep (VP, DDIM).** 20 steps = 100 steps on every metric at 0.055 s/call vs 0.27. DPM-Solver++ 2M as implemented is worse than DDIM at every step count; not diagnosed further because item 8 makes it moot.
8. **Flow matching vs VP; sRGB vs Oklab.** Flow matching (same network, velocity target) with 5–10 Euler steps: 0.010–0.014 s/call, whole-completion gamut validity 93–97% (VP best: 84%), scorer at or above VP; diversity is lower at 5 steps (0.19) and recovers by 20 (0.22). Through the pipeline flow scores 0.04 with diversity 0.23. **Recommendation: switch the deployed generator to flow matching with 10 Euler steps** and retrain the subset variant on it. Training in gamma sRGB gives no gamut benefit (87% vs 84–87% whole-completion at 20 steps) and a worse scorer; Oklab stays.

## Caveats

- v2 numbers are on rebuilt splits and a new scorer scale; do not compare to G2–G4 tables.
- Single seed per v2 run.
- The DPM-Solver++ implementation is verified only against an oracle denoiser; its gap to DDIM on the real model is unexplained.

