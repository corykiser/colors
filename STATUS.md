# Status against the handoff plan

| Phase | Gate | State | Notes |
| --- | --- | --- | --- |
| 0 Data audit | G0 | **done** | `data/manifest/AUDIT.md`. Kuler archive = 44,986 themes (not 104k). PAT unavailable (404). |
| 1 Data pipeline | G1 | **done** | `data/manifest/dataset_card.json`. 453,358 records, 406,315 duplicate groups. One single-linkage chain of 11,096 pale near-neutral palettes (mostly COLOURlovers) at the 0.02 threshold; kept as one group (conservative for leakage), see note below. |
| 2 Eval harness + baselines | G2 | **done** | `reports/G2_baselines.md`. DeepSets scorer Spearman 0.74 on test; ordered scorer does not win (R3). Retrieval is the baseline to beat. |
| 3 M-abs diffusion | G3 | **done** | `reports/G3_mabs.md`. M-abs (819k params, 15 min on M5 Max) beats retrieval on scorer mean and reconstruction. Subset augmentation costs ~0.01 val loss on 5-color data. |
| 4 Geometric variants | G4 | **done** | `reports/G4_ablation.md`. Geometry gives no measurable gain at 1/10/100% data; geometric-only is worse and drops raw gamut validity to 76%; combined +0.5% at 1.8× params. Recommendation: ship M-abs. |
| 5 Sampling pipeline | G5 | **done** | `palette/api.py`: rejection → repulsion → chroma-reduction fallback, scorer rerank, greedy diverse top-k; `scripts/sample.py` CLI |
| 6 Human-eval prep | — | **done** (no collection) | `scripts/make_study_package.py`, `reports/pilot_protocol.md`, `scripts/power_stub.py` |

## Post-review changes (reports/G5_review_changes.md)

External review items 1–8 implemented: bug fixes, exact-recall dedup and re-split, rater separation, fair pipelined retrieval, ordinal scorer, balanced study package, step sweep, flow-matching and sRGB ablations. Result: through a matched pipeline the generator beats retrieval on quality but not diversity; flow matching at 10 Euler steps is the recommended sampler (97% whole-completion gamut validity, 14 ms/call). Reference runs are now under `experiments/v2/`.

## G1 notes

- Dedup: exact (Oklab quantized 1e-3, permutation-invariant) + near (min-cost mean Oklab distance < 0.02, KD-tree blocking on sort-by-L). 26,094 exact-duplicate rows; 166,717 near pairs. Group `dg:5` (11,096 rows, mean L 0.84, 47% of colors with chroma < 0.03) is a transitive chain through the dense pale/neutral region. It is assigned to one split as a whole. Revisit with a stricter threshold or complete-linkage if it skews split composition.
- Splits: 80/10/10 by group, seed 0. 114 of 1,137 MTurk users held out for unseen-user evaluation.
- Cross-family groups confirm MTurk ⊂ Kuler (12,270 shared groups) and ~950 COLOURlovers/Kuler cross-posts.

## Decisions taken (owner said "personal use")

- Noncommercial prototype: S1, S2, S6 ingested under their stated terms. `commercial_reuse_status` stays `unreviewed` everywhere.
- Wada included, every record flagged `wada_us_status_unresolved`.
- PAT skipped: official link dead; mirrors have no provenance.

## Open for the owner (handoff Part 5)

3. Emailing Schloss–Palmer / Ou et al. for raw data.
5. Personalization: S1 has stable user IDs, so it is *possible*; deferred to v2 per reviewer.
6. Human study budget/platform.
