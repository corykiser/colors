# Status against the handoff plan

| Phase | Gate | State | Notes |
| --- | --- | --- | --- |
| 0 Data audit | G0 | **done** | `data/manifest/AUDIT.md`. Kuler archive = 44,986 themes (not 104k). PAT unavailable (404). |
| 1 Data pipeline | G1 | **done** | `data/manifest/dataset_card.json`. 453,358 records, 406,315 duplicate groups. One single-linkage chain of 11,096 pale near-neutral palettes (mostly COLOURlovers) at the 0.02 threshold; kept as one group (conservative for leakage), see note below. |
| 2 Eval harness + baselines | G2 | **done** | `reports/G2_baselines.md`. DeepSets scorer Spearman 0.74 on test; ordered scorer does not win (R3). Retrieval is the baseline to beat. |
| 3 M-abs diffusion | G3 | in progress | model, sampler, training script, invariance tests done; full runs in `experiments/` |
| 4 Geometric variants | G4 | not started | |
| 5 Sampling pipeline | G5 | not started | |
| 6 Human-eval prep | — | not started | |

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
