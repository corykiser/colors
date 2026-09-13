# G2 — Baselines and scorer diagnostic

Index: 45,924 training palettes (Kuler + MTurk 2014, train split). Probe contexts: 22 (see `palette/eval/probes.py`); K = 16 completions each. Held-out reconstruction: 300 val palettes, random 1–3 fixed colors, best of 8 completions — **diagnostic only**.

## Population scorer on MTurk 2014 mean ratings (test split, mean ± std over 3 seeds)

| Variant | Params | Spearman | Pairwise acc | MAE | Spearman, order shuffled |
| --- | --- | --- | --- | --- | --- |
| deepsets | 33,665 | 0.738 ± 0.003 | 0.775 ± 0.001 | 0.177 | — |
| sortedL | 18,689 | 0.722 ± 0.001 | 0.767 ± 0.000 | 0.183 | — |
| ordered | 20,225 | 0.718 ± 0.007 | 0.766 ± 0.002 | 0.182 | 0.683 ± 0.007 |

Train/val/test palettes: 10,530/1,439/1,374. Mean per-theme rating variance on test: 1.15 (1–5 scale). Training time for all 9 runs: 35.4s on CPU.

**R3 diagnostic:** the ordered scorer (stored presentation order + adjacent differences) does not beat the permutation-invariant DeepSets scorer; shuffling its input order at test time costs it a few hundredths of Spearman. The labels carry at most weak order information at this model size. The fresh equal-area, randomized-position evaluation set stays desirable but is not forced by this result.

## Generative baselines

| Method | Diversity | Gamut rate (raw) | Duplicate rate | Scorer mean | Recon min/K | Recon mean/K | Eval s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| colorwheel | 0.153 | 1.000 | 0.034 | 2.721 | 0.169 | 0.239 | 0.5 |
| retrieval | 0.228 | 1.000 | 0.003 | 2.682 | 0.125 | 0.229 | 5.0 |
| kde | 0.253 | 0.792 | 0.020 | 2.702 | 0.132 | 0.246 | 4.1 |

Diversity = mean pairwise min-cost Oklab distance among the K generated parts (after gamut mapping). Scorer mean = DeepSets population scorer on the full mapped palette (1–5 scale; 3.0 ≈ average MTurk theme). Colorwheel is a formula, so its scorer value is the reference for 'no data'.

Retrieval reconstructs held-out colors best (it returns real palettes) but cannot generalize beyond its index; KDE adds jitter and loses gamut validity. Any learned model must beat retrieval on scorer mean and diversity together, not on reconstruction.
