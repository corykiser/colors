# G3 — Diffusion models vs baselines

| Run | Params | Data | Train palettes | Train s | Final val loss | Best val loss (step) | Common val loss | diversity | gamut_rate_raw | duplicate_rate | score | recon_min_over_K | recon_mean_over_K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline:colorwheel | — | 100% | — | — | — | — | — | 0.153 | 1.000 | 0.034 | 2.721 | 0.169 | 0.239 |
| baseline:retrieval | — | 100% | — | — | — | — | — | 0.228 | 1.000 | 0.003 | 2.682 | 0.125 | 0.229 |
| baseline:kde | — | 100% | — | — | — | — | — | 0.253 | 0.792 | 0.020 | 2.702 | 0.132 | 0.246 |
| mabs | 819,459 | 100% | 45,924 | 897.9 | 1.0753 | 1.0753 (20000) | 1.0593 | 0.207 | 0.905 | 0.023 | 2.773 | 0.120 | 0.248 |
| mabs_subset | 819,459 | 100% | 45,924 | 999.2 | 1.1279 | 1.1279 (20000) | 1.0684 | 0.210 | 0.893 | 0.017 | 2.761 | 0.121 | 0.250 |
