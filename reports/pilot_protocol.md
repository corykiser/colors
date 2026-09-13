# Pilot protocol for the completion-judgment study (Phase 6)

**Status:** stimulus package generator written (`scripts/make_study_package.py`). No raters recruited; nothing collected.

## Design

- Unit: one *completed palette* image = fixed colors (from a probe context) + generated additions. Equal-area swatches, fixed neutral background (#c8c8c8), swatch positions randomized per item; fixed colors are **not** marked, so raters judge the whole set.
- Methods (blinded): color wheel, retrieval, and each model checkpoint passed to the generator. Two samples per method per context by default. 21 conditional contexts × M methods × 2 samples, plus 6 repeated items for within-rater reliability.
- Questions, asked separately, each on a 7-point scale: harmony ("how well do these colors go together?") then liking ("how much do you like this combination?"). Order of the two questions fixed within a rater, counterbalanced across raters.
- Each rater sees every item once (within-subject), random order, plus the repeats. Session length target ≤ 12 minutes (~120 items at ~5 s each).
- Attention checks: the 6 repeats double as consistency checks; exclude raters whose repeat |Δ| mean exceeds 2 scale points.

## Analysis plan

- Primary: mixed-effects ordinal or linear model, rating ~ method + (1|rater) + (1|context). Report method contrasts vs retrieval with 95% CIs, separately for harmony and liking.
- Secondary: per-context method ranking; correlation of human liking with the DeepSets scorer; diversity–acceptability trade-off (item diversity from the generator vs mean liking).
- Do not score by recovery of any "original" palette; there is none for these contexts.

## Power (stub — fill in from the pilot)

With a within-subject design, the relevant quantity is the rater-level SD of the *difference* between two methods' mean ratings, σ_d. For a paired t-test at α = 0.05, power 0.8: N ≈ (2.8 · σ_d / δ)². Example: if the pilot gives σ_d = 0.8 and the smallest effect worth detecting is δ = 0.3 scale points, N ≈ 56 raters. Run a pilot of 8–10 raters to estimate σ_d and the repeat reliability before choosing N. `scripts/power_stub.py` computes N from σ_d and δ.

## Owner decisions needed

Platform (Prolific / in-person / friends), payment, and whether to collect demographics (the MTurk data has age/gender/country/experience codes; matching that is optional).
