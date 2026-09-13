# Phase 0 data audit

**Date:** 2026-09-13. **Scope:** S1, S2, S3, S6 per handoff §4.2. S4, S5, S7–S10 not attempted.
**Intended use (per owner):** personal, noncommercial prototype.

Every count below was computed from the downloaded files (`inventory.json` has SHA-256 for each).
"Reported" = the number in the paper or handoff. "Verified" = what is in the archive.

## 1. Verified counts vs reported

| Source | File | Reported | Verified rows | Unique source IDs | Exact dups (perm-invariant, 8-bit) |
| --- | --- | --- | --- | --- | --- |
| S2 MTurk 2011 | `mturkData.mat` | 10,743 | **10,743** | 10,672 | 119 |
| S2 Kuler | `kulerData.mat` | 104,426 | **44,986** | 44,986 | 424 |
| S2 COLOURlovers | `colorLoversData.mat` | 383,938 | **383,938** | 383,938 | 1,445 |
| S1 themes 2014 | `themeData.mat` | 13,343 | **13,343** | 13,242 | 181 |
| S1 individual ratings | `allMTurkRatings.mat` | 528,106 | **528,106** | 1,137 raters | — |
| S6 Wada | `colors.json` | 348 palettes / 159 colors | **348 / 159** | — | 0 |
| S3 PAT | — | 10,183 | **not acquired** (see §5) | — | — |

**Kuler discrepancy:** the released Kuler file holds 44,986 themes, 43% of the 104,426 the paper reports. The paper's figure is the crawled set; the release is a subset. Plan around ~45k, not 104k.

**MTurk count reconciliation (2011: 10,743 vs 2014: 13,343):** every 2011 MTurk theme ID is in the 2014 theme set. The 2014 release adds 2,570 themes (2,504 of them Kuler IDs) rated in a later batch. Mean ratings for shared IDs agree (mean |Δ| = 0.005 on a 1–5 scale). So S1 ⊃ S2-MTurk; they are one study extended, not two studies.

**Duplicate IDs:** 71 MTurk-2011 IDs and 101 S1 IDs appear twice with identical colors but different mean ratings. These are the same theme rated in two batches. Keep both rating sets; assign one `duplicate_group_id`.

## 2. Field availability per source

| Field | S2 MTurk | S2 Kuler | S2 COLOURlovers | S1 themes | S1 ratings | S6 Wada |
| --- | --- | --- | --- | --- | --- | --- |
| Raw colors | 5×3 sRGB float [0,1] | 5×3 sRGB float [0,1] | 5×3 sRGB uint8 | 15 floats (5×3 row-major, verified against S2) | — | CMYK % + ICC-converted sRGB/Lab |
| Color space stated | no (sRGB assumed) | no (sRGB assumed) | no (sRGB assumed) | no (sRGB assumed) | — | yes |
| Stable palette ID | Kuler numeric | Kuler numeric | COLOURlovers numeric | Kuler numeric | index 1..13343 into themes | 1..348 |
| Stable user ID | no | no | no | no | **yes** (1..1137) | — |
| Individual ratings | no (mean only) | no (mean only) | no | no | **yes**, integer 1–5 | — |
| Rating question | not in archive (paper: "how much do you like this theme", 1–5) | Kuler site stars | hearts/views | as MTurk | as MTurk | none (curated) |
| Ratings per item | ~40 (paper) | mostly 1–4 (from mean granularity) | — | 26–80, mean 39.6 | — | — |
| User-normalized target | yes | no | no | no | derivable | — |
| Presentation order | stored order (paper: horizontal 5-swatch strip) | stored order | stored order | stored order | — | book order |
| Names/text | theme name | theme name | mixed: theme names and apparently author names ("dariusmonsef") | theme name | — | color names |
| Rater metadata | — | — | — | — | age/gender/country/experience codes for 1,461 slots; legend **not in archive** | — |

Other S1 facts: 66,647 (user, theme) pairs are rated more than once (repeat judgments for rater consistency). Ratings per user range 27–13,500 (median 90). `regressTargets` equals the mean of the individual ratings exactly. `train_vec/probe_vec/test_vec` are the authors' split of the same ratings (36 rows differ from `testRatings`). The 334 handcrafted features are also included.

COLOURlovers `targets` (1.81–5.0) is an undocumented derived score; it correlates 0.74 with hearts/views. Do not treat it as a rating. Hearts median is 1, 90th percentile 6.

## 3. Cross-release overlap

Matched by source ID, and separately by exact 8-bit color multiset.

| Pair | ID matches | Color matches | Reading |
| --- | --- | --- | --- |
| S1 ∩ S2-MTurk | 10,672 | 10,624 | S1 is a superset of S2-MTurk |
| S1 ∩ S2-Kuler | 12,955 | 12,898 | MTurk themes were drawn from Kuler |
| S2-MTurk ∩ S2-Kuler | 10,451 | 10,420 | same |
| S2-Kuler ∩ S2-COLOURlovers | 15,504 | 399 | **ID collision** — separate ID spaces; prefix IDs by site |
| S1 ∩ S2-COLOURlovers | 4,636 | 154 | same collision; ~150 genuinely cross-posted palettes |
| S2-MTurk ∩ S2-COLOURlovers | 3,715 | 129 | same |

Consequence: the "three sources" are one Kuler-derived pool plus COLOURlovers. Splits must be assigned on a unified key (site-prefixed ID ∪ color-multiset) so an MTurk-rated theme never lands in train via Kuler and in test via MTurk.

## 4. License evidence (copies in `license_evidence/`)

| Source | Evidence | Reading | Commercial reuse |
| --- | --- | --- | --- |
| S2 (2011) | `README.txt` in archive + project page: "Creative Commons BY-NC-SA", link to CC BY-NC-SA **2.5 Canada** | noncommercial OK with attribution + share-alike | unreviewed / not permitted |
| S1 (2014) | Project page + `runPrediction.m` header: permissive notice ("copy, use, modify, or distribute … for any purpose") inherited from Salakhutdinov's PMF code. The page's CC BY-NC-SA line sits next to a commented-out `importanceModel.zip` link that now 404s. | The permissive notice covers *programs and documents*. The theme/rating data is the same MTurk study as S2. **Treat the data as CC BY-NC-SA 2.5 CA** (most restrictive applicable). | unreviewed |
| S3 (PAT) | Repo `LICENSE` = MIT (code). README: "for the use of PAT dataset for your research, please cite". No dataset license text. | Dataset terms unstated. | unreviewed |
| S6 (Wada) | `LICENSE.md` = MIT (DesLauriers 2020 transcription, forked from dblodorn). | Transcription MIT. Underlying work: Japan PD since 2018; **U.S. status unresolved** (handoff §2.6). | unreviewed |

## 5. PAT acquisition failure

`install_pre.sh` fetches Google Drive file `1CtYnsTASXFZZ4RCJXZaxEWvybz_4AMa4`. Both the legacy `docs.google.com/uc` form and `drive.usercontent.google.com` return HTTP 404 (2026-09-13). GitHub code search shows the pickle files (`train_palettes_rgb.pkl` etc.) vendored in several unrelated student/project repos with no license. **Not downloaded** — a third-party copy has no provenance chain. Options for the owner: email the authors, or accept a mirror copy as "provenance: unverified mirror" for the noncommercial prototype. PAT is rank 4 and optional; nothing in Phases 1–5 requires it.

## 6. Other findings

- Presentation conditions are not recorded in any archive. Per the 2011 paper, MTurk raters saw horizontal five-swatch strips in the stored order. Stored order is preserved as `original_order`.
- No archive states its color space; all values are consistent with sRGB and are treated as such with `quality_flags: ["colorspace_assumed_srgb"]`.
- Wada sRGB values come from a SWOP-v2 → sRGB ICC conversion (relative colorimetric, BPC), documented in the repo. CMYK originals are preserved; flag `cmyk_icc_estimate`.
- `hueProbsRGB.mat` / `kulerX.mat` are the 2011 paper's hue-prior features, not data; ignored.
- Demographic codes exist but their legend is absent; stored as opaque integers, not interpreted.

## 7. Gate G0 status

Sources usable for the noncommercial personal prototype under their stated terms: **S1, S2 (all three files), S6**. Held for the owner: PAT (unavailable from an authoritative source), Wada's U.S. status (flagged in every Wada record, included for the prototype unless told otherwise).

`commercial_reuse_status = "unreviewed"` for every source.
