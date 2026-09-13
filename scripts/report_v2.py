"""Assemble reports/G5_review_changes.md from v2 experiment outputs (after the external review, items 1-8)."""
from __future__ import annotations

import json
from pathlib import Path

from palette.data import ROOT

REP = ROOT / "reports"; E = ROOT / "experiments"
COLS = ["diversity", "gamut_rate_raw", "duplicate_rate", "score", "recon_min_over_K", "recon_mean_over_K"]
f3 = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def main(runs=("v2/mabs", "v2/mabs_subset", "v2/flow", "v2/srgb")):
    card = json.load(open(ROOT / "data/manifest/dataset_card.json"))
    bl = json.load(open(E / "baselines/metrics.json"))
    sc = json.load(open(E / "scorer/results.json"))
    cv = json.load(open(E / "common_val.json")) if (E / "common_val.json").exists() else {}
    L = ["# G5 — Changes after the external review (items 1–8)", "",
         "All numbers below are on the **rebuilt splits** (exact-recall dedup, chains > 100 forced to train, MTurk means without the 114 held-out raters) and the **ordinal scorer** (latent scale, higher is better; not comparable to the 1–5 scale in G2–G4).", "",
         "## Data (items 1, 3, 4)", "",
         f"- Dedup: {card['dedup']['n_groups']:,} groups from {card['dedup']['n_records']:,} records; {card['dedup']['near_duplicate_pairs']:,} near pairs (was 166,717 with the heuristic blocking); largest chain {card['dedup']['largest_group']:,} rows, now in train.",
         f"- Cross-split near-duplicate pairs (exact audit): {card['cross_split_near_pairs']}.",
         f"- Rater separation: {card['rater_separation']['heldout_users']} users held out of every training signal; the rating-≥3 generator filter changed for {card['rater_separation']['ge3_decision_changed']} palettes.",
         "- Training counts now report post-filter eligible palettes per family; best checkpoints are saved and evaluated.", "",
         "## Scorer (items 4, 6)", "",
         "| Variant | Spearman vs seen-user mean | Pairwise acc | Spearman vs held-out-user mean |", "| --- | --- | --- | --- |"]
    for v in ("deepsets", "sortedL", "ordered", "deepsets-ordinal", "deepsets-pairwise"):
        m, s = sc[v]["mean"], sc[v]["std"]
        L.append(f"| {v} | {m['spearman_seen']:.3f} ± {s['spearman_seen']:.3f} | {m['pairwise_acc_seen']:.3f} | {m['spearman_heldout_users']:.3f} ± {s['spearman_heldout_users']:.3f} |")
    r = sc["reliability_test"]
    L += ["", f"Deployed: **{sc['deployed']}**. Reliability ceilings on the test palettes: repeat-judgment Pearson {r['repeat_pearson']:.3f} (n={r['n_repeat_pairs']:,}, exact agreement {r['repeat_exact_agree']:.2f}); split-half Spearman–Brown of seen-user means {r['split_half_spearman_brown']:.3f}; seen-vs-held-out mean Spearman {r['seen_vs_heldout_mean_spearman']:.3f} (median {r['median_heldout_ratings_per_palette']:.0f} held-out ratings per palette, so this ceiling is noise-limited, not a generalization failure).", "",
          "## Generators, raw vs full pipeline (items 2, 8)", "",
          "| Method | Pipeline | " + " | ".join(COLS) + " |", "| --- | --- | " + " | ".join("---" for _ in COLS) + " |"]
    for name, rr in bl["results"].items():
        s = rr["summary"]; L.append(f"| baseline:{name} | {'yes' if 'pipeline' in name else 'no'} | " + " | ".join(f3(s.get(k, '—')) for k in COLS) + " |")
    for d in runs:
        p = E / d / "metrics_v2.json"
        if not p.exists(): continue
        m = json.load(open(p))
        for mode in ("raw", "pipeline"):
            s = m[mode]; L.append(f"| {d} | {'yes' if mode == 'pipeline' else 'no'} | " + " | ".join(f3(s.get(k, '—')) for k in COLS) + " |")
    L += ["", "## Sampling-step sweep (item 7)", ""]
    for d in runs:
        p = E / d / "step_sweep.json"
        if not p.exists(): continue
        L += [f"**{d}**", "", "| method | steps | s/call | scorer | diversity | gamut per-color | gamut whole | dup rate |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in json.load(open(p)):
            L.append(f"| {row['method']} | {row['steps']} | {row['sec_per_call']} | {row['scorer']:.3f} | {row['diversity']:.3f} | {row['gamut_per_color']:.3f} | {row['gamut_whole']:.3f} | {row['dup_rate']:.3f} |")
        L.append("")
    if cv:
        L += ["## Common validation loss (same objective/space only)", "", "| run | common val loss | by m=1..5 |", "| --- | --- | --- |"]
        for d, v in cv.items():
            L.append(f"| {d} | {v['common_val_loss']:.4f} | " + ", ".join(f"{v['by_m'][k]:.3f}" for k in sorted(v['by_m'])) + " |")
        L.append("")
    notes = REP / "G5_notes.md"
    if notes.exists(): L.append(notes.read_text())
    (REP / "G5_review_changes.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
