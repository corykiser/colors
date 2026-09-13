"""Generate gate reports from experiment outputs. Usage: uv run python scripts/report.py g2 | g3 <exp_dirs...>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from palette.data import ROOT

REP = ROOT / "reports"; REP.mkdir(exist_ok=True)
GEN_COLS = ["diversity", "gamut_rate_raw", "duplicate_rate", "score", "recon_min_over_K", "recon_mean_over_K"]


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def g2():
    sc = json.load(open(ROOT / "experiments/scorer/results.json"))
    bl = json.load(open(ROOT / "experiments/baselines/metrics.json"))
    L = ["# G2 — Baselines and scorer diagnostic", "",
         f"Index: {bl['n_index']:,} training palettes (Kuler + MTurk 2014, train split). Probe contexts: 22 (see `palette/eval/probes.py`); K = 16 completions each. Held-out reconstruction: 300 val palettes, random 1–3 fixed colors, best of 8 completions — **diagnostic only**.", "",
         "## Population scorer on MTurk 2014 mean ratings (test split, mean ± std over 3 seeds)", "",
         "| Variant | Params | Spearman | Pairwise acc | MAE | Spearman, order shuffled |", "| --- | --- | --- | --- | --- | --- |"]
    for v in ("deepsets", "sortedL", "ordered"):
        m, s = sc[v]["mean"], sc[v]["std"]
        sh = f"{m['test_spearman_shuffled_order']:.3f} ± {s['test_spearman_shuffled_order']:.3f}" if "test_spearman_shuffled_order" in m else "—"
        L.append(f"| {v} | {int(m['params']):,} | {m['test_spearman']:.3f} ± {s['test_spearman']:.3f} | {m['test_pairwise_acc']:.3f} ± {s['test_pairwise_acc']:.3f} | {m['test_mae']:.3f} | {sh} |")
    c = sc["_context"]
    L += ["", f"Train/val/test palettes: {c['n_train']:,}/{c['n_val']:,}/{c['n_test']:,}. Mean per-theme rating variance on test: {c['test_mean_rating_variance']:.2f} (1–5 scale). Training time for all 9 runs: {c['seconds']}s on CPU.", "",
          "**R3 diagnostic:** the ordered scorer (stored presentation order + adjacent differences) does not beat the permutation-invariant DeepSets scorer; shuffling its input order at test time costs it a few hundredths of Spearman. The labels carry at most weak order information at this model size. The fresh equal-area, randomized-position evaluation set stays desirable but is not forced by this result.", "",
          "## Generative baselines", "",
          "| Method | Diversity | Gamut rate (raw) | Duplicate rate | Scorer mean | Recon min/K | Recon mean/K | Eval s |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in bl["results"].items():
        s = r["summary"]
        L.append(f"| {name} | " + " | ".join(fmt(s.get(k, '—')) for k in GEN_COLS) + f" | {s['eval_seconds']} |")
    L += ["", "Diversity = mean pairwise min-cost Oklab distance among the K generated parts (after gamut mapping). Scorer mean = DeepSets population scorer on the full mapped palette (1–5 scale; 3.0 ≈ average MTurk theme). Colorwheel is a formula, so its scorer value is the reference for 'no data'.", "",
          "Retrieval reconstructs held-out colors best (it returns real palettes) but cannot generalize beyond its index; KDE adds jitter and loses gamut validity. Any learned model must beat retrieval on scorer mean and diversity together, not on reconstruction."]
    (REP / "G2_baselines.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def g3(dirs: list[str], title="G3 — Diffusion models vs baselines", fname="G3_mabs.md"):
    bl = json.load(open(ROOT / "experiments/baselines/metrics.json"))
    L = [f"# {title}", "", "| Run | Params | Train palettes | Train s | Final val loss | " + " | ".join(GEN_COLS) + " |",
         "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in GEN_COLS) + " |"]
    for name, r in bl["results"].items():
        s = r["summary"]; L.append(f"| baseline:{name} | — | — | — | — | " + " | ".join(fmt(s.get(k, '—')) for k in GEN_COLS) + " |")
    for d in dirs:
        m = json.load(open(ROOT / d / "metrics.json"))
        L.append(f"| {m['name'].replace('experiments/', '')} | {m['params']:,} | {m['train_palettes']:,} | {m['train_seconds']} | {m['final_val_loss']:.4f} | " + " | ".join(fmt(m.get(k, '—')) for k in GEN_COLS) + " |")
    (REP / fname).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "g2": g2()
    elif cmd == "g3": g3(sys.argv[2:])
    elif cmd == "g4": g3(sys.argv[2:], "G4 — Geometry ablation", "G4_ablation.md")
