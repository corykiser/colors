"""Render probe completions to a self-contained HTML swatch grid.
Usage: uv run python scripts/render_probes.py experiments/mabs [experiments/baselines ...] -o experiments/probes.html"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from palette.data import ROOT


def load(d: Path) -> dict[str, dict]:
    if (d / "probes.json").exists():
        return {d.name: json.load(open(d / "probes.json"))}
    if (d / "metrics.json").exists():  # baselines file
        m = json.load(open(d / "metrics.json"))
        if "results" in m:
            return {f"baseline:{k}": v["probes"] for k, v in m["results"].items()}
    raise FileNotFoundError(d)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("-o", default="experiments/probes.html"); ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    runs = {}
    for d in a.dirs: runs.update(load(ROOT / d))
    names = list(next(iter(runs.values())).keys())
    css = "body{font:13px system-ui;margin:16px;background:#fafafa}.sw{display:inline-block;width:26px;height:26px;margin:0 1px;border:1px solid #0002;vertical-align:middle}.ctx{border:2px solid #000}.row{margin:2px 0;white-space:nowrap}.run{color:#666;display:inline-block;width:150px}h3{margin:18px 0 4px}"
    H = [f"<meta charset=utf-8><style>{css}</style><h2>Probe completions (first {a.n} of K; fixed colors outlined)</h2>"]
    for pn in names:
        H.append(f"<h3>{pn}</h3>")
        for rn, pr in runs.items():
            p = pr[pn]; ctx = "".join(f'<span class="sw ctx" style="background:{c}" title="{c}"></span>' for c in p["context"])
            comps = " &nbsp; ".join("".join(f'<span class="sw" style="background:{c}" title="{c}"></span>' for c in comp) for comp in p["completions"][: a.n])
            H.append(f'<div class="row"><span class="run">{rn}</span>{ctx} <b>→</b> {comps}</div>')
    Path(ROOT / a.o).write_text("\n".join(H))
    print("wrote", a.o)


if __name__ == "__main__":
    main()
