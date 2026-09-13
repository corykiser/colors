"""Re-evaluate trained runs with the *current* scorer, both raw and through the full API pipeline (matched
oversample budget with the pipelined retrieval baseline). Writes <run>/metrics_v2.json.
Usage: uv run python scripts/reeval.py experiments/v2/mabs [...]"""
from __future__ import annotations

import json, sys, time
import numpy as np
from palette.api import Completer, load_model
from palette.sampling import DiffusionCompleter
from palette.data import load_arrays, ROOT
from palette.eval.report import evaluate_generator, load_scorer


def main(dirs):
    val = load_arrays(("kuler", "mturk2014"), "val", None); valX = val.X[val.M.sum(1) == 5]
    score = load_scorer()
    for d in dirs:
        ck = ROOT / d / ("model_best.pt" if (ROOT / d / "model_best.pt").exists() else "model_ema.pt")
        model, sched, norm, dev = load_model(str(ck)); cfg = json.load(open(ROOT / d / "metrics.json"))["config"]
        s = cfg.get("sample", {}); space = getattr(model, "space", "oklab")
        raw = DiffusionCompleter(model, sched, norm, dev, steps=s.get("steps", 100), method=s.get("method", "ddim"), space=space)
        pipe = Completer(model, sched, norm, dev, steps=s.get("steps", 100), method=s.get("method", "ddim"), scorer=score)
        t0 = time.time(); r_raw = evaluate_generator(raw.complete, valX, score_fn=score)
        t1 = time.time(); r_pipe = evaluate_generator(lambda c, m, k, sd: pipe.complete_oklab(c, m, k, sd)[0], valX, score_fn=score)
        out = {"run": d, "checkpoint": ck.name, "raw": r_raw["summary"], "pipeline": r_pipe["summary"],
               "raw_seconds": round(t1 - t0, 1), "pipeline_seconds": round(time.time() - t1, 1)}
        json.dump(out, open(ROOT / d / "metrics_v2.json", "w"), indent=1)
        json.dump(r_pipe["probes"], open(ROOT / d / "probes_pipeline.json", "w"), indent=1)
        print(d, "raw", json.dumps({k: round(v, 3) for k, v in r_raw["summary"].items()}))
        print(d, "pipe", json.dumps({k: round(v, 3) for k, v in r_pipe["summary"].items()}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
