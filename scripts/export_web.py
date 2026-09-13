"""Export a trained denoiser + the deployed scorer + normalization for in-browser inference.
Writes <out>/model.json (manifest, config, norm, reference vector, float16 weights base64-embedded).
Usage: uv run python scripts/export_web.py experiments/v2/flow_subset web/"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
import torch
from palette.api import load_model
from palette.data import ROOT
from palette.models.scorer import Scorer


def main(run, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ck = ROOT / run / ("model_best.pt" if (ROOT / run / "model_best.pt").exists() else "model_ema.pt")
    model, sched, norm, dev = load_model(str(ck)); model = model.cpu().eval()
    cfg = torch.load(ck, map_location="cpu")["config"]
    sc_ck = torch.load(ROOT / "experiments/scorer/deepsets.pt", map_location="cpu"); scorer = Scorer(sc_ck["variant"]); scorer.load_state_dict(sc_ck["state_dict"]); scorer.eval()
    tensors, buf, off = {}, bytearray(), 0
    for prefix, m in (("model", model), ("scorer", scorer)):
        for k, v in m.state_dict().items():
            a = v.detach().numpy().astype(np.float16); tensors[f"{prefix}.{k}"] = {"shape": list(a.shape), "offset": off, "n": a.size}
            buf += a.tobytes(); off += a.size * 2
    import base64
    weights_b64 = base64.b64encode(bytes(buf)).decode()   # served as JSON: binary assets are not servable on the artifact host
    # reference vector: sage+brown context, 2 targets at fixed noise, t index 500
    torch.manual_seed(0)
    x = torch.zeros(1, 4, 3); x[0, :2] = torch.tensor(norm.normalize(np.array([[0.7, -0.03, 0.05], [0.45, 0.03, 0.06]])), dtype=torch.float32)
    x[0, 2:] = torch.randn(2, 3); f = torch.tensor([[True, True, False, False]]); msk = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        y = model(x, f, msk, torch.tensor([500]), None)[0].numpy().tolist()
        s = scorer(x).item()
    manifest = {"run": run, "checkpoint": ck.name, "model": cfg["model"], "diffusion": cfg["diffusion"], "space": cfg.get("data", {}).get("space", "oklab"),
                "norm": norm.to_dict(), "scorer_variant": sc_ck["variant"], "tensors": tensors, "bytes": off, "weights_b64": weights_b64,
                "reference": {"x": x[0].numpy().tolist(), "is_fixed": [True, True, False, False], "t": 500, "out": y, "score": s}}
    json.dump(manifest, open(out / "model.json", "w"))
    print(f"exported {len(tensors)} tensors, {off/1e6:.2f} MB float16 →", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
