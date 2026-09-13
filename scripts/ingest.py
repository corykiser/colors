"""Ingest all cleared sources into data/processed/<family>.parquet."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from palette.schema import write_parquet
from palette.ingest.odonovan import ingest_kuler, ingest_mturk2011, ingest_colourlovers, ingest_mturk2014
from palette.ingest.wada import ingest_wada

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"


def main(families: list[str] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {
        "kuler": lambda: ingest_kuler(RAW),
        "mturk2011": lambda: ingest_mturk2011(RAW),
        "colourlovers": lambda: ingest_colourlovers(RAW),
        "mturk2014": lambda: ingest_mturk2014(RAW)[0],
        "wada": lambda: ingest_wada(RAW),
    }
    for fam in families or list(jobs):
        t0 = time.time()
        recs = jobs[fam]()
        write_parquet(recs, str(OUT / f"{fam}.parquet"))
        print(f"{fam:14s} {len(recs):7d} records  {time.time() - t0:5.1f}s")
    if not families or "mturk2014" in families:
        users = ingest_mturk2014(RAW)[1]
        json.dump(users, open(OUT / "mturk2014_users.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
