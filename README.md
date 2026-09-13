# palette

Conditional color palette generation: given any subset of colors, propose additional colors that go with them.
A conditional set-diffusion model over Oklab, with permutation-invariant conditioning and an ablation on
hue-rotation-equivariant geometry.

The contract for this project is [`palette_model_handoff.md`](palette_model_handoff.md). Read it first.
Progress against its phases is tracked in [`STATUS.md`](STATUS.md).

## Setup

```sh
uv sync
uv run pytest
```

Raw source archives live in `data/raw/` (git-ignored; see `data/manifest/inventory.json` for URLs and checksums).
`data/manifest/AUDIT.md` is the Phase 0 audit.

## Layout

```
palette/          package: color math, schema, ingest, dedup, splits, sampler, diffusion, models, baselines, eval
scripts/          audit / ingest / train / sample / evaluate entry points
configs/          one yaml per experiment
tests/            pytest
data/manifest/    license evidence, inventories, verified counts (committed)
data/processed/   parquet in the §2.8 schema (git-ignored)
experiments/      run outputs (git-ignored)
```

Data licensing: the O'Donovan 2011/2014 releases are CC BY-NC-SA 2.5 CA. This project is a noncommercial personal prototype.
