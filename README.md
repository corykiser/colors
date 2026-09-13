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

## Results so far

Gate reports live in `reports/` (G2 baselines + scorer, G3 M-abs, G4 geometry ablation). Short version: the
absolute-color set-diffusion model (819k params, ~15 min on an M5 Max) beats retrieval and color-wheel
baselines on scorer-rated quality; adding rotation-invariant geometry features or an exact-equivariant
pathway does not measurably help at 1%, 10% or 100% of the data.

## Using a trained model

```sh
uv run python scripts/train.py configs/mabs_subset.yaml --out experiments/mabs_subset
uv run python scripts/sample.py --model experiments/mabs_subset/model_ema.pt --colors "#9caf88,#6b4a2b" --m 2 --k 4
```

`palette.api.complete(["#9caf88", "#6b4a2b"], m=2, k=4)` is the programmatic entry point: fixed colors are
returned verbatim; generated colors are gamut-checked, de-duplicated, reranked by the scorer and diversified.

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

## License

Code: MIT (`LICENSE`). Trained weights and derived data: CC BY-NC-SA 2.5 Canada, inherited from the O'Donovan 2011/2014 releases (`LICENSE-WEIGHTS`). Write-up and live demo: https://corykiser.github.io/colors/
