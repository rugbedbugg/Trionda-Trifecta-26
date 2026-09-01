# Contributing to Trionda-Trifecta-26

Contributions should protect the predictor's temporal evaluation boundary and keep the command-line modules independently runnable.

## Development setup

Use the uv-managed Python selected by `.python-version`:

```sh
uv venv
uv pip install -r requirements.txt
uv run python tests/test_pipeline.py
```

Use `uv run python -m wcpredictor` to inspect the command surface. The optional `soccerdata` dependency is only required for FBref enrichment.

## Change guidelines

- Keep implementation under `src/wcpredictor/` and tests under `tests/`.
- Maintain the documented train/validation/test chronology and reject feature leakage from future fixtures.
- Add a regression assertion for pipeline, model, calibration, or simulation changes.
- Make network-backed enrichment optional and retain the documented fallback behavior.
- State the dataset snapshot and random seed for metric changes.

## Pull requests

Include the 19-test pipeline result, relevant backtest metrics, and a concise explanation of changes to features, probabilities, or tournament rules.
