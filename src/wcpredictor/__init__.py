"""wcpredictor — a World Cup W/D/L outcome model built from raw sources up.

The package is a linear pipeline, each stage reading the previous stage's output
from the unified SQLite store:

    ingest  ->  resolve  ->  enrich  ->  features  ->  models/evaluate  ->  predict_live

See README.md for the full write-up. Run everything with ``python -m wcpredictor.pipeline``.
"""

__version__ = "0.1.0"
