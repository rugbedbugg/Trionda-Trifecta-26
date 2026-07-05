"""End-to-end pipeline runner.

Runs every stage in dependency order:

    ingest -> resolve -> enrich -> features -> evaluate -> predict_live

Run with:  python -m wcpredictor.pipeline
Optional:  --use-fbref   attempt the live FBref scrape in the enrich stage.
"""
from __future__ import annotations

import argparse
import warnings

from . import enrich, evaluate, features, ingest, paths, predict_live, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="World Cup match predictor pipeline")
    parser.add_argument("--use-fbref", action="store_true",
                        help="attempt live FBref scrape (network) in the enrich stage")
    args = parser.parse_args()

    # sklearn on a ~900-row dataset emits benign convergence chatter; hide it.
    warnings.filterwarnings("ignore")

    paths.ensure_dirs()
    print("=" * 70)
    print("World Cup match predictor — full pipeline")
    print("=" * 70)

    ingest.run()
    resolve.run()
    enrich.run(use_fbref=args.use_fbref)
    features.run()
    evaluate.run()
    print()
    predict_live.run()

    print("\nDone. Artifacts in:")
    print(f"  unified store : {paths.UNIFIED_DB}")
    print(f"  metrics       : {paths.OUTPUT_DIR / 'metrics.csv'}")
    print(f"  reliability   : {paths.OUTPUT_DIR / 'reliability_test.csv'}")
    print(f"  live preds    : {paths.OUTPUT_DIR / 'live_predictions.md'}")


if __name__ == "__main__":
    main()
