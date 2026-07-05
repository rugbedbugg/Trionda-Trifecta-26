"""`python -m wcpredictor` — command menu.

A lightweight help screen listing every command in the toolkit, so the full
surface is discoverable without digging through the README. No heavy imports
here — it just prints.
"""
from __future__ import annotations

import sys

_MENU = [
    ("Core pipeline", [
        ("wcpredictor.pipeline",
         "Build the unified store, train, evaluate, and write live predictions"),
        ("wcpredictor.pipeline --use-fbref",
         "…also attempt the optional FBref scrape in the enrich stage"),
        ("wcpredictor.ingest / resolve / enrich",
         "Run an individual early stage (unified store / player_link / ability)"),
        ("wcpredictor.features / evaluate / predict_live",
         "Run an individual later stage (features / metrics / live preds)"),
    ]),
    ("Tournament", [
        ("wcpredictor.analyst [--retrain]",
         "One-command prose briefing: state of play, track record, remaining "
         "fixtures + scores, and what changed since last run"),
        ("wcpredictor.sim",
         "Simulate the whole 2026 tournament from scratch, then validate it"),
        ("wcpredictor.sim.simulate",
         "Just the prediction — full bracket + champion (writes the report)"),
        ("wcpredictor.sim.validate",
         "Just the scoring — the simulation vs what actually happened"),
        ("wcpredictor.sim.continue_live [--scores]",
         "Predict the REAL remaining fixtures from all results so far "
         "(--scores adds a projected scoreline for each)"),
        ("wcpredictor.sim.dynamic_eval",
         "Walk-forward eval: predict each match from earlier real results"),
    ]),
    ("Single predictions", [
        ('wcpredictor.sim.matchup "A" "B" [--knockout]',
         "Win / draw / loss for any two teams (order-invariant)"),
        ('wcpredictor.sim.goals "A" "B" [--knockout] [--gbm]',
         "Scoreline: expected goals, over/under, both-teams-to-score, correct scores"),
        ("wcpredictor.sim.goals",
         "…with no teams: run the score-model backtest"),
    ]),
    ("Reference / data", [
        ("wcpredictor.sim.rankings [N]",
         "Teams ranked by current Elo (top N, or all)"),
        ("wcpredictor.sim.market",
         "Model W/D/L next to prediction-market odds (fill data/market_odds.csv)"),
        ("wcpredictor.sim.intl",
         "Ingest the Kaggle international-results dataset (optional score upgrade)"),
    ]),
]

_EXAMPLES = [
    'python -m wcpredictor.sim',
    'python -m wcpredictor.sim.matchup "Spain" "France"',
    'python -m wcpredictor.sim.goals "Brazil" "Argentina" --knockout',
    'python -m wcpredictor.sim.rankings 10',
]


def print_menu() -> None:
    print("\n  wcpredictor — FIFA World Cup match & tournament predictor")
    print("  " + "=" * 62)
    print("  Run any command with:  python -m <command>\n")
    for section, rows in _MENU:
        print(f"  {section}")
        print("  " + "-" * 62)
        for cmd, desc in rows:
            print(f"    {cmd}")
            print(f"        {desc}")
        print()
    print("  Examples")
    print("  " + "-" * 62)
    for ex in _EXAMPLES:
        print(f"    {ex}")
    print("\n  Tests:  python tests/test_pipeline.py")
    print("  Docs :  README.md\n")


def main(argv=None):
    print_menu()
    return 0


if __name__ == "__main__":
    sys.exit(main())
