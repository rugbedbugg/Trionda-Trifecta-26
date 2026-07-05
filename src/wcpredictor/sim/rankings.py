"""Show the 2026 teams ranked by current Elo — the signal that drives predictions.

    python -m wcpredictor.sim.rankings        # all 48 teams
    python -m wcpredictor.sim.rankings 10      # top 10 only

Elo is the model's dominant feature (roughly tied with squad value, and far
above host advantage), so this is effectively the model's power ranking.
"""
from __future__ import annotations

import sys

import pandas as pd

from .. import paths


def table(top: int | None = None) -> pd.DataFrame:
    teams = pd.read_csv(paths.WC2026_DIR / "teams.csv")
    squads = pd.read_csv(paths.WC2026_DIR / "squads_and_players.csv")
    value = squads.groupby("team_id")["market_value_eur"].sum()

    df = teams[["team_name", "fifa_code", "group_letter",
                "elo_rating", "fifa_ranking_pre_tournament"]].copy()
    df["squad_value_m"] = teams["team_id"].map(value).fillna(0) / 1e6
    df = df.sort_values("elo_rating", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df.head(top) if top else df


def show(top: int | None = None) -> None:
    df = table(top)
    scope = f"top {top}" if top else f"all {len(df)}"
    print(f"\n 2026 World Cup teams by current Elo ({scope})")
    print(f" {'#':>2}  {'Team':<24}{'Elo':>6}{'FIFA#':>7}{'Squad €M':>10}  Grp")
    print(" " + "-" * 58)
    for _, r in df.iterrows():
        print(f" {r['rank']:>2}  {r['team_name']:<24}{r['elo_rating']:>6.0f}"
              f"{int(r['fifa_ranking_pre_tournament']):>7}{r['squad_value_m']:>10.0f}"
              f"   {r['group_letter']}")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    top = None
    if argv:
        arg = argv[0].lower()
        if arg not in ("all", "0"):
            try:
                top = int(arg)
            except ValueError:
                print('Usage: python -m wcpredictor.sim.rankings [N | all]')
                return
    show(top)


if __name__ == "__main__":
    main()
