"""Show market odds next to the model's prediction, in W/D/L form.

Prediction markets (Kalshi, Polymarket, bookmakers) price in everything a crowd
with money knows — recent form, injuries, lineup news. Rather than fold them into
the model, we simply display them **alongside** our prediction so you can eyeball
where the model and the market agree or diverge.

Data flow (offline / reproducible):

  * Populate ``data/market_odds.csv`` with implied probabilities per match:
        match_id, home_team, away_team, p_home, p_draw, p_away, source
    Probabilities need not sum to 1 (vig / separate yes-no markets are fine) —
    each row is renormalised.
  * ``python -m wcpredictor.sim.market`` prints a model-vs-market table.

If the file doesn't exist yet, a ready-to-fill template of the knockout fixtures
is written so you only have to drop in the numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths
from . import bracket
from .model2026 import Specialist2026

MARKET_CSV = paths.DATA_DIR / "market_odds.csv"
COLUMNS = ["match_id", "home_team", "away_team", "p_home", "p_draw", "p_away", "source"]


def _result(row):
    hg, ag = row["home_score"], row["away_score"]
    return "H" if hg > ag else ("A" if ag > hg else "D")


def write_template(b: dict, overwrite: bool = False) -> None:
    """Write a blank odds sheet of the knockout fixtures for the user to fill."""
    if MARKET_CSV.exists() and not overwrite:
        return
    paths.ensure_dirs()
    meta = b["meta"]
    rows = []
    for mid, r in sorted(b["by_id"].items(), key=lambda kv: int(kv[0])):
        if r["stage_id"] == bracket.STAGE_GROUP:
            continue
        h, a = r.get("home_team_id"), r.get("away_team_id")
        if pd.isna(h) or pd.isna(a):
            continue  # participants not yet decided
        rows.append({"match_id": mid, "home_team": meta[int(h)]["name"],
                     "away_team": meta[int(a)]["name"],
                     "p_home": "", "p_draw": "", "p_away": "", "source": ""})
    pd.DataFrame(rows, columns=COLUMNS).to_csv(MARKET_CSV, index=False)


def load_market_odds() -> pd.DataFrame:
    """Read the odds sheet; drop unfilled rows; renormalise each row to sum 1."""
    if not MARKET_CSV.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(MARKET_CSV)
    for c in ("p_home", "p_draw", "p_away"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["p_home", "p_draw", "p_away"]).copy()
    if df.empty:
        return df
    s = df[["p_home", "p_draw", "p_away"]].sum(axis=1).replace(0, np.nan)
    for c in ("p_home", "p_draw", "p_away"):
        df[c] = df[c] / s
    return df.dropna(subset=["p_home", "p_draw", "p_away"])


def _wdl(v):
    return f"{v[0]:>3.0%} / {v[1]:>3.0%} / {v[2]:>3.0%}"


def display():
    b = bracket.derive()
    odds = load_market_odds()
    if odds.empty:
        write_template(b)
        print("\n[market] No market probabilities found yet.")
        print(f"[market] Wrote a template to: {MARKET_CSV}")
        print("[market] Fill p_home/p_draw/p_away (from Kalshi/Polymarket/bookmaker) "
              "and re-run\n         `python -m wcpredictor.sim.market`.")
        return None

    meta, by_id = b["meta"], b["by_id"]
    spec = Specialist2026().fit(
        [(int(r["home_team_id"]), int(r["away_team_id"]), _result(r))
         for r in by_id.values() if r["status"] == "Completed"]
    )

    print("\n" + "=" * 70)
    print(" MODEL vs MARKET  —  W / D / L")
    print("=" * 70)
    print(f" {'Match':<26}{'Model (H/D/A)':>20}{'Market (H/D/A)':>20}")
    print(" " + "-" * 68)
    rows = []
    for _, o in odds.iterrows():
        mid = o["match_id"]
        if mid not in by_id:
            continue
        row = by_id[mid]
        h, a = int(row["home_team_id"]), int(row["away_team_id"])
        mp = spec.predict_wdl(h, a)
        kp = np.array([o["p_home"], o["p_draw"], o["p_away"]])
        name = f"{meta[h]['fifa_code']} v {meta[a]['fifa_code']}"
        played = f"  ({_result(row)})" if row["status"] == "Completed" else ""
        print(f" {name + played:<26}{_wdl(mp):>20}{_wdl(kp):>20}")
        rows.append({"match": name, "model": mp, "market": kp})
    print("=" * 70)
    return rows


def run():
    display()


if __name__ == "__main__":
    run()
