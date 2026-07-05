"""Ingester for the Kaggle international-football-results dataset.

Dataset: "International football results from 1872 to 2024" (martj42) —
~48k full internationals with columns:
    date, home_team, away_team, home_score, away_score, tournament, city,
    country, neutral

It needs a Kaggle login to download, so this ingester reads a local copy you
drop at ``data/international_results.csv`` and fails gracefully with instructions
if it's missing.

What it gives the rest of the project:
  * ``ingest()``  — a cleaned, team-canonicalised ``international_matches`` table
    in the unified store (dates, canonical teams, goals, neutral flag).
  * ``team_goal_rates()`` — per team, a time-weighted attack rate (goals scored)
    and defence rate (goals conceded) over recent years. These are the features
    the Poisson score model is starved for: strengths estimated from *dozens* of
    matches per team instead of the ~3 available inside a single World Cup.

Everything canonicalises through ``wcpredictor.teams.canonical_team`` so it joins
straight onto the 2026 field.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..teams import canonical_team

INTL_CSV = paths.DATA_DIR / "international_results.csv"
DOWNLOAD_HINT = (
    "Download 'results.csv' from\n"
    "  https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017\n"
    f"and save it as:\n  {INTL_CSV}"
)
_REQUIRED = ["date", "home_team", "away_team", "home_score", "away_score"]


def available() -> bool:
    return INTL_CSV.exists()


def load(path=INTL_CSV, min_year: int | None = None) -> pd.DataFrame:
    """Read + clean + canonicalise the results file into a tidy frame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"International results file not found.\n{DOWNLOAD_HINT}")
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns {missing}. {DOWNLOAD_HINT}")

    df = df.dropna(subset=["home_score", "away_score", "home_team", "away_team"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["home"] = df["home_team"].map(canonical_team)
    df["away"] = df["away_team"].map(canonical_team)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df.get("neutral", False)
    df["tournament"] = df.get("tournament", "")
    if min_year is not None:
        df = df[df["date"].dt.year >= min_year]
    cols = ["date", "home", "away", "home_score", "away_score", "tournament", "neutral"]
    return df.sort_values("date").reset_index(drop=True)[cols]


def _wc2026_canon() -> set[str]:
    teams = pd.read_csv(paths.WC2026_DIR / "teams.csv")
    return {canonical_team(n) for n in teams["team_name"]}


def team_goal_rates(df: pd.DataFrame | None = None, since_year: int = 2015,
                    half_life_days: int = 1095) -> pd.DataFrame:
    """Time-weighted attack (goals scored) and defence (goals conceded) rate per
    team. Recent matches count more (exponential half-life, default ~3 years).

    Returns columns: team, matches, gf_rate, ga_rate.
    """
    if df is None:
        df = load(min_year=since_year)
    else:
        df = df[df["date"].dt.year >= since_year]
    if df.empty:
        return pd.DataFrame(columns=["team", "matches", "gf_rate", "ga_rate"])

    asof = df["date"].max()
    age = (asof - df["date"]).dt.days.to_numpy()
    w = 0.5 ** (age / half_life_days)

    # one row per team-appearance (both home and away perspectives)
    long = pd.DataFrame({
        "team": pd.concat([df["home"], df["away"]], ignore_index=True),
        "gf": pd.concat([df["home_score"], df["away_score"]], ignore_index=True),
        "ga": pd.concat([df["away_score"], df["home_score"]], ignore_index=True),
        "w": np.concatenate([w, w]),
    })
    g = long.groupby("team")
    out = pd.DataFrame({
        "matches": g.size(),
        "gf_rate": g.apply(lambda x: np.average(x["gf"], weights=x["w"]), include_groups=False),
        "ga_rate": g.apply(lambda x: np.average(x["ga"], weights=x["w"]), include_groups=False),
    }).reset_index()
    return out.sort_values("gf_rate", ascending=False).reset_index(drop=True)


def team_strengths(df: pd.DataFrame | None = None, since_year: int = 2011,
                   half_life_days: int = 1460, alpha: float = 2.0) -> pd.DataFrame:
    """Opponent-adjusted attack & defence per team (a Poisson / Dixon-Coles fit).

    Fits ``log E[goals] = μ + attack[scorer] - defence[conceder] + home``
    over all recent internationals (time-weighted), with per-team attack and
    defence parameters. Unlike raw goal averages, this *de-conflates* how much a
    team scores from who it played — beating up minnows no longer inflates a
    rating. Returns columns: team, attack, defence (both log-scale; higher attack
    = scores more, higher defence = concedes more).
    """
    from sklearn.linear_model import PoissonRegressor

    if df is None:
        df = load(min_year=since_year)
    else:
        df = df[df["date"].dt.year >= since_year]
    if df.empty:
        return pd.DataFrame(columns=["team", "attack", "defence"])

    asof = df["date"].max()
    w = 0.5 ** ((asof - df["date"]).dt.days.to_numpy() / half_life_days)
    # two rows per match: each team's goals as attacker vs the other's defence.
    att = pd.concat([df["home"], df["away"]], ignore_index=True)
    dfn = pd.concat([df["away"], df["home"]], ignore_index=True)
    y = pd.concat([df["home_score"], df["away_score"]], ignore_index=True).to_numpy()
    is_home = np.concatenate([(~df["neutral"].astype(bool)).astype(int).to_numpy(),
                              np.zeros(len(df), dtype=int)])
    weights = np.concatenate([w, w])

    A = pd.get_dummies(att, prefix="att").astype(np.float32)
    D = pd.get_dummies(dfn, prefix="def").astype(np.float32)
    X = pd.concat([A, D, pd.Series(is_home, name="home")], axis=1)
    reg = PoissonRegressor(alpha=alpha, max_iter=1000).fit(X, y, sample_weight=weights)
    coef = dict(zip(X.columns, reg.coef_))

    teams = sorted(set(att))
    return pd.DataFrame({
        "team": teams,
        "attack": [coef.get(f"att_{t}", 0.0) for t in teams],
        "defence": [coef.get(f"def_{t}", 0.0) for t in teams],
    }).sort_values("attack", ascending=False).reset_index(drop=True)


def ingest() -> pd.DataFrame:
    """Store the cleaned matches in the unified DB and print a coverage report."""
    paths.ensure_dirs()
    df = load()
    store = df.copy()
    store["date"] = store["date"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(paths.UNIFIED_DB) as con:
        store.to_sql("international_matches", con, if_exists="replace", index=False)

    wc = _wc2026_canon()
    rates = team_goal_rates(df)
    covered = set(rates["team"]) & wc
    thin = sorted(t for t in wc if 0 < int(rates.set_index("team")["matches"].get(t, 0)) < 10)
    missing = sorted(wc - set(rates["team"]))

    print(f"[intl] ingested {len(df):,} international matches "
          f"{df['date'].dt.year.min()}-{df['date'].dt.year.max()} "
          f"-> international_matches")
    print(f"[intl] 2026 field coverage: {len(covered)}/{len(wc)} teams have "
          f"recent-form data (since {rates.attrs.get('since', 2015)}).")
    if thin:
        print(f"[intl] thin (<10 recent matches): {', '.join(thin)}")
    if missing:
        print(f"[intl] no match (name mismatch?): {', '.join(missing)}")
    return df


def run():
    if not available():
        print("[intl] No international results file yet.\n" + DOWNLOAD_HINT)
        return
    df = ingest()
    wc = _wc2026_canon()
    rates = team_goal_rates(df)
    rates = rates[rates["team"].isin(wc)]   # 2026 field only (dataset includes minnows)
    print("\n Top-scoring 2026 teams (recent, time-weighted goals-for per match):")
    for _, r in rates.head(10).iterrows():
        print(f"   {r['team']:<20} {r['gf_rate']:.2f} scored / "
              f"{r['ga_rate']:.2f} conceded   ({int(r['matches'])} matches)")


if __name__ == "__main__":
    run()
