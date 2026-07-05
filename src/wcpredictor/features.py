"""Stage 4 — Leakage-safe feature builder.

The single organising discipline of this project: every feature for a match
must be computable strictly *before* kickoff. We enforce that structurally by
sweeping matches in date order and, for each match, reading the current rolling
state *first*, then updating that state with the result *after* the row is
emitted. A match never sees its own outcome, and never sees a future match.

Features (all cross-era, i.e. computable for 1930 and for 2026 alike):

  Elo        : a rolling World-Cup Elo rating per nation, carried across
               tournaments (home advantage applied only to host nations).
  Pedigree   : cumulative WC history — matches, win rate, goals for/against,
               knockout appearances — up to but excluding the current match.
  Form       : rolling last-5-match points and goals.
  Context    : knockout flag, host flags, rest days.

Squad ability (2026-only, from Stage 3) is attached where available but is NOT
part of the core cross-era feature set the models train on — it exists only for
the live-2026 augmented layer, because the historical data has no equivalent.
"""
from __future__ import annotations

import sqlite3
from collections import deque

import numpy as np
import pandas as pd

from . import paths
from .teams import canonical_team

# Core feature columns — available in every era, safe to train/validate/test on.
CORE_FEATURES = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "pedigree_matches_diff",
    "pedigree_winrate_diff",
    "pedigree_gd_diff",
    "pedigree_ko_diff",
    "form_points_diff",
    "form_gd_diff",
    "rest_days_diff",
    "is_knockout",
    "host_diff",
]

ELO_INIT = 1500.0
ELO_K = 40.0            # World Cup: high match importance
HOME_ADVANTAGE = 65.0   # applied to host nations only
FORM_WINDOW = 5
REST_CAP = 21           # days; first-of-tournament gaps are capped


def _elo_expected(home_elo: float, away_elo: float, home_ha: float, away_ha: float) -> float:
    diff = (away_elo + away_ha) - (home_elo + home_ha)
    return 1.0 / (1.0 + 10 ** (diff / 400.0))


def _gd_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


class _TeamState:
    __slots__ = ("elo", "n", "wins", "gf", "ga", "ko", "recent", "last_date")

    def __init__(self):
        self.elo = ELO_INIT
        self.n = 0
        self.wins = 0
        self.gf = 0
        self.ga = 0
        self.ko = 0
        self.recent = deque(maxlen=FORM_WINDOW)  # (points, gf, ga)
        self.last_date = None


# ---------------------------------------------------------------------------
# Reusable per-match feature/update primitives.
# Both build() (training) and the tournament simulator call these, so a
# simulated matchup is featurised by the exact same formulas as a training row.
# ---------------------------------------------------------------------------
def _winrate(s: _TeamState) -> float:
    return s.wins / s.n if s.n else 0.5


def _gd_avg(s: _TeamState) -> float:
    return (s.gf - s.ga) / s.n if s.n else 0.0


def _form_points(s: _TeamState) -> float:
    return float(np.mean([r[0] for r in s.recent])) if s.recent else 1.0


def _form_gd(s: _TeamState) -> float:
    return float(np.mean([r[1] - r[2] for r in s.recent])) if s.recent else 0.0


def _rest(s: _TeamState, date) -> float:
    if s.last_date is None or date is None:
        return REST_CAP
    return min((date - s.last_date).days, REST_CAP)


def core_feature_row(hs: _TeamState, as_: _TeamState, is_knockout: int,
                     home_is_host: int, away_is_host: int, date=None) -> dict:
    """The 12 core (cross-era, leakage-safe) features for one matchup."""
    return {
        "home_elo": hs.elo,
        "away_elo": as_.elo,
        "elo_diff": hs.elo - as_.elo,
        "pedigree_matches_diff": hs.n - as_.n,
        "pedigree_winrate_diff": _winrate(hs) - _winrate(as_),
        "pedigree_gd_diff": _gd_avg(hs) - _gd_avg(as_),
        "pedigree_ko_diff": hs.ko - as_.ko,
        "form_points_diff": _form_points(hs) - _form_points(as_),
        "form_gd_diff": _form_gd(hs) - _form_gd(as_),
        "rest_days_diff": _rest(hs, date) - _rest(as_, date),
        "is_knockout": int(is_knockout),
        "host_diff": int(home_is_host) - int(away_is_host),
    }


def update_after_match(hs: _TeamState, as_: _TeamState, hg: int, ag: int,
                       is_knockout: int, home_is_host: int, away_is_host: int,
                       date=None) -> None:
    """Fold a completed result into both teams' rolling state (Elo + tallies)."""
    h_ha = HOME_ADVANTAGE * int(home_is_host)
    a_ha = HOME_ADVANTAGE * int(away_is_host)
    exp_h = _elo_expected(hs.elo, as_.elo, h_ha, a_ha)
    score_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    k = ELO_K * _gd_multiplier(hg - ag)
    delta = k * (score_h - exp_h)
    hs.elo += delta
    as_.elo -= delta
    for s, gf, ga, won, drew in (
        (hs, hg, ag, hg > ag, hg == ag),
        (as_, ag, hg, ag > hg, hg == ag),
    ):
        s.n += 1
        s.wins += int(won)
        s.gf += gf
        s.ga += ga
        s.ko += int(is_knockout)
        s.recent.append((3 if won else (1 if drew else 0), gf, ga))
        s.last_date = date


def snapshot_states(cutoff_year: int = 2022) -> dict[str, _TeamState]:
    """Replay all completed matches with year <= cutoff and return each team's
    rolling state as it stood *entering* the next tournament. This is what the
    simulator uses so no in-tournament (2026) result leaks into a prediction."""
    with sqlite3.connect(paths.UNIFIED_DB) as con:
        matches = pd.read_sql("SELECT * FROM matches", con)
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    matches = matches[matches["year"] <= cutoff_year]
    matches = matches[matches["result"].notna()]
    matches = matches.sort_values(["match_date", "match_key"]).reset_index(drop=True)

    state: dict[str, _TeamState] = {}

    def get(team):
        if team not in state:
            state[team] = _TeamState()
        return state[team]

    for _, m in matches.iterrows():
        hs, as_ = get(m["home_team"]), get(m["away_team"])
        update_after_match(
            hs, as_, int(m["home_goals"]), int(m["away_goals"]),
            int(m["is_knockout"]), int(m["home_is_host"]), int(m["away_is_host"]),
            m["match_date"],
        )
    return state


def _squad_ability_by_team() -> dict[str, float]:
    """Mean ability of a 2026 squad, keyed by canonical team name."""
    with sqlite3.connect(paths.UNIFIED_DB) as con:
        ab = pd.read_sql("SELECT wc26_player_id, ability_score FROM player_ability", con)
        wc = pd.read_sql("SELECT wc26_player_id, team FROM wc26_players", con)
    merged = ab.merge(wc, on="wc26_player_id")
    return merged.groupby("team")["ability_score"].mean().to_dict()


def build() -> pd.DataFrame:
    with sqlite3.connect(paths.UNIFIED_DB) as con:
        matches = pd.read_sql("SELECT * FROM matches", con)
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    matches = matches.sort_values(["match_date", "match_key"]).reset_index(drop=True)

    squad_ability = _squad_ability_by_team()
    state: dict[str, _TeamState] = {}

    def get(team: str) -> _TeamState:
        if team not in state:
            state[team] = _TeamState()
        return state[team]

    rows = []
    for _, m in matches.iterrows():
        h, a = m["home_team"], m["away_team"]
        hs, as_ = get(h), get(a)
        date = m["match_date"]

        feat = {
            "match_key": m["match_key"],
            "source": m["source"],
            "year": int(m["year"]),
            "tournament_id": m["tournament_id"],
            "match_date": date,
            "stage_name": m["stage_name"],
            "status": m["status"],
            "home_team": h,
            "away_team": a,
            "result": m["result"],
            # --- core features (pre-match) ---
            **core_feature_row(hs, as_, m["is_knockout"],
                               m["home_is_host"], m["away_is_host"], date),
            # --- 2026-only augmentation (NaN for history, by design) ---
            "home_squad_ability": squad_ability.get(h, np.nan) if m["year"] == 2026 else np.nan,
            "away_squad_ability": squad_ability.get(a, np.nan) if m["year"] == 2026 else np.nan,
        }
        feat["squad_ability_diff"] = feat["home_squad_ability"] - feat["away_squad_ability"]
        rows.append(feat)

        # ---- update state AFTER emitting the row (never before) ----
        if pd.isna(m["home_goals"]) or pd.isna(m["away_goals"]):
            continue  # scheduled match: no result to learn from yet
        update_after_match(
            hs, as_, int(m["home_goals"]), int(m["away_goals"]),
            int(m["is_knockout"]), int(m["home_is_host"]), int(m["away_is_host"]),
            date,
        )

    return pd.DataFrame(rows)


def _assert_leakage_safe(df: pd.DataFrame) -> None:
    """Structural sanity checks that features are strictly pre-kickoff."""
    completed = df[df["result"].notna()].sort_values("match_date")
    # The very first completed match of all time must have a blank slate.
    first = completed.iloc[0]
    assert first["home_elo"] == ELO_INIT and first["away_elo"] == ELO_INIT, "Elo leak"
    assert first["pedigree_matches_diff"] == 0, "pedigree leak"
    # No feature column may correlate perfectly with the current-match goals:
    # they aren't in the frame at all, but assert we never accidentally kept them.
    assert "home_goals" not in df.columns and "away_goals" not in df.columns
    # Elo must stay finite and bounded.
    assert df["home_elo"].between(800, 2400).all(), "Elo out of sane range"


def run() -> None:
    paths.ensure_dirs()
    df = build()
    _assert_leakage_safe(df)
    with sqlite3.connect(paths.UNIFIED_DB) as con:
        # sqlite can't store datetime objects directly via to_sql on all versions
        out = df.copy()
        out["match_date"] = out["match_date"].dt.strftime("%Y-%m-%d")
        out.to_sql("features", con, if_exists="replace", index=False)

    n_2026 = (df["year"] == 2026).sum()
    print(
        f"[features] built {len(df)} rows x {len(CORE_FEATURES)} core features; "
        f"leakage assertions passed. 2026 rows: {n_2026} "
        f"({df[df['year'] == 2026]['result'].isna().sum()} still to be played)."
    )


if __name__ == "__main__":
    run()
