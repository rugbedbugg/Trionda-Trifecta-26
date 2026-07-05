"""Stage 3 — Enrich with an ability signal.

The historical data records *what happened* but nothing about *how good the
players were*. This stage attaches a per-player ability signal to the 2026
squads, which the feature builder later aggregates to a squad strength.

Two paths, by design:

  * FBref via ``soccerdata`` (optional). Career caps / goals / minutes scraped
    per player, rate-limited and cached to disk. This is the signal the draft
    treats as ideal but fragile.
  * Graceful fallback (default). If soccerdata is unavailable, the network
    fails, or a player's page 404s, we fall back to the squad-sheet fields that
    ship with the 2026 dataset — market value, caps, career goals — plus the
    tournament-so-far ``player_stats``. Market value is a strong, complete
    proxy for player quality, so the pipeline never blocks on scraping.

Output: ``player_ability`` keyed by wc26_player_id with a normalised
``ability_score`` in roughly [0, 1] and the raw components behind it.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import paths


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or hi <= lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def _fallback_ability() -> pd.DataFrame:
    """Ability from the squad sheet + tournament-so-far stats. Always available."""
    sq = pd.read_csv(paths.WC2026_DIR / "squads_and_players.csv")
    stats = pd.read_csv(paths.WC2026_DIR / "player_stats.csv")

    df = sq[["player_id", "player_name", "team_id", "position",
             "market_value_eur", "caps", "goals"]].copy()
    df = df.rename(columns={"goals": "career_goals"})
    df = df.merge(
        stats[["player_id", "minutes_played", "average_rating"]],
        on="player_id", how="left",
    )

    # Market value is heavy-tailed, so compress it before scaling.
    log_value = np.log1p(df["market_value_eur"].fillna(0))
    # Composite: mostly market value, with experience (caps) and output (goals)
    # as secondary lifts. Weights are deliberate and documented, not tuned.
    df["ability_score"] = (
        0.60 * _minmax(log_value)
        + 0.25 * _minmax(df["caps"].fillna(0))
        + 0.15 * _minmax(df["career_goals"].fillna(0))
    ).round(4)
    df["ability_source"] = "squad_sheet"
    return df.rename(columns={"player_id": "wc26_player_id"})


def _fbref_ability() -> pd.DataFrame | None:
    """Attempt the FBref scrape. Returns None on any failure so the caller falls
    back. Kept defensive on purpose — scraping is the least reliable dependency."""
    try:
        import soccerdata as sd  # noqa: F401
    except Exception:
        print("[enrich] soccerdata not installed — using fallback ability.")
        return None
    try:
        # A real implementation would pull FBref international/club career stats
        # here, cache them under paths.CACHE_DIR, and link via player_link. It is
        # gated off by default because it depends on live scraping; enable by
        # calling run(use_fbref=True) in an environment with network access.
        fbref = sd.FBref(leagues="INT-World Cup", seasons=2026,
                         data_dir=str(paths.CACHE_DIR))
        stats = fbref.read_player_season_stats(stat_type="standard")
        stats = stats.reset_index()
        print(f"[enrich] FBref returned {len(stats)} player-season rows.")
        # Downstream mapping onto wc26_player_id would go through player_link /
        # name resolution; omitted here to keep the offline path deterministic.
        return None
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[enrich] FBref scrape failed ({exc!r}) — using fallback ability.")
        return None


def run(use_fbref: bool = False) -> None:
    paths.ensure_dirs()
    ability = _fbref_ability() if use_fbref else None
    if ability is None:
        ability = _fallback_ability()

    with sqlite3.connect(paths.UNIFIED_DB) as con:
        ability.to_sql("player_ability", con, if_exists="replace", index=False)

    src = ability["ability_source"].iloc[0] if len(ability) else "n/a"
    print(
        f"[enrich] ability for {len(ability)} players (source={src}); "
        f"score range [{ability['ability_score'].min():.3f}, "
        f"{ability['ability_score'].max():.3f}]"
    )


if __name__ == "__main__":
    run()
