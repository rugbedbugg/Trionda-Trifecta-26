"""Stage 1 — Ingest.

Fuse the two raw sources into one unified SQLite store:

  * Fjelstul men's World Cups 1930-2022  -> historical backbone + labels
  * FIFA-World-Cup-2026-Dataset          -> the live 2026 tournament

The centrepiece is a single cross-era ``matches`` fact table with one
consistent schema and one consistent W/D/L label (derived from goals, so a
match decided on penalties is correctly a *draw*). We also mirror the player
tables from both sources so the resolver has something to link.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from . import paths
from .teams import canonical_team

HOSTS_2026 = {"united states", "canada", "mexico"}


def _result_from_goals(hg, ag):
    """Home-perspective W/D/L label from regulation+ET goals."""
    if pd.isna(hg) or pd.isna(ag):
        return None
    hg, ag = int(hg), int(ag)
    if hg > ag:
        return "H"
    if hg < ag:
        return "A"
    return "D"


def _load_fjelstul_matches() -> pd.DataFrame:
    """Men's World Cup matches, one row per match, canonical schema."""
    m = pd.read_csv(paths.FJELSTUL_CSV / "matches.csv")
    tournaments = pd.read_csv(paths.FJELSTUL_CSV / "tournaments.csv")

    mens = tournaments[tournaments["tournament_name"].str.contains("Men's")]
    year_by_tid = dict(zip(mens["tournament_id"], mens["year"]))
    host_by_tid = dict(zip(mens["tournament_id"], mens["host_country"]))
    m = m[m["tournament_id"].isin(year_by_tid)].copy()

    # Fjelstul excludes replays' duplicate rows via the `replay` flag; keep the
    # played match, drop the (later) replay marker rows that duplicate a fixture.
    m = m[m["replay"] == 0].copy()

    out = pd.DataFrame(
        {
            "match_key": m["match_id"],
            "source": "fjelstul",
            "tournament_id": m["tournament_id"],
            "year": m["tournament_id"].map(year_by_tid).astype(int),
            "stage_name": m["stage_name"],
            "is_knockout": m["knockout_stage"].astype(int),
            "match_date": pd.to_datetime(m["match_date"]),
            "home_team_raw": m["home_team_name"],
            "away_team_raw": m["away_team_name"],
            "home_goals": m["home_team_score"],
            "away_goals": m["away_team_score"],
            "status": "Completed",
        }
    )
    out["host_country"] = out["tournament_id"].map(host_by_tid)
    return out


def _load_2026_matches() -> pd.DataFrame:
    m = pd.read_csv(paths.WC2026_DIR / "matches.csv")
    teams = pd.read_csv(paths.WC2026_DIR / "teams.csv")
    stages = pd.read_csv(paths.WC2026_DIR / "tournament_stages.csv")

    name_by_id = dict(zip(teams["team_id"], teams["team_name"]))
    ko_by_stage = dict(zip(stages["stage_id"], stages["is_knockout"]))

    out = pd.DataFrame(
        {
            "match_key": "WC2026-M" + m["match_id"].astype(str),
            "source": "wc2026",
            "tournament_id": "WC-2026",
            "year": 2026,
            "stage_name": m["stage_id"].map(dict(zip(stages["stage_id"], stages["stage_name"]))),
            "is_knockout": m["stage_id"].map(ko_by_stage).astype(int),
            "match_date": pd.to_datetime(m["date"]),
            "home_team_raw": m["home_team_id"].map(name_by_id),
            "away_team_raw": m["away_team_id"].map(name_by_id),
            "home_goals": m["home_score"],
            "away_goals": m["away_score"],
            "status": m["status"],
        }
    )
    out["host_country"] = "United States"  # tri-hosted; host flag handled below
    return out


def build_matches() -> pd.DataFrame:
    frames = [_load_fjelstul_matches(), _load_2026_matches()]
    matches = pd.concat(frames, ignore_index=True)

    matches["home_team"] = matches["home_team_raw"].map(canonical_team)
    matches["away_team"] = matches["away_team_raw"].map(canonical_team)
    matches["result"] = [
        _result_from_goals(h, a) for h, a in zip(matches["home_goals"], matches["away_goals"])
    ]

    # Host flags. Pre-2026 a single host nation; 2026 has three co-hosts.
    def _is_host(team_key, host_country, year):
        if year == 2026:
            return int(team_key in HOSTS_2026)
        return int(team_key == canonical_team(host_country))

    matches["home_is_host"] = [
        _is_host(t, h, y)
        for t, h, y in zip(matches["home_team"], matches["host_country"], matches["year"])
    ]
    matches["away_is_host"] = [
        _is_host(t, h, y)
        for t, h, y in zip(matches["away_team"], matches["host_country"], matches["year"])
    ]

    matches = matches.sort_values(["match_date", "match_key"]).reset_index(drop=True)
    cols = [
        "match_key", "source", "tournament_id", "year", "stage_name", "is_knockout",
        "match_date", "home_team", "away_team", "home_team_raw", "away_team_raw",
        "home_goals", "away_goals", "result", "home_is_host", "away_is_host", "status",
    ]
    return matches[cols]


def _load_players() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Player rosters from both sources, lightly standardised for the resolver."""
    fj = pd.read_csv(paths.FJELSTUL_CSV / "players.csv")
    fj = fj[fj["female"] == 0].copy()
    fj_players = pd.DataFrame(
        {
            "fjelstul_player_id": fj["player_id"],
            "given_name": fj["given_name"].fillna(""),
            "family_name": fj["family_name"].fillna(""),
            "birth_date": fj["birth_date"],
        }
    )

    sq = pd.read_csv(paths.WC2026_DIR / "squads_and_players.csv")
    teams = pd.read_csv(paths.WC2026_DIR / "teams.csv")
    team_name = dict(zip(teams["team_id"], teams["team_name"]))
    wc26_players = pd.DataFrame(
        {
            "wc26_player_id": sq["player_id"],
            "player_name": sq["player_name"],
            "team": sq["team_id"].map(team_name).map(canonical_team),
            "birth_date": sq["date_of_birth"],
            "position": sq["position"],
            "caps": sq["caps"],
            "goals": sq["goals"],
            "market_value_eur": sq["market_value_eur"],
        }
    )
    return fj_players, wc26_players


def run() -> None:
    paths.ensure_dirs()
    matches = build_matches()
    fj_players, wc26_players = _load_players()

    with sqlite3.connect(paths.UNIFIED_DB) as con:
        matches.to_sql("matches", con, if_exists="replace", index=False)
        fj_players.to_sql("fjelstul_players", con, if_exists="replace", index=False)
        wc26_players.to_sql("wc26_players", con, if_exists="replace", index=False)

    completed = matches["result"].notna().sum()
    print(
        f"[ingest] {len(matches)} matches "
        f"({completed} completed, {len(matches) - completed} scheduled), "
        f"{matches['year'].min()}-{matches['year'].max()} | "
        f"{len(fj_players)} historical players, {len(wc26_players)} 2026 players "
        f"-> {paths.UNIFIED_DB.name}"
    )


if __name__ == "__main__":
    run()
