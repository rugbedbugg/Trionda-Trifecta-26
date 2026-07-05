"""Stage 2 — Entity resolution (the keystone).

Every 2026 player who has previously appeared at a World Cup exists in the
Fjelstul data too, but under a different id and often a differently spelled
name ("Gvardiol" vs "Josko Gvardiol", accents dropped, single-name players,
duplicated names). This stage builds ``player_link`` mapping

    wc26_player_id  <->  fjelstul_player_id

so any downstream code can reach a 2026 player's historical appearances.

The matching cascade, strongest first:

  1. exact   — normalised full name AND birth date both match
  2. dob+fuzzy — birth date matches and names are fuzzily close (>= 0.80)
  3. name    — normalised full name matches uniquely (no DOB available)
  4. manual  — from data/player_link_overrides.csv (human-resolved stragglers)

Everything that fails to link is written to outputs/unresolved_players.csv so a
human can inspect it and, if warranted, add an override. A miss here is not a
bug: most 2026 players are simply too young to appear in data ending in 2022.
"""
from __future__ import annotations

import sqlite3
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from . import paths

FUZZY_THRESHOLD = 0.80


def _norm_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    n = n.lower()
    keep = [c if (c.isalnum() or c.isspace()) else " " for c in n]
    return " ".join("".join(keep).split())


def _iso_date(value) -> str | None:
    d = pd.to_datetime(value, errors="coerce")
    if pd.isna(d):
        return None
    return d.date().isoformat()


def _fuzzy(a: str, b: str) -> float:
    """Blend of full-string ratio and token-set overlap — robust to word order
    and to the family/given name being split differently across sources."""
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    jacc = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return max(seq, jacc)


def _load_overrides() -> dict[int, int]:
    """wc26_player_id -> fjelstul_player_id, human-resolved."""
    path = paths.PLAYER_LINK_OVERRIDES
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["wc26_player_id"], df["fjelstul_player_id"]))


def run() -> None:
    paths.ensure_dirs()
    with sqlite3.connect(paths.UNIFIED_DB) as con:
        fj = pd.read_sql("SELECT * FROM fjelstul_players", con)
        wc = pd.read_sql("SELECT * FROM wc26_players", con)

    fj["full"] = (fj["given_name"].fillna("") + " " + fj["family_name"].fillna("")).map(_norm_name)
    fj["dob"] = fj["birth_date"].map(_iso_date)
    wc["full"] = wc["player_name"].map(_norm_name)
    wc["dob"] = wc["birth_date"].map(_iso_date)

    # Indexes for fast candidate lookup.
    by_dob: dict[str, list[int]] = {}
    for i, dob in zip(fj.index, fj["dob"]):
        if dob:
            by_dob.setdefault(dob, []).append(i)
    exact_name: dict[str, list[int]] = {}
    for i, full in zip(fj.index, fj["full"]):
        exact_name.setdefault(full, []).append(i)

    overrides = _load_overrides()
    links = []
    unresolved = []

    for _, row in wc.iterrows():
        wid = int(row["wc26_player_id"])
        name, dob = row["full"], row["dob"]

        # 4. manual override wins outright
        if wid in overrides:
            links.append((wid, str(overrides[wid]), "manual", 1.0))
            continue

        cand = by_dob.get(dob, []) if dob else []
        # 1. exact name + dob
        hit = next((i for i in cand if fj.at[i, "full"] == name), None)
        if hit is not None:
            links.append((wid, fj.at[hit, "fjelstul_player_id"], "exact", 1.0))
            continue
        # 2. dob + fuzzy name
        if cand:
            scored = sorted(((_fuzzy(name, fj.at[i, "full"]), i) for i in cand), reverse=True)
            best_score, best_i = scored[0]
            if best_score >= FUZZY_THRESHOLD:
                links.append(
                    (wid, fj.at[best_i, "fjelstul_player_id"], "dob+fuzzy", round(best_score, 3))
                )
                continue
        # 3. unique exact name, no DOB corroboration
        name_hits = exact_name.get(name, [])
        if len(name_hits) == 1:
            links.append((wid, fj.at[name_hits[0], "fjelstul_player_id"], "name", 0.9))
            continue

        unresolved.append(
            {
                "wc26_player_id": wid,
                "player_name": row["player_name"],
                "team": row["team"],
                "birth_date": row["birth_date"],
                "reason": "ambiguous_name" if name_hits else "no_match",
                "candidate_count": len(name_hits) or len(cand),
            }
        )

    link_df = pd.DataFrame(
        links, columns=["wc26_player_id", "fjelstul_player_id", "match_method", "match_score"]
    )
    unresolved_df = pd.DataFrame(unresolved)

    with sqlite3.connect(paths.UNIFIED_DB) as con:
        link_df.to_sql("player_link", con, if_exists="replace", index=False)
    unresolved_df.to_csv(paths.UNRESOLVED_LOG, index=False)

    method_counts = link_df["match_method"].value_counts().to_dict() if len(link_df) else {}
    print(
        f"[resolve] linked {len(link_df)}/{len(wc)} 2026 players "
        f"({method_counts}); {len(unresolved_df)} unresolved "
        f"(logged to {paths.UNRESOLVED_LOG.name}). "
        f"Most 2026 players are debut-era and correctly have no pre-2022 record."
    )


if __name__ == "__main__":
    run()
