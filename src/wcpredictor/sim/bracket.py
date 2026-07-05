"""Reconstruct the official 2026 knockout bracket from the dataset itself.

The 2026 dataset encodes the real bracket through its match numbering and the
teams that actually advanced. We recover, deterministically and with no
hard-coded table:

  * the **R32 seeding template** — each of the 16 Round-of-32 slots expressed as
    a (group, finishing-position) pair, e.g. "winner of Group A vs a 3rd-placed
    team", by reading who actually filled each slot and what position they
    finished in their group;
  * the **feeder tree** for every later round — which two earlier matches feed
    each Round-of-16 / quarter-final, found by tracing each participant back to
    the match it advanced from;
  * the **semi-final pairing**, for the (unplayed) later rounds, from the actual
    2026 layout — the four quarter-finals pair up sequentially by match number
    (QF1+QF2 -> SF1, QF3+QF4 -> SF2).

The result is a bracket the simulator can fill with its *own* predicted
qualifiers while preserving the exact official structure.
"""
from __future__ import annotations

import pandas as pd

from .. import paths
from ..teams import canonical_team
from . import rules

STAGE_GROUP = "1"
STAGE_R32 = "2"
STAGE_R16 = "3"
STAGE_QF = "4"
STAGE_SF = "5"
STAGE_3RD = "6"
STAGE_FINAL = "7"


def load_2026():
    m = pd.read_csv(paths.WC2026_DIR / "matches.csv", dtype={"stage_id": str})
    teams = pd.read_csv(paths.WC2026_DIR / "teams.csv")
    return m, teams


def _team_meta(teams: pd.DataFrame) -> dict:
    meta = {}
    for _, r in teams.iterrows():
        meta[r["team_id"]] = {
            "team_id": r["team_id"],
            "name": r["team_name"],
            "canonical": canonical_team(r["team_name"]),
            "fifa_code": r["fifa_code"],
            "group": r["group_letter"],
            "elo": float(r["elo_rating"]),
        }
    return meta


def actual_advancer(row):
    """Team id that advanced from a completed match (goals, then penalties)."""
    if row["status"] != "Completed":
        return None
    hs, as_ = row["home_score"], row["away_score"]
    if pd.isna(hs) or pd.isna(as_):
        return None
    if hs > as_:
        return row["home_team_id"]
    if as_ > hs:
        return row["away_team_id"]
    hp, ap = row.get("home_penalty_score"), row.get("away_penalty_score")
    if pd.notna(hp) and pd.notna(ap):
        return row["home_team_id"] if hp > ap else row["away_team_id"]
    return None


def actual_standings(matches: pd.DataFrame, meta: dict):
    """Real group tables -> {team_id: (group, position)} and per-group order."""
    seed = {tid: meta[tid]["elo"] for tid in meta}
    group_matches = matches[matches["stage_id"] == STAGE_GROUP]
    positions, group_order = {}, {}
    for group in sorted({meta[t]["group"] for t in meta}):
        team_ids = [t for t in meta if meta[t]["group"] == group]
        gm = group_matches[
            group_matches["home_team_id"].isin(team_ids)
            & group_matches["away_team_id"].isin(team_ids)
        ]
        results = [
            (r["home_team_id"], r["away_team_id"], int(r["home_score"]), int(r["away_score"]))
            for _, r in gm.iterrows()
            if pd.notna(r["home_score"]) and pd.notna(r["away_score"])
        ]
        order, _ = rules.rank_group(team_ids, results, seed)
        group_order[group] = order
        for pos, tid in enumerate(order, start=1):
            positions[tid] = (group, pos)
    return positions, group_order


def derive():
    """Return the full bracket blueprint (see module docstring)."""
    matches, teams = load_2026()
    meta = _team_meta(teams)
    positions, group_order = actual_standings(matches, meta)

    by_id = {r["match_id"]: r for _, r in matches.iterrows()}
    advancers = {mid: actual_advancer(r) for mid, r in by_id.items()}

    def stage_ids(stage):
        return sorted(
            (mid for mid, r in by_id.items() if r["stage_id"] == stage), key=int
        )

    # R32 seeding template: each slot = (position, group) of who actually filled it.
    r32 = []
    for mid in stage_ids(STAGE_R32):
        r = by_id[mid]
        r32.append({
            "match_id": mid,
            "stage": STAGE_R32,
            "home_slot": positions[r["home_team_id"]],   # (group, pos)
            "away_slot": positions[r["away_team_id"]],
        })

    def feeder_of(team_id, prev_stage):
        for mid in stage_ids(prev_stage):
            if advancers.get(mid) == team_id:
                return mid
        return None

    # R16 and QF: trace each participant back to the match it advanced from.
    ko = []
    for stage, prev in ((STAGE_R16, STAGE_R32), (STAGE_QF, STAGE_R16)):
        for mid in stage_ids(stage):
            r = by_id[mid]
            ko.append({
                "match_id": mid,
                "stage": stage,
                "home_src": ("winner", feeder_of(r["home_team_id"], prev)),
                "away_src": ("winner", feeder_of(r["away_team_id"], prev)),
            })

    # Semi-finals: the four quarter-finals pair up sequentially by match number
    # (QF1+QF2 -> SF1, QF3+QF4 -> SF2), which is the actual 2026 bracket layout.
    qf_ids = stage_ids(STAGE_QF)          # sorted by match id, e.g. [97, 98, 99, 100]
    sf_ids = stage_ids(STAGE_SF)
    ko.append({"match_id": sf_ids[0], "stage": STAGE_SF,
               "home_src": ("winner", qf_ids[0]), "away_src": ("winner", qf_ids[1])})
    ko.append({"match_id": sf_ids[1], "stage": STAGE_SF,
               "home_src": ("winner", qf_ids[2]), "away_src": ("winner", qf_ids[3])})

    # Third-place = the two SF losers; final = the two SF winners.
    third_id = stage_ids(STAGE_3RD)[0]
    final_id = stage_ids(STAGE_FINAL)[0]
    ko.append({"match_id": third_id, "stage": STAGE_3RD,
               "home_src": ("loser", sf_ids[0]), "away_src": ("loser", sf_ids[1])})
    ko.append({"match_id": final_id, "stage": STAGE_FINAL,
               "home_src": ("winner", sf_ids[0]), "away_src": ("winner", sf_ids[1])})

    return {
        "meta": meta,
        "positions": positions,        # actual: team_id -> (group, pos)
        "group_order": group_order,    # actual: group -> [team_ids]
        "advancers": advancers,        # actual: match_id -> advancing team_id
        "by_id": by_id,                # actual match rows
        "r32": r32,
        "ko": ko,
        "third_slots": [               # bracket order + the group whose 3rd sat there
            {"match_id": s["match_id"], "side": side, "group": slot[0]}
            for s in r32
            for side, slot in (("home", s["home_slot"]), ("away", s["away_slot"]))
            if slot[1] == 3
        ],
    }


STAGE_NAMES = {
    STAGE_R32: "Round of 32", STAGE_R16: "Round of 16", STAGE_QF: "Quarter-final",
    STAGE_SF: "Semi-final", STAGE_3RD: "Third-place", STAGE_FINAL: "Final",
}
