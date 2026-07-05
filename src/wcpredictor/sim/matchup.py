"""Predict a single matchup between any two 2026 teams from the command line.

    python -m wcpredictor.sim.matchup "Spain" "Brazil"
    python -m wcpredictor.sim.matchup Argentina France --knockout

Uses the 2026-specialist model (current Elo + squad market value), trained on
every 2026 match played so far. Team names are matched case-insensitively with a
fuzzy fallback, so "usa", "Korea", "Turkiye" all resolve.
"""
from __future__ import annotations

import sys
from difflib import get_close_matches

from ..teams import canonical_team
from . import bracket
from .model2026 import Specialist2026


def _resolve(name: str, meta: dict):
    canon = canonical_team(name)
    by_canon = {m["canonical"]: tid for tid, m in meta.items()}
    if canon in by_canon:
        return by_canon[canon]
    # fuzzy fallback on canonical names and fifa codes
    names = list(by_canon)
    hit = get_close_matches(canon, names, n=1, cutoff=0.6)
    if hit:
        return by_canon[hit[0]]
    for tid, m in meta.items():
        if m["fifa_code"].lower() == name.strip().lower():
            return tid
    return None


def _result(row):
    hg, ag = row["home_score"], row["away_score"]
    return "H" if hg > ag else ("A" if ag > hg else "D")


def predict_matchup(home_name: str, away_name: str, knockout: bool = False):
    b = bracket.derive()
    meta = b["meta"]
    h = _resolve(home_name, meta)
    a = _resolve(away_name, meta)
    if h is None or a is None:
        missing = home_name if h is None else away_name
        teams = sorted(m["name"] for m in meta.values())
        print(f"Could not resolve team: {missing!r}")
        print("Known teams: " + ", ".join(teams))
        return None

    spec = Specialist2026().fit(
        [(int(r["home_team_id"]), int(r["away_team_id"]), _result(r))
         for r in b["by_id"].values() if r["status"] == "Completed"]
    )
    pH, pD, pA = spec.predict_wdl(h, a)
    hn, an = meta[h]["name"], meta[a]["name"]

    print(f"\n  {hn}  vs  {an}")
    print("  " + "-" * (len(hn) + len(an) + 6))
    print(f"  {hn} win : {pH:5.1%}")
    print(f"  Draw{'':>{max(0,len(hn)-4)}} : {pD:5.1%}")
    print(f"  {an} win : {pA:5.1%}")
    if knockout:
        adv_home = pH + pD / 2
        winner, p = (hn, adv_home) if adv_home >= 0.5 else (an, 1 - adv_home)
        print(f"  → advances (knockout): {winner} ({p:.0%})")
    else:
        pick = {"H": hn, "D": "Draw", "A": an}[["H", "D", "A"][int(max(
            range(3), key=[pH, pD, pA].__getitem__))]]
        print(f"  → most likely: {pick}")
    return pH, pD, pA


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    knockout = False
    for flag in ("--knockout", "-k"):
        if flag in argv:
            argv = [x for x in argv if x != flag]
            knockout = True
    if len(argv) != 2:
        print('Usage: python -m wcpredictor.sim.matchup "Team A" "Team B" [--knockout]')
        return
    predict_matchup(argv[0], argv[1], knockout)


if __name__ == "__main__":
    main()
