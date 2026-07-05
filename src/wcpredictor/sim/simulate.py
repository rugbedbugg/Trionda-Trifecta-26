"""Simulate the entire 2026 World Cup, group stage to final.

Flow:
  1. Predict all 72 group matches as W/D/L outcomes (no scoreline — exact scores
     are not something the model predicts reliably).
  2. Rank each group by the official FIFA rules; take the 12 winners, 12
     runners-up and the 8 best third-placed teams -> 32 qualifiers.
  3. Drop the qualifiers into the official bracket (see bracket.py).
  4. Predict every knockout, advancing the winner (draws in a knockout are
     decided by the model's advance probability, i.e. a weighted shootout).
  5. Crown a champion; write a full report and a machine-readable prediction
     file that validate.py scores against the real results.

All predictions use only pre-2026 information (see predict.py), so this is an
honest "what the model thought would happen", not a retrofit.
"""
from __future__ import annotations

import json
from collections import defaultdict

from .. import paths
from . import bracket, report, rules
from .model2026 import HistoryRatingModel

BEST_THIRDS = 8


def _predict_groups(rating: HistoryRatingModel, b: dict):
    meta = b["meta"]
    group_matches = [
        (int(r["home_team_id"]), int(r["away_team_id"]))
        for mid, r in b["by_id"].items()
        if r["stage_id"] == bracket.STAGE_GROUP
    ]
    results = defaultdict(list)   # group -> [(home_id, away_id, hg, ag)]
    for home_id, away_id in group_matches:
        # Predict the OUTCOME (W/D/L), not a scoreline. We encode it as a minimal
        # unit result purely so the standings machinery can tally points; goal
        # magnitudes are never predicted or used (see rank with use_goals=False).
        proba = rating.predict_wdl(home_id, away_id)
        outcome = int(proba.argmax())          # 0=home win, 1=draw, 2=away win
        hg, ag = {0: (1, 0), 1: (0, 0), 2: (0, 1)}[outcome]
        results[meta[home_id]["group"]].append((home_id, away_id, hg, ag))
    return results


def _rank_all_groups(b: dict, results: dict):
    meta = b["meta"]
    seed = {tid: meta[tid]["elo"] for tid in meta}
    group_order, positions, recs, third_entries = {}, {}, {}, []
    for group in sorted(results):
        team_ids = [t for t in meta if meta[t]["group"] == group]
        # No predicted scores -> rank by points, head-to-head, then team rating.
        order, rec = rules.rank_group(team_ids, results[group], seed, use_goals=False)
        group_order[group] = order
        recs[group] = rec
        for pos, tid in enumerate(order, 1):
            positions[tid] = (group, pos)
        third = order[2]
        third_entries.append({"team": third, "group": group, "pts": rec[third]["pts"]})
    best_thirds = rules.rank_thirds(third_entries, seed, use_goals=False)[:BEST_THIRDS]
    return group_order, positions, recs, best_thirds


def _assign_third_slots(b: dict, best_thirds: list):
    """Map the model's 8 best thirds onto the 8 third-place bracket slots.

    Preference is given to a third from the group the official bracket assigned
    to that slot; any left over fill the remaining slots in ranked order. When
    the model's qualifying thirds match reality this reproduces the official
    allocation exactly."""
    meta = b["meta"]
    ranked = [e["team"] for e in best_thirds]
    assignment, remaining = {}, list(ranked)
    for slot in b["third_slots"]:
        match = next((t for t in remaining if meta[t]["group"] == slot["group"]), None)
        if match is not None:
            assignment[(slot["match_id"], slot["side"])] = match
            remaining.remove(match)
    leftover_slots = [s for s in b["third_slots"]
                      if (s["match_id"], s["side"]) not in assignment]
    for slot, tid in zip(leftover_slots, remaining):
        assignment[(slot["match_id"], slot["side"])] = tid
    return assignment


def _simulate_knockouts(rating: HistoryRatingModel, b: dict, group_order, third_assign):
    meta = b["meta"]
    r32_by_id = {n["match_id"]: n for n in b["r32"]}
    ko_by_id = {n["match_id"]: n for n in b["ko"]}
    node = {}   # match_id -> result dict

    def slot_team(match_id, side, slot):
        group, pos = slot
        if pos in (1, 2):
            return group_order[group][pos - 1]
        return third_assign[(match_id, side)]

    def play(match_id, stage, home_id, away_id):
        proba = rating.predict_wdl(home_id, away_id)
        adv_home = float(proba[0] + proba[1] / 2)
        winner = home_id if adv_home >= 0.5 else away_id
        loser = away_id if winner == home_id else home_id
        node[match_id] = {
            "match_id": match_id, "stage": stage,
            "home": home_id, "away": away_id,
            "proba": [round(float(x), 4) for x in proba],
            "adv_home": round(adv_home, 4), "winner": winner, "loser": loser,
        }

    for mid in sorted(list(r32_by_id) + list(ko_by_id)):
        if mid in r32_by_id:
            n = r32_by_id[mid]
            home_id = slot_team(mid, "home", n["home_slot"])
            away_id = slot_team(mid, "away", n["away_slot"])
        else:
            n = ko_by_id[mid]
            home_id = node[n["home_src"][1]][n["home_src"][0]]
            away_id = node[n["away_src"][1]][n["away_src"][0]]
        play(mid, n["stage"], home_id, away_id)
    return node


def simulate(rating: HistoryRatingModel | None = None, b: dict | None = None,
             report_stdout: bool = True):
    rating = rating or HistoryRatingModel()
    b = b or bracket.derive()
    results = _predict_groups(rating, b)
    group_order, positions, recs, best_thirds = _rank_all_groups(b, results)
    third_assign = _assign_third_slots(b, best_thirds)
    node = _simulate_knockouts(rating, b, group_order, third_assign)

    final_id = next(n["match_id"] for n in b["ko"] if n["stage"] == bracket.STAGE_FINAL)
    third_id = next(n["match_id"] for n in b["ko"] if n["stage"] == bracket.STAGE_3RD)
    champion = node[final_id]["winner"]
    runner_up = node[final_id]["loser"]
    third_place = node[third_id]["winner"]

    out = {
        "positions": {str(k): v for k, v in positions.items()},
        "group_order": {g: [int(t) for t in order] for g, order in group_order.items()},
        "best_thirds": [int(e["team"]) for e in best_thirds],
        "nodes": {str(mid): n for mid, n in node.items()},
        "champion": int(champion), "runner_up": int(runner_up),
        "third_place": int(third_place),
    }
    paths.ensure_dirs()
    (paths.OUTPUT_DIR / "tournament_simulation.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    report.write_report(b, group_order, recs, best_thirds, node,
                        champion, runner_up, third_place)
    if report_stdout:
        _report(b, group_order, recs, best_thirds, node, champion, runner_up, third_place)
    return out


# --------------------------------------------------------------------------- #
#  Reporting                                                                   #
# --------------------------------------------------------------------------- #
def _name(b, tid):
    return b["meta"][tid]["fifa_code"]


def _full(b, tid):
    return b["meta"][tid]["name"]


def _report(b, group_order, recs, best_thirds, node, champion, runner_up, third_place):
    lines = ["# 2026 World Cup — full model simulation", "",
             "Predicted from pre-2026 data only. Group qualification by official "
             "FIFA rules; knockouts advanced by model probability.", ""]

    print("\n" + "=" * 64)
    print(" GROUP STAGE (predicted) — top 2 qualify, plus 8 best thirds")
    print("=" * 64)
    lines += ["## Group stage", ""]
    for group in sorted(group_order):
        order = group_order[group]
        rec = recs[group]
        header = f"Group {group}: " + "  ".join(
            f"{_name(b, t)}({rec[t]['pts']}p {rec[t]['w']}-{rec[t]['d']}-{rec[t]['l']})"
            for t in order)
        print(" " + header)
        lines.append(f"- **Group {group}** — " + ", ".join(
            f"{_full(b, t)} ({rec[t]['pts']} pts, "
            f"{rec[t]['w']}W-{rec[t]['d']}D-{rec[t]['l']}L, "
            f"{'Q' if i < 2 else '3rd' if i == 2 else '-'})"
            for i, t in enumerate(order)))
    thirds_str = ", ".join(_name(b, e["team"]) for e in best_thirds)
    print("\n 8 best third-placed teams advancing: " + thirds_str)
    lines += ["", f"**8 best third-placed qualifiers:** {thirds_str}", "", "## Knockouts", ""]

    print("\n" + "=" * 64)
    print(" KNOCKOUT BRACKET (predicted)")
    print("=" * 64)
    stage_order = [bracket.STAGE_R32, bracket.STAGE_R16, bracket.STAGE_QF,
                   bracket.STAGE_SF, bracket.STAGE_3RD, bracket.STAGE_FINAL]
    for stage in stage_order:
        stage_nodes = [n for n in node.values() if n["stage"] == stage]
        if not stage_nodes:
            continue
        title = bracket.STAGE_NAMES[stage]
        print(f"\n {title}")
        lines += [f"### {title}", ""]
        for n in sorted(stage_nodes, key=lambda x: x["match_id"]):
            h, a = n["home"], n["away"]
            p = n["adv_home"] if n["winner"] == h else 1 - n["adv_home"]
            win = _name(b, n["winner"])
            print(f"   {_name(b,h)} v {_name(b,a)}  →  {win} ({p:.0%})")
            lines.append(f"- {_full(b,h)} v {_full(b,a)} → **{_full(b,n['winner'])}** ({p:.0%})")
        lines.append("")

    print("\n" + "=" * 64)
    print(f"  PREDICTED CHAMPION:  {_full(b, champion)}")
    print(f"  Runner-up:           {_full(b, runner_up)}")
    print(f"  Third place:         {_full(b, third_place)}")
    print("=" * 64)
    lines += ["## Result", "",
              f"- 🏆 **Champion: {_full(b, champion)}**",
              f"- 🥈 Runner-up: {_full(b, runner_up)}",
              f"- 🥉 Third place: {_full(b, third_place)}", ""]
    (paths.OUTPUT_DIR / "tournament_simulation.md").write_text(
        "\n".join(lines), encoding="utf-8")


def run():
    simulate()
    print(f"\n[simulate] report -> {paths.OUTPUT_DIR / 'tournament_simulation.md'}")
    print(f"[simulate] predictions -> {paths.OUTPUT_DIR / 'tournament_simulation.json'}")


if __name__ == "__main__":
    run()
