"""Validate the predictions against what actually happened.

Two independent lenses:

  A. Match-level accuracy — feed the model the *real* fixtures that were played
     (all 72 group matches + every completed knockout tie) and ask: did it call
     the right result / the right team to advance? This isolates the model's
     per-match skill from any bracket knock-on effects, and is the honest "how
     much did it get wrong" number, plus Brier / log loss vs reality.

  B. Bracket-level accuracy — take the full from-scratch simulation and compare
     its bracket to reality: correct group winners, how many of its 32
     qualifiers actually qualified, and round-by-round survivor overlap.

The 2026 tournament is only played out through the quarter-finals in the data,
so later rounds (and the champion) can't be scored yet; the report says so
rather than inventing a number.
"""
from __future__ import annotations

import numpy as np

from ..evaluate import brier_multiclass, log_loss_wdl
from ..models import WDL
from . import bracket, simulate
from .model2026 import HistoryRatingModel
from .predict import MatchPredictor


def _actual_result(row):
    hg, ag = row["home_score"], row["away_score"]
    if hg > ag:
        return "H"
    if hg < ag:
        return "A"
    return "D"


def match_level(pred: MatchPredictor, b: dict):
    meta = b["meta"]
    proba_rows, y_rows = [], []
    group_hit = group_n = ko_hit = ko_n = 0

    for mid, row in b["by_id"].items():
        if row["status"] != "Completed":
            continue
        home_id, away_id = int(row["home_team_id"]), int(row["away_team_id"])
        is_ko = row["stage_id"] != bracket.STAGE_GROUP
        p = pred.predict(meta[home_id]["canonical"], meta[away_id]["canonical"], int(is_ko))
        proba_rows.append(p["proba"])
        y_rows.append(_actual_result(row))

        if is_ko:
            ko_n += 1
            pred_adv = home_id if p["adv_home"] >= 0.5 else away_id
            if pred_adv == bracket.actual_advancer(row):
                ko_hit += 1
        else:
            group_n += 1
            pred_res = WDL[int(np.argmax(p["proba"]))]
            if pred_res == _actual_result(row):
                group_hit += 1

    proba = np.array(proba_rows)
    y = np.array(y_rows)
    # Reference baseline: back the higher-Elo team every time (draw impossible).
    higher_elo_hit = 0
    for mid, row in b["by_id"].items():
        if row["status"] != "Completed":
            continue
        h, a = int(row["home_team_id"]), int(row["away_team_id"])
        pick = "H" if meta[h]["elo"] >= meta[a]["elo"] else "A"
        higher_elo_hit += int(pick == _actual_result(row))

    return {
        "n": len(y),
        "group_result_acc": group_hit / group_n if group_n else float("nan"),
        "ko_advance_acc": ko_hit / ko_n if ko_n else float("nan"),
        "overall_result_acc": float((np.array(WDL)[proba.argmax(1)] == y).mean()),
        "higher_elo_acc": higher_elo_hit / len(y),
        "brier": brier_multiclass(proba, y),
        "log_loss": log_loss_wdl(proba, y),
        "group_n": group_n, "ko_n": ko_n,
    }


def sim_model_accuracy(b: dict):
    """Match-level accuracy of the from-scratch sim's model (history rating->outcome
    curve applied with current 2026 Elo — "use both eras") on real fixtures."""
    rm = HistoryRatingModel()
    oh = ot = kh = kt = 0
    for row in b["by_id"].values():
        if row["status"] != "Completed":
            continue
        h, a = int(row["home_team_id"]), int(row["away_team_id"])
        proba = rm.predict_wdl(h, a)
        pred = WDL[int(proba.argmax())]
        ot += 1
        oh += pred == _actual_result(row)
        if row["stage_id"] != bracket.STAGE_GROUP:
            kt += 1
            adv = h if (proba[0] + proba[1] / 2) >= 0.5 else a
            kh += adv == int(bracket.actual_advancer(row))
    return {"overall": oh / ot, "ko_adv": kh / kt}


def _round_survivors_actual(b: dict):
    """Teams that actually reached each knockout round (from real participants)."""
    survivors = {}
    for stage in (bracket.STAGE_R32, bracket.STAGE_R16, bracket.STAGE_QF):
        teams = set()
        for mid, row in b["by_id"].items():
            if row["stage_id"] == stage:
                teams.add(int(row["home_team_id"]))
                teams.add(int(row["away_team_id"]))
        survivors[stage] = teams
    return survivors


def _round_survivors_sim(sim: dict):
    survivors = {}
    for stage in (bracket.STAGE_R32, bracket.STAGE_R16, bracket.STAGE_QF):
        teams = set()
        for n in sim["nodes"].values():
            if n["stage"] == stage:
                teams.add(int(n["home"]))
                teams.add(int(n["away"]))
        survivors[stage] = teams
    return survivors


def bracket_level(b: dict, sim: dict):
    meta = b["meta"]
    # Actual qualifiers = everyone who reached the real Round of 32.
    actual_r32 = set()
    for mid, row in b["by_id"].items():
        if row["stage_id"] == bracket.STAGE_R32:
            actual_r32.add(int(row["home_team_id"]))
            actual_r32.add(int(row["away_team_id"]))

    sim_qualifiers = set()
    for g, order in sim["group_order"].items():
        sim_qualifiers.update(int(t) for t in order[:2])
    sim_qualifiers.update(int(t) for t in sim["best_thirds"])

    # Group winners.
    gw_correct = sum(
        int(int(sim["group_order"][g][0]) == int(b["group_order"][g][0]))
        for g in b["group_order"]
    )

    act = _round_survivors_actual(b)
    smv = _round_survivors_sim(sim)
    round_overlap = {
        bracket.STAGE_NAMES[s]: (len(act[s] & smv[s]), len(act[s]))
        for s in (bracket.STAGE_R32, bracket.STAGE_R16, bracket.STAGE_QF)
    }

    return {
        "qualifiers_correct": (len(sim_qualifiers & actual_r32), 32),
        "group_winners_correct": (gw_correct, 12),
        "round_overlap": round_overlap,
        "missed_qualifiers": sorted(meta[t]["fifa_code"] for t in actual_r32 - sim_qualifiers),
        "wrong_qualifiers": sorted(meta[t]["fifa_code"] for t in sim_qualifiers - actual_r32),
    }


def run(pred: MatchPredictor | None = None, b: dict | None = None, sim: dict | None = None):
    pred = pred or MatchPredictor()
    b = b or bracket.derive()
    if sim is None:
        sim = simulate.simulate(b=b, report_stdout=False)

    m = match_level(pred, b)
    br = bracket_level(b, sim)

    print("\n" + "=" * 66)
    print(" VALIDATION vs ACTUAL RESULTS")
    print("=" * 66)
    print("\n A) Match-level accuracy on real fixtures already played")
    print(f"    matches scored              : {m['n']}  "
          f"({m['group_n']} group, {m['ko_n']} knockout)")
    print(f"    group result accuracy       : {m['group_result_acc']:.1%}")
    print(f"    knockout advancer accuracy  : {m['ko_advance_acc']:.1%}")
    print(f"    overall result accuracy     : {m['overall_result_acc']:.1%}")
    print(f"    (baseline: higher-Elo pick) : {m['higher_elo_acc']:.1%}")
    print(f"    Brier score (vs actual)     : {m['brier']:.4f}")
    print(f"    log loss   (vs actual)      : {m['log_loss']:.4f}")

    sm = sim_model_accuracy(b)
    print("\n B) From-scratch simulation (uses both eras: history rating→outcome "
          "curve + current 2026 Elo)")
    print(f"    sim model match accuracy    : {sm['overall']:.1%} overall, "
          f"{sm['ko_adv']:.1%} knockout advancer  (stale-Elo model was ~63.9%)")
    qc, qt = br["qualifiers_correct"]
    gw, gt = br["group_winners_correct"]
    print(f"    correct group winners       : {gw}/{gt}")
    print(f"    qualifiers also real R32     : {qc}/{qt}")
    for rnd, (hit, tot) in br["round_overlap"].items():
        print(f"    reached {rnd:<14}: {hit}/{tot} teams matched reality")
    print(f"\n    missed real qualifiers      : {', '.join(br['missed_qualifiers'])}")
    print(f"    predicted, didn't qualify   : {', '.join(br['wrong_qualifiers'])}")

    champ = b["meta"][sim["champion"]]["name"]
    print(f"\n    predicted champion          : {champ}")
    print("    actual champion             : not yet decided in the data "
          "(final still to be played) — re-run once results land.")
    print("=" * 66)
    return m, br


if __name__ == "__main__":
    run()
