"""Continue the REAL tournament — predict the actual remaining matches.

Unlike simulate.py (which invents a bracket from scratch), this takes the 2026
tournament exactly as it actually stands — real qualifiers, real results through
the last match played — and predicts what happens next:

  * the specialist model is trained on **every completed 2026 match**, so its
    predictions absorb all the true results that have happened so far;
  * the immediate next matches have real, known participants, so they are
    predicted purely from true data;
  * deeper rounds whose participants aren't decided yet are filled by advancing
    our own predicted winners, so we can project all the way to the champion.

Predictions are W/D/L outcomes only — the model doesn't claim exact scorelines.
Re-run it whenever the dataset refreshes with new results and the frontier of
true-data predictions moves forward.
"""
from __future__ import annotations

import warnings

import numpy as np

from .. import paths
from ..models import WDL
from . import bracket
from .model2026 import Specialist2026

WARMUP = 24  # matches before the walk-forward backtest starts scoring


def _result(row):
    hg, ag = row["home_score"], row["away_score"]
    return "H" if hg > ag else ("A" if ag > hg else "D")


def _completed_chrono(b):
    return sorted(
        ((mid, row) for mid, row in b["by_id"].items() if row["status"] == "Completed"),
        key=lambda kv: (str(kv[1]["date"]), int(kv[0])),
    )


def continue_tournament(b: dict | None = None):
    warnings.filterwarnings("ignore")
    b = b or bracket.derive()
    meta = b["meta"]
    completed = _completed_chrono(b)

    # The 2026-specialist (current Elo + squad value), trained on every real match
    # played so far, makes the W/D/L call. It absorbs the true results through its
    # training set, so predictions use everything that has actually happened.
    spec = Specialist2026().fit(
        [(int(r["home_team_id"]), int(r["away_team_id"]), _result(r)) for _, r in completed]
    )

    r32 = {n["match_id"]: n for n in b["r32"]}
    ko = {n["match_id"]: n for n in b["ko"]}
    node = {}

    def resolve(src):
        """(team_id, is_real) for a feeder reference."""
        kind, feeder = src
        f = node[feeder]
        return f[kind], f["actual"]

    for mid in sorted(list(r32) + list(ko)):
        row = b["by_id"][mid]
        if mid in r32:
            h, a = int(row["home_team_id"]), int(row["away_team_id"])
            h_real = a_real = True
        else:
            n = ko[mid]
            # Scheduled quarter-finals still list their real participants directly.
            if row["status"] != "Completed" and str(row.get("home_team_id")) not in ("", "nan") \
                    and not _isnan(row.get("home_team_id")):
                h, a = int(row["home_team_id"]), int(row["away_team_id"])
                h_real = a_real = True
            else:
                (h, h_real), (a, a_real) = resolve(n["home_src"]), resolve(n["away_src"])
        stage = (r32.get(mid) or ko.get(mid))["stage"]

        if row["status"] == "Completed":
            winner = int(bracket.actual_advancer(row))
            node[mid] = {"match_id": mid, "home": h, "away": a, "stage": stage,
                         "actual": True, "winner": winner,
                         "loser": a if winner == h else h, "inputs_real": True}
            continue

        # Scheduled -> predict the OUTCOME (W/D/L) with the specialist.
        proba = spec.predict_wdl(h, a)
        adv_home = float(proba[0] + proba[1] / 2)
        winner = h if adv_home >= 0.5 else a
        node[mid] = {"match_id": mid, "home": h, "away": a, "stage": stage,
                     "actual": False, "winner": winner,
                     "loser": a if winner == h else h,
                     "proba": list(proba),
                     "adv_home": adv_home, "inputs_real": bool(h_real and a_real)}

    return b, node, len(completed)


def _isnan(x):
    try:
        return np.isnan(float(x))
    except (TypeError, ValueError):
        return False


def backtest(b: dict | None = None, warmup: int = WARMUP, show_ko: bool = True):
    """Walk-forward accuracy of the live predictor on matches already played.

    For each completed match in chronological order we train the specialist on
    *only the earlier* matches, predict this one, and compare to the real result.
    Strictly leakage-safe, and the honest answer to "how accurate is continue_live
    on games that have already happened?"
    """
    b = b or bracket.derive()
    meta = b["meta"]
    completed = _completed_chrono(b)
    spec = Specialist2026()

    tally = {"overall": [0, 0], "group": [0, 0], "ko_result": [0, 0], "ko_adv": [0, 0]}
    ko_rows = []
    for i, (mid, row) in enumerate(completed):
        if i < warmup:
            continue
        spec.fit([(int(r["home_team_id"]), int(r["away_team_id"]), _result(r))
                  for _, r in completed[:i]])
        h, a = int(row["home_team_id"]), int(row["away_team_id"])
        proba = spec.predict_wdl(h, a)
        pred_res = WDL[int(proba.argmax())]
        y = _result(row)
        ko = row["stage_id"] != bracket.STAGE_GROUP
        tally["overall"][0] += pred_res == y; tally["overall"][1] += 1
        if ko:
            tally["ko_result"][0] += pred_res == y; tally["ko_result"][1] += 1
            adv = h if (proba[0] + proba[1] / 2) >= 0.5 else a
            true_adv = int(bracket.actual_advancer(row))
            hit = adv == true_adv
            tally["ko_adv"][0] += hit; tally["ko_adv"][1] += 1
            ko_rows.append((meta[h]["name"], meta[a]["name"], int(row["home_score"]),
                            int(row["away_score"]), meta[true_adv]["name"], hit))
        else:
            tally["group"][0] += pred_res == y; tally["group"][1] += 1

    print("\n" + "=" * 70)
    print(" CONTINUE_LIVE BACKTEST — accuracy on matches already played")
    print(f" walk-forward, each match predicted from only the matches before it")
    print("=" * 70)
    for label, key in (("Overall result accuracy", "overall"),
                       ("Group result accuracy", "group"),
                       ("Knockout result accuracy", "ko_result"),
                       ("Knockout advancer accuracy", "ko_adv")):
        hit, tot = tally[key]
        print(f" {label:<30}: {hit/tot:.1%}  ({hit}/{tot})")
    if show_ko and ko_rows:
        print("\n knockout ties (predicted advancer vs actual):")
        for h, a, hs, as_, adv, hit in ko_rows:
            mark = "✓" if hit else "✗"
            pens = " (pens)" if hs == as_ else ""
            print(f"   {mark} {h} {hs}-{as_}{pens} {a}  →  {adv}")
    return tally


def _fc(b, tid):
    return b["meta"][tid]["name"]


def _row_line(b, n):
    h, a = n["home"], n["away"]
    pH, pD, pA = n["proba"]
    return (f"{_fc(b,h)} vs {_fc(b,a)}", f"{pH:.0%}/{pD:.0%}/{pA:.0%}",
            _fc(b, n["winner"]))


def _scorelines(b, scheduled):
    """Projected scoreline for every remaining fixture, from the Poisson score
    model (`goals.py`) — the same model `sim.goals` uses, fit on all completed
    2026 matches. Remaining ties are all knockouts, so the KO goal deflation is
    applied. Returns (match, exp_home, exp_away, home_goals, away_goals) rows."""
    from . import intl
    from .goals import KO_FACTOR, PoissonScoreModel, _completed
    model = PoissonScoreModel("glm", use_intl=intl.available(),
                              ko_factor=KO_FACTOR).fit(_completed(b))
    rows = []
    for n in sorted(scheduled, key=lambda x: x["match_id"]):
        h, a = n["home"], n["away"]
        r = model.predict(h, a, is_knockout=1)
        i, j = r["most_likely"]
        rows.append((f"{_fc(b,h)} vs {_fc(b,a)}", r["exp_home"], r["exp_away"], i, j))
    return rows


def run(b: dict | None = None, do_backtest: bool = True, scores: bool = False):
    b = b or bracket.derive()
    bt = backtest(b) if do_backtest else None
    b, node, n_completed = continue_tournament(b)
    scheduled = [n for n in node.values() if not n["actual"]]
    next_up = [n for n in scheduled if n["inputs_real"]]
    projected = [n for n in scheduled if not n["inputs_real"]]

    lines = ["# 2026 World Cup — live continuation", "",
             f"Predicted from the **real results of all {n_completed} matches played "
             "so far**, using the 2026-specialist model (current Elo + squad market "
             "value).", ""]
    if bt:
        o_h, o_t = bt["overall"]; k_h, k_t = bt["ko_adv"]
        lines += [f"> **Backtest accuracy** (walk-forward on matches already played): "
                  f"{o_h/o_t:.1%} overall, {k_h/k_t:.1%} on knockout ties.", ""]

    print("\n" + "=" * 70)
    print(" CONTINUE THE REAL TOURNAMENT")
    print(f" predictions built from {n_completed} true completed matches")
    print("=" * 70)

    if not scheduled:
        print("\n No scheduled matches remain in the data — the tournament is complete.")
    else:
        print("\n NEXT MATCHES — predicted purely from true results so far")
        print(f" {'Match':<28}{'P(win/draw/lose)':>20}   Winner")
        print(" " + "-" * 66)
        lines += ["## Next matches — predicted purely from true results so far", "",
                  "| Match | P(win / draw / lose) | Predicted winner |",
                  "| --- | :-: | --- |"]
        for n in sorted(next_up, key=lambda x: x["match_id"]):
            m, prob, win = _row_line(b, n)
            print(f" {m:<28}{prob:>20}   {win}")
            lines.append(f"| {m} | {prob} | **{win}** |")

        if projected:
            print("\n PROJECTED ONWARD — contingent on the predicted results above")
            lines += ["", "## Projected onward (contingent on predictions above)", "",
                      "| Round | Match | P(win / draw / lose) | Predicted winner |",
                      "| --- | --- | :-: | --- |"]
            for n in sorted(projected, key=lambda x: x["match_id"]):
                m, prob, win = _row_line(b, n)
                rnd = bracket.STAGE_NAMES[n["stage"]]
                print(f"   {rnd:<16}{m:<32}-> {win}")
                lines.append(f"| {rnd} | {m} | {prob} | **{win}** |")

        if scores:
            score_rows = _scorelines(b, scheduled)
            print("\n PROJECTED SCORELINES — most-likely score (goals model)")
            print(f" {'Match':<28}{'exp goals':>12}   Score")
            print(" " + "-" * 66)
            lines += ["", "## Projected scorelines (most-likely score, goals model)", "",
                      "| Match | Expected goals | Most-likely score |",
                      "| --- | :-: | :-: |"]
            for m, eh, ea, i, j in score_rows:
                print(f" {m:<28}{f'{eh:.2f}-{ea:.2f}':>12}   {i}-{j}")
                lines.append(f"| {m} | {eh:.2f}–{ea:.2f} | **{i}–{j}** |")
            print("\n (Scorelines are the mode of a wide distribution — read them as a"
                  " lean, not a lock; see `sim.goals` for the full spread.)")

        final = next((n for n in node.values() if n["stage"] == bracket.STAGE_FINAL), None)
        if final:
            champ = _fc(b, final["winner"])
            print("\n" + "=" * 70)
            print(f"  PROJECTED CHAMPION: {champ}")
            print("=" * 70)
            lines += ["", f"## Projected champion: **{champ}**", ""]

    paths.ensure_dirs()
    (paths.OUTPUT_DIR / "live_continuation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[continue_live] report -> {paths.OUTPUT_DIR / 'live_continuation.md'}")
    return node


def main(argv=None):
    import sys
    argv = argv if argv is not None else sys.argv[1:]
    scores = any(f in argv for f in ("--scores", "-s"))
    run(scores=scores)


if __name__ == "__main__":
    main()
