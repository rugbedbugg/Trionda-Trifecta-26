"""State-of-the-tournament analyst — one command, a full prose briefing.

A thin orchestration layer over the deterministic tools. It asks each model the
question it is best at, stitches the answers into a single narrative, and — this
is the point — **never invents a number**. Every figure is read straight from the
underlying command output (the walk-forward backtest, the live continuation, the
goals model), so the briefing is exactly as reproducible as the models it quotes.

It also remembers the last briefing and reports what *changed* since — a new
result landing, the projected champion flipping — which is the actual analyst
value on top of the raw predictions.

    python -m wcpredictor.analyst            # brief from the current data
    python -m wcpredictor.analyst --retrain  # rebuild the store first, then brief
"""
from __future__ import annotations

import json
import warnings

from . import paths
from .sim import bracket, continue_live

_STATE = None  # set lazily to paths.OUTPUT_DIR / "analyst_state.json"


def _state_path():
    return paths.OUTPUT_DIR / "analyst_state.json"


def _pct(hit_tot) -> str:
    h, t = hit_tot
    return f"{h/t:.1%} ({h}/{t})" if t else "n/a"


def _load_state() -> dict:
    p = _state_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def _save_state(state: dict) -> None:
    paths.ensure_dirs()
    _state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


def _changes(prior: dict, now: dict) -> list[str]:
    """Human-readable diff between the previous briefing and this one."""
    if not prior:
        return ["First briefing — no prior state to compare against."]
    out = []
    new_games = now["completed"] - prior.get("completed", 0)
    if new_games > 0:
        out.append(f"{new_games} new result(s) since the last briefing.")
    if prior.get("champion") != now["champion"]:
        out.append(f"Projected champion changed: "
                   f"{prior.get('champion')} → {now['champion']}.")
    flips = [f"match {mid}: {prior['projected'].get(mid)} → {win}"
             for mid, win in now["projected"].items()
             if mid in prior.get("projected", {}) and prior["projected"][mid] != win]
    if flips:
        out.append("Projected winner flipped for " + "; ".join(flips) + ".")
    if not out:
        out.append("No change since the last briefing.")
    return out


def brief(retrain: bool = False, do_backtest: bool = True) -> dict:
    warnings.filterwarnings("ignore")
    if retrain:
        from . import pipeline
        pipeline.run()

    b = bracket.derive()
    meta = b["meta"]
    n_total = len(b["by_id"])
    n_done = sum(1 for r in b["by_id"].values() if r["status"] == "Completed")

    # 1) How good has the live model been — walk-forward, leakage-safe.
    tally = continue_live.backtest(b, show_ko=False) if do_backtest else None

    # 2) Project the rest of the REAL tournament from all results so far.
    _, node, _ = continue_live.continue_tournament(b)
    scheduled = [n for n in node.values() if not n["actual"]]
    final = next((n for n in node.values() if n["stage"] == bracket.STAGE_FINAL), None)
    champion = meta[final["winner"]]["name"] if final else None

    # 3) A most-likely scoreline for each remaining fixture (goals model).
    score_rows = continue_live._scorelines(b, scheduled) if scheduled else []
    score_by_match = {m: (i, j) for m, _, _, i, j in score_rows}

    # 4) What changed since last time.
    now_state = {
        "completed": n_done,
        "champion": champion,
        "projected": {str(n["match_id"]): meta[n["winner"]]["name"] for n in scheduled},
    }
    deltas = _changes(_load_state(), now_state)
    _save_state(now_state)

    _write_and_print(b, meta, n_done, n_total, tally, node, scheduled,
                     score_by_match, champion, deltas)
    return {"completed": n_done, "champion": champion, "changes": deltas}


def _write_and_print(b, meta, n_done, n_total, tally, node, scheduled,
                     score_by_match, champion, deltas):
    def name(tid):
        return meta[tid]["name"]

    print("\n" + "=" * 70)
    print(" TOURNAMENT ANALYST BRIEFING")
    print("=" * 70)
    print(f"\n State of play      : {n_done}/{n_total} matches played")
    if champion:
        print(f" Projected champion : {champion}")
    if tally:
        print(f" Live model track record (walk-forward on played matches):")
        print(f"   overall {_pct(tally['overall'])} | "
              f"knockout advancer {_pct(tally['ko_adv'])}")

    print("\n What changed since the last briefing:")
    for d in deltas:
        print(f"   • {d}")

    lines = ["# Tournament analyst briefing", "",
             f"**State of play:** {n_done}/{n_total} matches played."]
    if champion:
        lines += [f"**Projected champion:** {champion}."]
    if tally:
        lines += ["", f"**Live model track record** (walk-forward, leakage-safe): "
                  f"{_pct(tally['overall'])} overall, "
                  f"{_pct(tally['ko_adv'])} on knockout advancers."]
    lines += ["", "## What changed since the last briefing", ""]
    lines += [f"- {d}" for d in deltas]

    if scheduled:
        print("\n Remaining fixtures (predicted from real results so far):")
        print(f"   {'Match':<28}{'W/D/L':>16}   Winner     Score")
        print("   " + "-" * 68)
        lines += ["", "## Remaining fixtures", "",
                  "| Round | Match | P(win/draw/lose) | Winner | Likely score |",
                  "| --- | --- | :-: | --- | :-: |"]
        for n in sorted(scheduled, key=lambda x: x["match_id"]):
            match = f"{name(n['home'])} vs {name(n['away'])}"
            pH, pD, pA = n["proba"]
            wdl = f"{pH:.0%}/{pD:.0%}/{pA:.0%}"
            win = name(n["winner"])
            i, j = score_by_match.get(match, ("?", "?"))
            rnd = bracket.STAGE_NAMES[n["stage"]]
            print(f"   {match:<28}{wdl:>16}   {win:<10} {i}-{j}")
            lines.append(f"| {rnd} | {match} | {wdl} | **{win}** | {i}–{j} |")
        lines += ["", "_Scorelines are the mode of a wide distribution — a lean, "
                  "not a lock. Fixtures past the next real round are contingent on "
                  "the predictions above._"]
    else:
        print("\n No scheduled matches remain — the tournament is complete.")
        lines += ["", "_No scheduled matches remain — the tournament is complete._"]

    out = paths.OUTPUT_DIR / "analyst_brief.md"
    paths.ensure_dirs()
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[analyst] briefing -> {out}")


def main(argv=None):
    import sys
    argv = argv if argv is not None else sys.argv[1:]
    brief(retrain="--retrain" in argv, do_backtest="--no-backtest" not in argv)


if __name__ == "__main__":
    main()
