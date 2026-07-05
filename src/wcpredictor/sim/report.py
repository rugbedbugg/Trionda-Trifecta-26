"""Render the simulated tournament to a single, neatly formatted file.

Produces outputs/tournament_2026_prediction.md:
  * a champion banner and an ASCII bracket of the final eight,
  * clean group-stage standings tables (official FIFA order),
  * the full knockout results, round by round.
"""
from __future__ import annotations

from .. import paths
from . import bracket

_STAGE_ORDER = [
    bracket.STAGE_R32, bracket.STAGE_R16, bracket.STAGE_QF,
    bracket.STAGE_SF, bracket.STAGE_3RD, bracket.STAGE_FINAL,
]


def _code(b, tid):
    return b["meta"][tid]["fifa_code"]


def _name(b, tid):
    return b["meta"][tid]["name"]


def _ascii_last8(b, node) -> str:
    """Connected bracket for quarter-finals -> final (fifa codes).

    Structure (which QF feeds which SF) comes from the bracket blueprint `b`;
    the teams and winners come from the simulated results `node`."""
    struct = {n["match_id"]: n for n in b["ko"]}
    sf_ids = sorted(mid for mid, n in struct.items() if n["stage"] == bracket.STAGE_SF)
    final_id = next(mid for mid, n in struct.items() if n["stage"] == bracket.STAGE_FINAL)
    if len(sf_ids) != 2:
        return ""
    # QF order so semi-finals sit adjacent: [SF_A top, SF_A bot, SF_B top, SF_B bot]
    qf_ids = [struct[sf_ids[0]]["home_src"][1], struct[sf_ids[0]]["away_src"][1],
              struct[sf_ids[1]]["home_src"][1], struct[sf_ids[1]]["away_src"][1]]
    qf = [node[i] for i in qf_ids]
    c = []
    for n in qf:
        c += [_code(b, n["home"]), _code(b, n["away"])]
    w = [_code(b, n["winner"]) for n in qf]                 # QF winners
    f1, f2 = _code(b, node[sf_ids[0]]["winner"]), _code(b, node[sf_ids[1]]["winner"])
    ch = _code(b, node[final_id]["winner"])

    p = "{:>3}".format  # codes are 3-wide -> fixed columns line up
    return "\n".join([
        f"{p(c[0])} ┐",
        f"    ├── {p(w[0])} ┐",
        f"{p(c[1])} ┘       │",
        f"            ├── {p(f1)} ┐",
        f"{p(c[2])} ┐       │       │",
        f"    ├── {p(w[1])} ┘       │",
        f"{p(c[3])} ┘               │",
        f"                    ├── {p(ch)}  🏆",
        f"{p(c[4])} ┐               │",
        f"    ├── {p(w[2])} ┐       │",
        f"{p(c[5])} ┘       │       │",
        f"            ├── {p(f2)} ┘",
        f"{p(c[6])} ┐       │",
        f"    ├── {p(w[3])} ┘",
        f"{p(c[7])} ┘",
    ])


def _group_table(b, group, order, rec) -> list[str]:
    # No goal columns — the model predicts outcomes (W/D/L), not scores.
    lines = [
        f"**Group {group}**", "",
        "| Pos | Team | P | W | D | L | Pts | |",
        "| --- | --- | -: | -: | -: | -: | -: | --- |",
    ]
    for i, t in enumerate(order):
        r = rec[t]
        tag = "✅ Q" if i < 2 else ("▸ 3rd" if i == 2 else "")
        lines.append(
            f"| {i+1} | {_name(b, t)} | {r['p']} | {r['w']} | {r['d']} | {r['l']} "
            f"| **{r['pts']}** | {tag} |"
        )
    lines.append("")
    return lines


def write_report(b, group_order, recs, best_thirds, node,
                 champion, runner_up, third_place) -> None:
    L = []
    L += [
        "# 🏆 2026 FIFA World Cup — Model Prediction",
        "",
        "*Predicted from pre-2026 data only. The model predicts match outcomes "
        "(W/D/L), not scorelines, so group ties break on points → head-to-head → "
        "team rating; knockout ties are advanced by model win probability.*",
        "",
        "## Predicted podium",
        "",
        f"| | Team |",
        f"| --- | --- |",
        f"| 🥇 Champion | **{_name(b, champion)}** |",
        f"| 🥈 Runner-up | {_name(b, runner_up)} |",
        f"| 🥉 Third place | {_name(b, third_place)} |",
        "",
        "## Bracket — the final eight",
        "",
        "```",
        _ascii_last8(b, node),
        "```",
        "",
    ]

    L += ["## Group stage", "",
          "Top two of each group qualify (✅), plus the eight best third-placed "
          "teams (﹡).", ""]
    for group in sorted(group_order):
        L += _group_table(b, group, group_order[group], recs[group])
    thirds = ", ".join(_name(b, e["team"]) for e in best_thirds)
    L += [f"**8 best third-placed qualifiers:** {thirds}", ""]

    L += ["## Knockout stage", ""]
    for stage in _STAGE_ORDER:
        stage_nodes = sorted((n for n in node.values() if n["stage"] == stage),
                             key=lambda n: n["match_id"])
        if not stage_nodes:
            continue
        L += [f"### {bracket.STAGE_NAMES[stage]}", "",
              "| Match | P(win / draw / lose) | Advances |", "| --- | :-: | --- |"]
        for n in stage_nodes:
            ph, pd_, pa = n["proba"]
            L.append(
                f"| {_name(b, n['home'])} vs {_name(b, n['away'])} "
                f"| {ph:.0%} / {pd_:.0%} / {pa:.0%} | **{_name(b, n['winner'])}** |"
            )
        L.append("")

    path = paths.OUTPUT_DIR / "tournament_2026_prediction.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
