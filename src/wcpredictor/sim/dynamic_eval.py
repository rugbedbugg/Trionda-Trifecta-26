"""Dynamic (walk-forward) evaluation.

The tournament simulator predicts every 2026 match from a *frozen* pre-tournament
snapshot — it never lets a team's 2026 form influence a later 2026 prediction,
because when simulating from scratch those later results don't exist yet.

This evaluation asks a different, complementary question:

    if we predict each real match using the REAL results of every earlier match
    already played in this tournament, how much better do we do?

So we walk the actual 2026 fixtures in chronological order. Before each match we
featurise the two teams from their *current* rolling state — Elo, form, pedigree
that already absorbed every earlier 2026 result — predict, then fold in the
match's real result and move on. It stays strictly leakage-safe (only earlier
matches are ever used) and is a fairer test of the model when in-tournament
information is available.

We run the frozen ("static") predictor over the same fixtures side by side, so
the report shows exactly what updating on real results buys you.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd

from .. import paths
from ..evaluate import brier_multiclass, log_loss_wdl
from ..features import CORE_FEATURES, _TeamState, core_feature_row, update_after_match
from ..models import WDL
from . import bracket
from .predict import HOSTS, WDL_MODEL, MatchPredictor

_BLANK = _TeamState()


def _actual_result(row):
    hg, ag = row["home_score"], row["away_score"]
    return "H" if hg > ag else ("A" if ag > hg else "D")


def _row_features(states, meta, h, a, ko):
    hc, ac = meta[h]["canonical"], meta[a]["canonical"]
    return core_feature_row(states.get(hc, _BLANK), states.get(ac, _BLANK),
                            ko, int(hc in HOSTS), int(ac in HOSTS))


def _metrics(proba, y, ko_flags, adv_pred, adv_true):
    proba = np.array(proba)
    y = np.array(y)
    ko = np.array(ko_flags, dtype=bool)
    pred = np.array(WDL)[proba.argmax(1)]
    ko_hits = [p == t for p, t, k in zip(adv_pred, adv_true, ko_flags) if k]
    grp = ~ko
    return {
        "overall_acc": float((pred == y).mean()),
        "group_acc": float((pred[grp] == y[grp]).mean()) if grp.any() else float("nan"),
        "ko_adv_acc": float(np.mean(ko_hits)) if ko_hits else float("nan"),
        "brier": brier_multiclass(proba, y),
        "log_loss": log_loss_wdl(proba, y),
    }


def run(pred: MatchPredictor | None = None, b: dict | None = None):
    warnings.filterwarnings("ignore")
    pred = pred or MatchPredictor()
    b = b or bracket.derive()
    meta = b["meta"]
    model = pred.models[WDL_MODEL]

    fixtures = [(mid, row) for mid, row in b["by_id"].items()
                if row["status"] == "Completed"]
    fixtures.sort(key=lambda kv: (str(kv[1]["date"]), int(kv[0])))

    static_states = pred.states                       # frozen at entering-2026
    dyn_states = copy.deepcopy(pred.states)           # evolves through 2026

    acc = {"static": {"proba": [], "y": [], "ko": [], "advp": [], "advt": []},
           "dynamic": {"proba": [], "y": [], "ko": [], "advp": [], "advt": []}}

    for mid, row in fixtures:
        h, a = int(row["home_team_id"]), int(row["away_team_id"])
        ko = int(row["stage_id"] != bracket.STAGE_GROUP)
        y = _actual_result(row)
        adv_true = bracket.actual_advancer(row)

        for tag, states in (("static", static_states), ("dynamic", dyn_states)):
            X = pd.DataFrame([_row_features(states, meta, h, a, ko)])[CORE_FEATURES]
            p = model.predict_wdl(X)[0]
            advp = h if (p[0] + p[1] / 2) >= 0.5 else a
            d = acc[tag]
            d["proba"].append(p); d["y"].append(y); d["ko"].append(ko)
            d["advp"].append(advp); d["advt"].append(adv_true)

        # fold the REAL result into the dynamic state, then continue
        hc, ac = meta[h]["canonical"], meta[a]["canonical"]
        for nm in (hc, ac):
            dyn_states.setdefault(nm, _TeamState())
        update_after_match(dyn_states[hc], dyn_states[ac],
                           int(row["home_score"]), int(row["away_score"]),
                           ko, int(hc in HOSTS), int(ac in HOSTS), row["date"])

    static = _metrics(**{k: acc["static"][k] for k in ("proba", "y")},
                      ko_flags=acc["static"]["ko"],
                      adv_pred=acc["static"]["advp"], adv_true=acc["static"]["advt"])
    dynamic = _metrics(**{k: acc["dynamic"][k] for k in ("proba", "y")},
                       ko_flags=acc["dynamic"]["ko"],
                       adv_pred=acc["dynamic"]["advp"], adv_true=acc["dynamic"]["advt"])
    _report(len(fixtures), static, dynamic)
    return static, dynamic


def _fmt(name, s, d):
    def cell(x, pct):
        return f"{x:.1%}" if pct else f"{x:.4f}"
    return name, s, d


def _report(n, static, dynamic):
    rows = [
        ("Overall result accuracy", "overall_acc", True),
        ("Group result accuracy", "group_acc", True),
        ("Knockout advancer accuracy", "ko_adv_acc", True),
        ("Brier score (lower better)", "brier", False),
        ("Log loss (lower better)", "log_loss", False),
    ]
    print("\n" + "=" * 68)
    print(" DYNAMIC vs STATIC EVALUATION")
    print(f" each match predicted from real results of all earlier matches ({n} played)")
    print("=" * 68)
    print(f" {'Metric':<32}{'Static':>12}{'Dynamic':>12}{'Δ':>10}")
    print(" " + "-" * 66)
    lines = ["# Dynamic vs static evaluation", "",
             f"Each of the {n} completed 2026 matches predicted two ways: from a "
             "frozen pre-tournament snapshot (**static**) and from a state that "
             "absorbs the **real results of every earlier match** (**dynamic**).",
             "", "| Metric | Static | Dynamic | Δ |", "| --- | -: | -: | -: |"]
    for label, key, pct in rows:
        s, d = static[key], dynamic[key]
        delta = d - s
        fs = f"{s:.1%}" if pct else f"{s:.4f}"
        fd = f"{d:.1%}" if pct else f"{d:.4f}"
        fdlt = (f"{delta:+.1%}" if pct else f"{delta:+.4f}")
        print(f" {label:<32}{fs:>12}{fd:>12}{fdlt:>10}")
        lines.append(f"| {label} | {fs} | {fd} | {fdlt} |")
    print("=" * 68)
    better = "improves" if dynamic["brier"] < static["brier"] else "does not improve"
    note = (f"\nUpdating on real in-tournament results {better} calibration "
            f"(Brier {static['brier']:.3f} → {dynamic['brier']:.3f}).")
    print(note)
    lines += ["", note.strip()]
    paths.ensure_dirs()
    (paths.OUTPUT_DIR / "dynamic_vs_static_eval.md").write_text(
        "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
