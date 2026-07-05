"""Stage 7 — Live 2026 predictions.

Train on *every* completed match up to now (all history + the 2026 matches
already played) and predict the tournament's remaining, unplayed fixtures.

For a knockout tie a "draw" means level after extra time — i.e. the match goes
to penalties — so we report the three-way probability and, for knockouts, also
the probability each side advances (splitting the draw mass evenly, the
long-run rate of a coin-flip shootout).

As the real results land in a refreshed 2026 feed, ``score_live()`` compares the
stored predictions against actual outcomes and reports a running Brier score —
the project's final, honest number on matches it never trained on.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import models, paths
from .evaluate import brier_multiclass, load_feature_frame, log_loss_wdl
from .models import WDL

PRIMARY_MODEL = "ensemble"  # robust choice for out-of-sample knockout calls


def _train_on_all_completed():
    df = load_feature_frame()
    train = df[df["result"].notna()].copy()
    return models.train_all(train), df


def predict() -> pd.DataFrame:
    trained, df = _train_on_all_completed()
    scheduled = df[df["status"] == "Scheduled"].copy()

    # Later knockout rounds have "TBD" participants that depend on unplayed
    # results — we can only honestly predict fixtures whose teams are known.
    known = ~scheduled["home_team"].isin(["nan", "", None]) & \
            ~scheduled["away_team"].isin(["nan", "", None])
    n_tbd = (~known).sum()
    scheduled = scheduled[known].copy()
    if n_tbd:
        print(f"[predict_live] skipping {n_tbd} fixture(s) with undetermined "
              f"(TBD) participants — they depend on matches not yet played.")
    if scheduled.empty:
        print("[predict_live] no scheduled matches with known teams remain.")
        return scheduled

    proba = trained[PRIMARY_MODEL].predict_wdl(scheduled)
    out = scheduled[["match_key", "match_date", "stage_name",
                     "home_team", "away_team", "is_knockout"]].copy()
    out["p_home"] = proba[:, 0].round(3)
    out["p_draw"] = proba[:, 1].round(3)
    out["p_away"] = proba[:, 2].round(3)
    out["predicted"] = np.array(WDL)[proba.argmax(1)]

    # Knockout advance probability: split the draw (shootout) mass 50/50.
    adv_home = out["p_home"] + out["p_draw"] / 2
    out["advance"] = np.where(
        out["is_knockout"] == 1,
        np.where(adv_home >= 0.5, out["home_team"], out["away_team"]),
        "-",
    )
    out["p_advance"] = np.where(
        out["is_knockout"] == 1, np.maximum(adv_home, 1 - adv_home).round(3), np.nan
    )

    with sqlite3.connect(paths.UNIFIED_DB) as con:
        out.to_sql("live_predictions", con, if_exists="replace", index=False)
    out.to_csv(paths.OUTPUT_DIR / "live_predictions.csv", index=False)
    _write_markdown(out)
    return out


def _write_markdown(out: pd.DataFrame) -> None:
    lines = [
        "# 2026 World Cup — live knockout predictions",
        "",
        f"Model: **{PRIMARY_MODEL}**, trained on all completed matches "
        f"(1930–2026). Probabilities are pre-kickoff and leakage-safe.",
        "",
        "| Date | Stage | Match | P(home) | P(draw) | P(away) | Advances |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in out.iterrows():
        date = pd.to_datetime(r["match_date"]).date()
        match = f"{r['home_team'].title()} vs {r['away_team'].title()}"
        adv = (f"{r['advance'].title()} ({r['p_advance']:.0%})"
               if r["advance"] != "-" else "—")
        lines.append(
            f"| {date} | {r['stage_name']} | {match} | "
            f"{r['p_home']:.2f} | {r['p_draw']:.2f} | {r['p_away']:.2f} | {adv} |"
        )
    (paths.OUTPUT_DIR / "live_predictions.md").write_text("\n".join(lines), encoding="utf-8")


def score_live() -> None:
    """Score stored predictions against results that have since landed."""
    df = load_feature_frame()
    with sqlite3.connect(paths.UNIFIED_DB) as con:
        try:
            preds = pd.read_sql("SELECT * FROM live_predictions", con)
        except Exception:
            print("[predict_live] no stored predictions to score yet.")
            return
    landed = df[df["result"].notna()][["match_key", "result"]]
    merged = preds.merge(landed, on="match_key", how="inner")
    if merged.empty:
        print("[predict_live] predicted matches have not been played yet — "
              "re-run after refreshing the 2026 feed to get a live Brier score.")
        return
    proba = merged[["p_home", "p_draw", "p_away"]].to_numpy()
    y = merged["result"].to_numpy()
    print(f"[predict_live] scored {len(merged)} now-played predictions | "
          f"Brier={brier_multiclass(proba, y):.4f} "
          f"log_loss={log_loss_wdl(proba, y):.4f}")


def run() -> None:
    paths.ensure_dirs()
    out = predict()
    if not out.empty:
        print(f"\n[predict_live] {len(out)} upcoming matches predicted "
              f"(model={PRIMARY_MODEL}) -> outputs/live_predictions.md")
        show = out.copy()
        show["match"] = show["home_team"].str.title() + " vs " + show["away_team"].str.title()
        print(show[["stage_name", "match", "p_home", "p_draw", "p_away",
                    "advance", "p_advance"]].to_string(index=False))
    score_live()


if __name__ == "__main__":
    run()
