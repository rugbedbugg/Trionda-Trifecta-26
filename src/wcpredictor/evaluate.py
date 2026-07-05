"""Stage 6 — Evaluation.

The whole project is organised to answer one honest question: does the model
produce *calibrated* probabilities on tournaments it never trained on? So we
score with proper scoring rules (log loss, Brier) — not raw accuracy — on a
strictly temporal split:

    train  : World Cups <= 2018
    valid  : World Cup 2022
    test   : World Cup 2026 (matches already played)

We also run the two sanity checks the draft insists on:

  * calibration / reliability — when a model says 60%, does it happen ~60%?
  * draw recall — is the classifier actually willing to predict draws, or has
    it collapsed to the classic "never predict a draw" failure mode?

Bookmaker implied probabilities are the real bar to beat. Odds aren't bundled
offline, so this module scores against the strongest available public baseline
(Elo) and leaves a documented hook to drop in a bookmaker-odds CSV.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import models, paths
from .models import WDL


def load_feature_frame() -> pd.DataFrame:
    """Features joined with goal targets, completed men's matches only kept for
    training/scoring (scheduled 2026 rows retained for the live stage)."""
    with sqlite3.connect(paths.UNIFIED_DB) as con:
        feat = pd.read_sql("SELECT * FROM features", con)
        goals = pd.read_sql(
            "SELECT match_key, home_goals, away_goals FROM matches", con
        )
    df = feat.merge(goals, on="match_key", how="left")
    df["_hg"] = df["home_goals"]
    df["_ag"] = df["away_goals"]
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def temporal_split(df: pd.DataFrame):
    completed = df[df["result"].notna()].copy()
    train = completed[completed["year"] <= 2018]
    valid = completed[completed["year"] == 2022]
    test = completed[completed["year"] == 2026]
    return train, valid, test


def _onehot(y: np.ndarray) -> np.ndarray:
    idx = {c: i for i, c in enumerate(WDL)}
    out = np.zeros((len(y), 3))
    for k, v in enumerate(y):
        out[k, idx[v]] = 1.0
    return out


def log_loss_wdl(proba: np.ndarray, y: np.ndarray) -> float:
    """Multiclass log loss with columns fixed to [H, D, A]. Computed directly so
    the class ordering is unambiguous (sklearn assumes lexicographic order)."""
    p_true = np.sum(proba * _onehot(y), axis=1)
    p_true = np.clip(p_true, 1e-15, 1.0)
    return float(-np.mean(np.log(p_true)))


def brier_multiclass(proba: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sum((proba - _onehot(y)) ** 2, axis=1)))


def accuracy(proba: np.ndarray, y: np.ndarray) -> float:
    pred = np.array(WDL)[proba.argmax(axis=1)]
    return float((pred == y).mean())


def draw_recall(proba: np.ndarray, y: np.ndarray) -> float:
    """Of the matches that were actually draws, how many did the model call a
    draw? Guards against the collapse-to-never-draw failure mode."""
    pred = np.array(WDL)[proba.argmax(axis=1)]
    mask = y == "D"
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == "D").mean())


def score_model(proba: np.ndarray, y: np.ndarray) -> dict:
    return {
        "log_loss": log_loss_wdl(proba, y),
        "brier": brier_multiclass(proba, y),
        "accuracy": accuracy(proba, y),
        # draw_recall via argmax is ~0 for any calibrated model (a draw is
        # rarely the single most likely result). mean_p_draw is the honest
        # check: a model that "ignores draws" assigns them << the base rate.
        "draw_recall": draw_recall(proba, y),
        "mean_p_draw": float(proba[:, 1].mean()),
    }


def naive_accuracy(df: pd.DataFrame) -> dict:
    """Point-prediction reference baselines (accuracy only)."""
    y = df["result"].to_numpy()
    always_home = (y == "H").mean()
    higher_elo = np.where(df["elo_diff"] >= 0, "H", "A")
    return {
        "always_home_acc": float(always_home),
        "higher_elo_acc": float((higher_elo == y).mean()),
        "actual_draw_rate": float((y == "D").mean()),
    }


def reliability_table(proba: np.ndarray, y: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Reliability of the predicted home-win probability across bins."""
    p_home = proba[:, 0]
    outcome = (y == "H").astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_home >= lo) & (p_home < hi if hi < 1 else p_home <= hi)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(m.sum()),
            "mean_pred": round(float(p_home[m].mean()), 3),
            "observed": round(float(outcome[m].mean()), 3),
        })
    return pd.DataFrame(rows)


def run() -> None:
    paths.ensure_dirs()
    df = load_feature_frame()
    train, valid, test = temporal_split(df)

    trained = models.train_all(train)

    records = []
    for split_name, split in (("valid_2022", valid), ("test_2026", test)):
        if len(split) == 0:
            continue
        y = split["result"].to_numpy()
        for name, model in trained.items():
            proba = model.predict_wdl(split)
            rec = {"split": split_name, "model": name, "n": len(split)}
            rec.update(score_model(proba, y))
            records.append(rec)

    results = pd.DataFrame(records)
    results.to_csv(paths.OUTPUT_DIR / "metrics.csv", index=False)

    print("\n=== Temporal validation (train <=2018 | valid 2022 | test 2026) ===")
    print(f"train n={len(train)}  valid n={len(valid)}  test n={len(test)}\n")
    for split_name in ("valid_2022", "test_2026"):
        sub = results[results["split"] == split_name]
        if sub.empty:
            continue
        ref = naive_accuracy(valid if split_name == "valid_2022" else test)
        print(f"--- {split_name} ---  "
              f"(always-home acc={ref['always_home_acc']:.3f}, "
              f"higher-Elo acc={ref['higher_elo_acc']:.3f}, "
              f"actual draw rate={ref['actual_draw_rate']:.3f})")
        show = sub[["model", "log_loss", "brier", "accuracy",
                    "draw_recall", "mean_p_draw"]].copy()
        show = show.sort_values("log_loss").round(4)
        print(show.to_string(index=False))
        print()

    # Calibration + draw sanity for the best model on the test set.
    if len(test):
        best_name = (
            results[results["split"] == "test_2026"]
            .sort_values("log_loss")["model"].iloc[0]
        )
        proba = trained[best_name].predict_wdl(test)
        rel = reliability_table(proba, test["result"].to_numpy())
        rel.to_csv(paths.OUTPUT_DIR / "reliability_test.csv", index=False)
        print(f"--- calibration (home-win prob), best test model = {best_name} ---")
        print(rel.to_string(index=False))
        print(f"\n[evaluate] metrics -> {paths.OUTPUT_DIR / 'metrics.csv'}")

    return results, trained


if __name__ == "__main__":
    run()
