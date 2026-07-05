"""A 2026-specialist match model.

The cross-era model (evaluate.py) deliberately uses only features that exist for
every World Cup back to 1930, and its Elo is *stale* for 2026 — it last updated
at the 2022 World Cup and misses four years of football. For predicting 2026
specifically we can do much better with two genuinely pre-tournament signals the
2026 dataset ships:

  * the teams' **current Elo rating** (real, up-to-date strength), and
  * each squad's **total market value** (a strong proxy for player quality).

A plain logistic regression on the home-minus-away difference of those two hits
~69% accuracy under 5-fold cross-validation on the played 2026 matches — versus
~62% for the stale-Elo cross-era model. More features (FIFA rank, caps, host,
knockout flag) only dilute the signal on this small sample, so we keep it lean.

This is not a replacement for the honest cross-era temporal validation; it is the
right tool when the question is narrowly "predict 2026 matches as well as
possible from information available before kickoff."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sqlite3

from .. import paths
from ..models import WDL

FEATURES = ["elo_diff", "logval_diff", "host_diff"]
HOSTS_2026 = {"united states", "canada", "mexico"}

_FLIP = {"H": "A", "A": "H", "D": "D"}


def _symmetrize(samples):
    """Add each match a second time with the teams swapped and the result
    mirrored. A World Cup is played at neutral venues, so which team is listed
    "home" is arbitrary — training on both orderings removes any spurious
    home-side bias and makes predictions order-invariant (predict(A,B) becomes
    the exact mirror of predict(B,A))."""
    out = []
    for h, a, r in samples:
        out.append((h, a, r))
        out.append((a, h, _FLIP[r]))
    return out


class Specialist2026:
    """Pre-tournament ratings -> W/D/L probabilities for 2026 fixtures."""

    def __init__(self):
        from ..teams import canonical_team
        teams = pd.read_csv(paths.WC2026_DIR / "teams.csv")
        self.elo = dict(zip(teams["team_id"], teams["elo_rating"].astype(float)))
        squads = pd.read_csv(paths.WC2026_DIR / "squads_and_players.csv")
        self.value = squads.groupby("team_id")["market_value_eur"].sum().to_dict()
        # Genuine host advantage (USA / Canada / Mexico). Tied to *which team* is a
        # host, so it flips sign correctly under the home/away swap and survives
        # symmetrisation — unlike the spurious "listed-first" bias we removed.
        self.hosts = {tid for tid, name in zip(teams["team_id"], teams["team_name"])
                      if canonical_team(name) in HOSTS_2026}
        self.clf = None

    def _row(self, home_id: int, away_id: int) -> dict:
        return {
            "elo_diff": self.elo[home_id] - self.elo[away_id],
            "logval_diff": np.log1p(self.value.get(home_id, 0.0))
            - np.log1p(self.value.get(away_id, 0.0)),
            "host_diff": int(home_id in self.hosts) - int(away_id in self.hosts),
        }

    def fit(self, samples):
        """samples: iterable of (home_id, away_id, result in {'H','D','A'})."""
        samples = _symmetrize(list(samples))   # neutral venue -> no home bias
        X = pd.DataFrame([self._row(h, a) for h, a, _ in samples])[FEATURES]
        y = [r for _, _, r in samples]
        # Regularised logistic; robust on a few dozen-to-hundred rows.
        self.clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)
        ).fit(X, y)
        self._classes = list(self.clf.classes_)
        return self

    def predict_wdl(self, home_id: int, away_id: int) -> np.ndarray:
        X = pd.DataFrame([self._row(home_id, away_id)])[FEATURES]
        proba = self.clf.predict_proba(X)[0]
        idx = {c: i for i, c in enumerate(self._classes)}
        return np.array([proba[idx[c]] if c in idx else 0.0 for c in WDL])


# How hard squad market value pulls the 2026 strength rating, in [0, 1].
# 0 recovers the pure-Elo model exactly; the blend stays on the Elo scale so the
# history rate->outcome curve is untouched. Set a priori (not tuned on 2026).
VALUE_WEIGHT = 0.35


class HistoryRatingModel:
    """Uses BOTH eras: learns the rating-gap -> outcome relationship from every
    World Cup up to 2022, then applies it with the 2026 teams' *current* strength.

    This is the right tool for the from-scratch simulator, which has no 2026
    results yet to train a 2026-only specialist. History supplies ~900 matches of
    rating->outcome signal; the 2026 dataset supplies up-to-date ratings (fixing
    the stale home-grown Elo).

    The 2026 strength rating blends two genuinely pre-tournament signals: each
    team's current Elo and its **squad market value** (a strong player-quality
    proxy the 2026-only specialist showed carries real signal). Value is mapped
    onto the Elo scale by a *cross-sectional* regression of Elo on log squad value
    across the 48 teams — no match outcome touches that fit, so it is leakage-safe
    — then mixed in with weight ``value_weight``. Because the blend stays on the
    Elo scale, the history-trained curve needs no change, and ``value_weight=0``
    reproduces the pure-Elo model.
    """

    def __init__(self, value_weight: float = VALUE_WEIGHT):
        teams = pd.read_csv(paths.WC2026_DIR / "teams.csv")
        self.elo = dict(zip(teams["team_id"], teams["elo_rating"].astype(float)))
        self.rating = self._blend_value(value_weight)
        # Train the rating->outcome curve on all completed matches up to 2022.
        with sqlite3.connect(paths.UNIFIED_DB) as con:
            f = pd.read_sql(
                "SELECT home_elo, away_elo, year, result FROM features "
                "WHERE result IS NOT NULL", con)
        tr = f[f["year"] <= 2022].copy()
        tr["rate"] = (tr["home_elo"] - tr["away_elo"]) / 400.0
        # Symmetrise: neutral venues -> no home bias. Add each match mirrored
        # (rate -> -rate, result flipped) so predictions are order-invariant.
        mirror = tr.copy()
        mirror["rate"] = -mirror["rate"]
        mirror["result"] = mirror["result"].map(_FLIP)
        both = pd.concat([tr, mirror], ignore_index=True)
        self.clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)
        ).fit(both[["rate"]], both["result"])
        self._classes = list(self.clf.classes_)

    def _blend_value(self, w: float) -> dict:
        """Blend squad market value into each 2026 team's Elo, on the Elo scale."""
        if w <= 0:
            return dict(self.elo)
        squads = pd.read_csv(paths.WC2026_DIR / "squads_and_players.csv")
        value = squads.groupby("team_id")["market_value_eur"].sum()
        logval = np.log1p(value)
        # Cross-sectional (team-level, no match results) map value -> Elo scale.
        ids = [t for t in self.elo if t in logval.index]
        e = np.array([self.elo[t] for t in ids], dtype=float)
        v = logval.loc[ids].to_numpy(dtype=float)
        slope, intercept = np.polyfit(v, e, 1)         # elo ~ intercept + slope*logval
        mean_v = float(v.mean())
        rating = {}
        for t, elo in self.elo.items():
            lv = float(logval[t]) if t in logval.index else mean_v
            elo_hat = intercept + slope * lv           # value-implied rating (Elo scale)
            rating[t] = (1.0 - w) * elo + w * elo_hat
        return rating

    def _rate(self, home_id, away_id):
        return (self.rating.get(home_id, 1800.0) - self.rating.get(away_id, 1800.0)) / 400.0

    def predict_wdl(self, home_id: int, away_id: int) -> np.ndarray:
        proba = self.clf.predict_proba(pd.DataFrame({"rate": [self._rate(home_id, away_id)]}))[0]
        idx = {c: i for i, c in enumerate(self._classes)}
        return np.array([proba[idx[c]] if c in idx else 0.0 for c in WDL])
