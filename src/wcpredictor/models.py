"""Stage 5 — Models.

Every model exposes the same interface: ``predict_wdl(df) -> ndarray[n, 3]``
giving calibrated probabilities in the fixed column order **[H, D, A]** (home
win, draw, away win). That uniform interface is what lets evaluate.py score
wildly different models — a constant baseline, a Poisson goals model, a random
forest — with the same proper scoring rules.

Models, in the draft's build order:

  BaseRate      : constant training-set class frequencies (the floor).
  EloLogistic   : multinomial logistic on the single Elo-difference feature —
                  the "always back the higher-Elo team", made probabilistic.
  PoissonDC     : two Poisson goal regressions -> a score matrix with a
                  Dixon-Coles low-score correction -> W/D/L. This is how we get
                  honest *draw* probabilities instead of a 3-way classifier that
                  learns to never predict a draw.
  LogReg / RF   : multinomial classifiers on the full core feature set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import CORE_FEATURES

WDL = ["H", "D", "A"]
_MAX_GOALS = 8
_DC_RHO = 0.03  # Dixon-Coles low-score correction strength


def _X(df: pd.DataFrame) -> np.ndarray:
    return df[CORE_FEATURES].fillna(0.0).to_numpy(dtype=float)


def _y(df: pd.DataFrame) -> np.ndarray:
    return df["result"].to_numpy()


def _reorder(proba: np.ndarray, classes) -> np.ndarray:
    """Map a classifier's (possibly alphabetical) class order onto [H, D, A]."""
    idx = {c: i for i, c in enumerate(classes)}
    out = np.zeros((proba.shape[0], 3))
    for j, c in enumerate(WDL):
        if c in idx:
            out[:, j] = proba[:, idx[c]]
    return out


class BaseRate:
    name = "base_rate"

    def fit(self, df):
        counts = df["result"].value_counts(normalize=True)
        self.p = np.array([counts.get(c, 1e-6) for c in WDL])
        self.p = self.p / self.p.sum()
        return self

    def predict_wdl(self, df):
        return np.tile(self.p, (len(df), 1))


class EloLogistic:
    name = "elo"

    def fit(self, df):
        self.clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        )
        self.clf.fit(df[["elo_diff"]].to_numpy(dtype=float), _y(df))
        return self

    def predict_wdl(self, df):
        proba = self.clf.predict_proba(df[["elo_diff"]].to_numpy(dtype=float))
        return _reorder(proba, self.clf.classes_)


class LogRegWDL:
    name = "logreg"

    def fit(self, df):
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.3),  # regularised: tiny dataset
        )
        self.clf.fit(_X(df), _y(df))
        return self

    def predict_wdl(self, df):
        return _reorder(self.clf.predict_proba(_X(df)), self.clf.classes_)


class RandomForestWDL:
    name = "random_forest"

    def fit(self, df):
        self.clf = RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=15,
            random_state=0, n_jobs=-1,
        )
        self.clf.fit(_X(df), _y(df))
        return self

    def predict_wdl(self, df):
        return _reorder(self.clf.predict_proba(_X(df)), self.clf.classes_)


def _dc_tau(i, j, lh, la, rho):
    """Dixon-Coles adjustment to the independent-Poisson score probabilities."""
    if i == 0 and j == 0:
        return 1.0 - lh * la * rho
    if i == 0 and j == 1:
        return 1.0 + lh * rho
    if i == 1 and j == 0:
        return 1.0 + la * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


class PoissonDC:
    name = "poisson_dc"

    def fit(self, df):
        # Standardise inside the regressor for stable, fast convergence.
        X = _X(df)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self.home = PoissonRegressor(alpha=1.0, max_iter=3000).fit(Xs, df["_hg"].to_numpy())
        self.away = PoissonRegressor(alpha=1.0, max_iter=3000).fit(Xs, df["_ag"].to_numpy())
        return self

    def _lambdas(self, df):
        Xs = self._scaler.transform(_X(df))
        lh = np.clip(self.home.predict(Xs), 1e-3, 6.0)
        la = np.clip(self.away.predict(Xs), 1e-3, 6.0)
        return lh, la

    def _score_matrix(self, lh_k, la_k):
        gr = np.arange(_MAX_GOALS + 1)
        mat = np.outer(poisson.pmf(gr, lh_k), poisson.pmf(gr, la_k))
        for i in (0, 1):
            for j in (0, 1):
                mat[i, j] *= _dc_tau(i, j, lh_k, la_k, _DC_RHO)
        mat = np.clip(mat, 0, None)
        return mat / mat.sum()

    def predict_wdl(self, df):
        lh, la = self._lambdas(df)
        out = np.zeros((len(df), 3))
        for k in range(len(df)):
            mat = self._score_matrix(lh[k], la[k])
            out[k, 0] = np.tril(mat, -1).sum()   # home win: i > j
            out[k, 1] = np.trace(mat)            # draw:     i == j
            out[k, 2] = np.triu(mat, 1).sum()    # away win: i < j
        return out

    def predict_scoreline(self, df):
        """Most-likely exact scoreline per match, as (home_goals, away_goals).
        Gives the goals the group-stage tiebreakers (GD, GF) need."""
        lh, la = self._lambdas(df)
        scores = []
        for k in range(len(df)):
            mat = self._score_matrix(lh[k], la[k])
            i, j = np.unravel_index(mat.argmax(), mat.shape)
            scores.append((int(i), int(j)))
        return scores

    def predict_expected_goals(self, df):
        """Expected (lambda) goals per side — a smoother goal signal."""
        lh, la = self._lambdas(df)
        return list(zip(lh, la))


class Ensemble:
    """Average of the probabilistic models. Averaging independent, imperfect
    probability estimates is a cheap, robust regulariser — it pulls in the
    over-confident tails that log loss punishes, and tends to be the best
    *calibrated* model on tiny data."""

    name = "ensemble"

    def __init__(self, members: dict):
        # Base rate anchors the average toward the observed class frequencies.
        self.members = [members[n] for n in ("base_rate", "elo", "poisson_dc",
                                             "logreg", "random_forest")]

    def fit(self, df):  # members are already fitted
        return self

    def predict_wdl(self, df):
        return np.mean([m.predict_wdl(df) for m in self.members], axis=0)


def all_model_classes():
    return [BaseRate, EloLogistic, PoissonDC, LogRegWDL, RandomForestWDL]


def train_all(train_df: pd.DataFrame) -> dict:
    """Fit every model on the training frame. PoissonDC needs goal columns, so
    the caller must attach ``_hg``/``_ag`` (home/away goals) to train_df."""
    models = {}
    for cls in all_model_classes():
        models[cls.name] = cls().fit(train_df)
    models["ensemble"] = Ensemble(models).fit(train_df)
    return models
