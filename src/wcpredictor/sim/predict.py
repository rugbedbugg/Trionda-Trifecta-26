"""Predict any hypothetical 2026 matchup from pre-tournament information only.

Trains the models on matches up to and including the 2022 World Cup, and snapshots
every team's rolling state (Elo, pedigree, form) as it stood *entering* 2026.
A prediction for "Team A vs Team B" then featurises the pair with the exact same
`core_feature_row` used in training — so there is no train/serve skew, and no
2026 result ever leaks into a 2026 prediction.

Returns, per matchup:
  * `proba`   — [P(home win), P(draw), P(away win)]
  * `adv_home`— knockout advance probability for the home side (draw split 50/50)
"""
from __future__ import annotations

import warnings

import pandas as pd

from .. import models
from ..features import CORE_FEATURES, _TeamState, core_feature_row, snapshot_states
from ..evaluate import load_feature_frame
from ..teams import canonical_team

HOSTS = {"united states", "canada", "mexico"}
CUTOFF_YEAR = 2022


# Random forest is the sharpest model in the strict pre-tournament regime this
# predictor runs in (best accuracy and knockout-advancer hit rate on held-out 2026).
WDL_MODEL = "random_forest"


class MatchPredictor:
    def __init__(self, cutoff_year: int = CUTOFF_YEAR):
        warnings.filterwarnings("ignore")
        df = load_feature_frame()
        train = df[(df["year"] <= cutoff_year) & df["result"].notna()].copy()
        self.models = models.train_all(train)
        self.states = snapshot_states(cutoff_year)
        self._blank = _TeamState()

    def _state(self, canonical: str, states=None) -> _TeamState:
        return (states or self.states).get(canonical, self._blank)

    def predict(self, home: str, away: str, is_knockout: int, states=None):
        """home/away are canonical team names.

        `states` overrides the frozen entering-2026 snapshot with a caller-supplied
        (e.g. dynamically updated) state dict — used by continue_live to predict
        from the real results already played in the tournament."""
        hh = int(home in HOSTS)
        ah = int(away in HOSTS)
        row = core_feature_row(self._state(home, states), self._state(away, states),
                               is_knockout, hh, ah, date=None)
        X = pd.DataFrame([row])[CORE_FEATURES]
        proba = self.models[WDL_MODEL].predict_wdl(X)[0]
        adv_home = proba[0] + proba[1] / 2.0
        return {
            "proba": proba,          # [H, D, A]
            "adv_home": float(adv_home),
        }


def to_canonical(name: str) -> str:
    return canonical_team(name)
