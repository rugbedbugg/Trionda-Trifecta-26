"""Score prediction via a Poisson / Dixon-Coles model.

The right way to predict scores is NOT to guess an exact "2-1" — it's to model
each team's goal *count* as Poisson, which yields a full distribution over every
scoreline (and, consistently, the W/D/L probabilities). We predict each side's
expected goals (lambda) and build the score grid from them, with a Dixon-Coles
correction so low scores / draws are weighted realistically.

The lambda estimator is pluggable — this is where XGBoost / RandomForest belong:

  * ``glm``  — sklearn PoissonRegressor (a Poisson GLM). Robust on small data;
               the sensible default with only ~97 played 2026 matches.
  * ``gbm``  — sklearn HistGradientBoostingRegressor(loss="poisson"), i.e. a
               gradient-boosted Poisson model (same family as XGBoost's
               ``objective="count:poisson"``). Shines with more data/features.

Training uses the played 2026 matches, with symmetric per-attacking-team rows
(current Elo, squad value, host) so there is no home-side bias. Honest caveat:
exact-score accuracy tops out ~15-20% for *anyone* — the modal 1-1/1-0/2-1
dominate — so the real deliverable is a calibrated goal distribution, not a
point "2-1" claim. A larger international-match dataset (see README) is the main
lever to push it further.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import paths
from ..teams import canonical_team
from . import bracket

MAX_GOALS = 8
DC_RHO = 0.06          # Dixon-Coles low-score correction
BASE_FEATURES = ["own_elo", "opp_elo", "own_logval", "opp_logval", "host"]
XG_FEATURES = ["own_xgf", "opp_xga"]        # attacker's xG-for rate, defender's xG-against rate
INTL_FEATURES = ["own_intl_att", "opp_intl_def"]  # international attack / opp-defence rating
KO_FACTOR = 0.90            # knockouts score ~10% fewer goals (observed 2.68 vs 2.99)
HOSTS_2026 = {"united states", "canada", "mexico"}


def _intl_maps(id2c, adjusted, source):
    """Return (own_map, opp_map, lg_own, lg_opp) keyed by team_id, or None.

    `source` is a precomputed rates/strengths frame (or None to auto-load). With
    ``adjusted`` it uses the opponent-adjusted attack/defence columns; otherwise
    the raw goal-rate columns.
    """
    if source is None:
        from . import intl
        if not intl.available():
            return None
        source = intl.team_strengths() if adjusted else intl.team_goal_rates()
    if source is None or len(source) == 0:
        return None
    own_col, opp_col = ("attack", "defence") if adjusted else ("gf_rate", "ga_rate")
    rmap = source.set_index("team")
    own = {t: float(rmap[own_col].get(c, np.nan)) for t, c in id2c.items()}
    opp = {t: float(rmap[opp_col].get(c, np.nan)) for t, c in id2c.items()}
    return own, opp, float(source[own_col].mean()), float(source[opp_col].mean())


def _make_regressor(kind: str):
    if kind == "gbm":
        return HistGradientBoostingRegressor(
            loss="poisson", max_depth=3, max_iter=250,
            learning_rate=0.05, min_samples_leaf=20, random_state=0)
    # Features span very different scales (Elo ~2000 vs goal rates ~1.5), so
    # standardise before the Poisson GLM for stable convergence and fair weights.
    return make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=2000))


class PoissonScoreModel:
    def __init__(self, kind: str = "glm", use_xg: bool = False,
                 use_intl: bool = False, intl_adjusted: bool = True,
                 intl_source=None, ko_factor: float = 1.0):
        teams = pd.read_csv(paths.WC2026_DIR / "teams.csv")
        self.elo = dict(zip(teams["team_id"], teams["elo_rating"].astype(float)))
        squads = pd.read_csv(paths.WC2026_DIR / "squads_and_players.csv")
        val = squads.groupby("team_id")["market_value_eur"].sum()
        self.logval = {t: float(np.log1p(v)) for t, v in val.items()}
        self.hosts = {tid for tid, name in zip(teams["team_id"], teams["team_name"])
                      if canonical_team(name) in HOSTS_2026}
        self.kind = kind
        self.use_xg = use_xg
        self.ko_factor = ko_factor   # multiplies both lambdas for knockout matches
        self.xgf, self.xga, self.league_xg = {}, {}, 1.3

        # International attack/defence (Kaggle). Opponent-adjusted strengths by
        # default (intl_adjusted); raw goal rates otherwise. Auto-loads when the
        # file is present, silently off when not. `intl_source` lets a caller
        # inject a precomputed frame (fast backtests, tests).
        self.use_intl = use_intl
        self.intl_own, self.intl_opp = {}, {}
        if use_intl:
            id2c = {tid: canonical_team(n)
                    for tid, n in zip(teams["team_id"], teams["team_name"])}
            maps = _intl_maps(id2c, intl_adjusted, intl_source)
            if maps:
                self.intl_own, self.intl_opp, self.intl_lg_own, self.intl_lg_opp = maps
            else:
                self.use_intl = False   # graceful: no usable international data

        self.features = (BASE_FEATURES
                         + (XG_FEATURES if use_xg else [])
                         + (INTL_FEATURES if self.use_intl else []))
        self.reg = None

    def _lookup(self, m, tid, league):
        v = m.get(tid, np.nan)
        return league if v != v else v   # NaN-safe fallback to league average

    # --- feature row for "goals scored by `att` against `dfn`" ---
    def _row(self, att, dfn):
        r = {
            "own_elo": self.elo[att], "opp_elo": self.elo[dfn],
            "own_logval": self.logval.get(att, 0.0),
            "opp_logval": self.logval.get(dfn, 0.0),
            "host": int(att in self.hosts),
        }
        if self.use_xg:
            r["own_xgf"] = self.xgf.get(att, self.league_xg)   # how much xG att creates
            r["opp_xga"] = self.xga.get(dfn, self.league_xg)   # how much xG dfn concedes
        if self.use_intl:
            r["own_intl_att"] = self._lookup(self.intl_own, att, self.intl_lg_own)
            r["opp_intl_def"] = self._lookup(self.intl_opp, dfn, self.intl_lg_opp)
        return r

    def _build_xg_rates(self, samples):
        """Each team's mean xG created / conceded across the training matches — a
        leading indicator of scoring (prior-avg xG predicts future goals better
        than prior-avg goals). Falls back to actual goals where xG is missing."""
        for_, against = {}, {}
        allxg = []
        for s in samples:
            h, a, hg, ag = s[0], s[1], s[2], s[3]
            hxg = s[4] if len(s) > 4 and s[4] == s[4] else hg
            axg = s[5] if len(s) > 5 and s[5] == s[5] else ag
            for_.setdefault(h, []).append(hxg); against.setdefault(h, []).append(axg)
            for_.setdefault(a, []).append(axg); against.setdefault(a, []).append(hxg)
            allxg += [hxg, axg]
        self.xgf = {t: float(np.mean(v)) for t, v in for_.items()}
        self.xga = {t: float(np.mean(v)) for t, v in against.items()}
        self.league_xg = float(np.mean(allxg)) if allxg else 1.3

    def fit(self, samples, xg_weight: float = 0.0):
        """samples: (home_id, away_id, home_goals, away_goals[, home_xg, away_xg]).

        xG enters as *features* (each team's xG-for/against rate — the strong,
        leading-indicator use). The ``xg_weight`` target blend is kept for
        experiments but defaults off (goals target), since on this data the
        feature form is what helps.
        """
        samples = list(samples)
        if self.use_xg:
            self._build_xg_rates(samples)
        rows, y = [], []
        for s in samples:
            h, a, hg, ag = s[0], s[1], s[2], s[3]
            hxg = s[4] if len(s) > 4 and s[4] == s[4] else hg
            axg = s[5] if len(s) > 5 and s[5] == s[5] else ag
            rows.append(self._row(h, a)); y.append((1 - xg_weight) * hg + xg_weight * hxg)
            rows.append(self._row(a, h)); y.append((1 - xg_weight) * ag + xg_weight * axg)
        X = pd.DataFrame(rows)[self.features]
        self.reg = _make_regressor(self.kind).fit(X, y)
        return self

    def lambdas(self, home_id, away_id, is_knockout=0):
        X = pd.DataFrame([self._row(home_id, away_id),
                          self._row(away_id, home_id)])[self.features]
        lh, la = np.clip(self.reg.predict(X), 0.05, 6.0)
        f = self.ko_factor if is_knockout else 1.0   # knockout deflation
        return float(lh * f), float(la * f)

    def score_grid(self, home_id, away_id, is_knockout=0):
        lh, la = self.lambdas(home_id, away_id, is_knockout)
        gr = np.arange(MAX_GOALS + 1)
        mat = np.outer(poisson.pmf(gr, lh), poisson.pmf(gr, la))
        # Dixon-Coles adjustment to the four lowest scorelines.
        mat[0, 0] *= 1 - lh * la * DC_RHO
        mat[0, 1] *= 1 + lh * DC_RHO
        mat[1, 0] *= 1 + la * DC_RHO
        mat[1, 1] *= 1 - DC_RHO
        mat = np.clip(mat, 0, None)
        return mat / mat.sum(), lh, la

    def predict(self, home_id, away_id, is_knockout=0):
        mat, lh, la = self.score_grid(home_id, away_id, is_knockout)
        i, j = np.unravel_index(mat.argmax(), mat.shape)
        wdl = [np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()]
        # top correct scorelines
        flat = sorted(((mat[x, y], x, y) for x in range(MAX_GOALS + 1)
                       for y in range(MAX_GOALS + 1)), reverse=True)[:4]
        return {
            "exp_home": lh, "exp_away": la,
            "most_likely": (int(i), int(j)),
            "wdl": [float(x) for x in wdl],
            "top_scores": [((x, y), float(p)) for p, x, y in flat],
        }

    def markets(self, home_id, away_id, is_knockout=0):
        """Every derived market from the same score grid: total-goals
        distribution, over/under lines, both-teams-to-score, correct scores."""
        mat, lh, la = self.score_grid(home_id, away_id, is_knockout)
        g = mat.shape[0] - 1
        total = np.zeros(2 * g + 1)
        for x in range(g + 1):
            for y in range(g + 1):
                total[x + y] += mat[x, y]

        def over(line):
            return float(sum(total[k] for k in range(len(total)) if k > line))

        btts = float(sum(mat[x, y] for x in range(1, g + 1) for y in range(1, g + 1)))
        return {
            "exp_home": lh, "exp_away": la, "exp_total": lh + la,
            "total_dist": total,
            "over": {ln: over(ln) for ln in (0.5, 1.5, 2.5, 3.5)},
            "btts": btts,
            "grid": mat,
        }


def _sample(r):
    return (int(r["home_team_id"]), int(r["away_team_id"]),
            int(r["home_score"]), int(r["away_score"]),
            float(r["home_xg"]) if r.get("home_xg") == r.get("home_xg") else None,
            float(r["away_xg"]) if r.get("away_xg") == r.get("away_xg") else None,
            int(r["stage_id"] != bracket.STAGE_GROUP))   # is_knockout


def _completed(b):
    return [_sample(r) for r in b["by_id"].values() if r["status"] == "Completed"]


def backtest(kind: str = "glm", warmup: int = 40, use_xg: bool = False,
             use_intl: bool = False, intl_adjusted: bool = True, ko_factor: float = 1.0):
    """Walk-forward exact-score / within-1 / goal-MAE on played 2026 matches."""
    b = bracket.derive()
    comp = sorted(
        ((mid, r) for mid, r in b["by_id"].items() if r["status"] == "Completed"),
        key=lambda kv: (str(kv[1]["date"]), int(kv[0])))
    # Fit the international strengths/rates once — they don't depend on 2026 games.
    intl_source = None
    if use_intl:
        from . import intl
        if intl.available():
            intl_source = (intl.team_strengths() if intl_adjusted
                           else intl.team_goal_rates())
    exact = within1 = ou_hit = n = 0
    abs_err = []
    total_err = []
    for i, (mid, r) in enumerate(comp):
        if i < warmup:
            continue
        train = [_sample(x) for _, x in comp[:i]]
        model = PoissonScoreModel(kind, use_xg=use_xg, use_intl=use_intl,
                                  intl_adjusted=intl_adjusted, intl_source=intl_source,
                                  ko_factor=ko_factor).fit(train)
        h, a = int(r["home_team_id"]), int(r["away_team_id"])
        ko = int(r["stage_id"] != bracket.STAGE_GROUP)
        pi, pj = model.predict(h, a, ko)["most_likely"]
        mk = model.markets(h, a, ko)
        ah, aa = int(r["home_score"]), int(r["away_score"])
        n += 1
        exact += (pi == ah and pj == aa)
        within1 += (abs(pi - ah) <= 1 and abs(pj - aa) <= 1)
        abs_err += [abs(pi - ah), abs(pj - aa)]
        # Over/Under 2.5: did the model's side (P>0.5) match reality?
        actual_over = (ah + aa) > 2.5
        pred_over = mk["over"][2.5] > 0.5
        ou_hit += (pred_over == actual_over)
        total_err.append(abs(mk["exp_total"] - (ah + aa)))
    return {"n": n, "exact": exact / n, "within1": within1 / n,
            "goal_mae": float(np.mean(abs_err)), "ou25_acc": ou_hit / n,
            "total_mae": float(np.mean(total_err)), "kind": kind}


def run_backtest():
    print("\n" + "=" * 60)
    print(" SCORE-MODEL BACKTEST (walk-forward on played 2026 matches)")
    print("=" * 60)
    from . import intl
    configs = [("GLM", "glm", dict()), ("GBM", "gbm", dict())]
    if intl.available():
        configs += [
            ("GLM +intl raw", "glm", dict(use_intl=True, intl_adjusted=False)),
            ("GLM +intl adj", "glm", dict(use_intl=True, intl_adjusted=True)),
            ("  +KO factor", "glm", dict(use_intl=True, intl_adjusted=True,
                                          ko_factor=KO_FACTOR)),
        ]
    for label, kind, kw in configs:
        m = backtest(kind, **kw)
        print(f" {label:<14}| exact {m['exact']:.0%} | within-1 {m['within1']:.0%} "
              f"| O/U 2.5 {m['ou25_acc']:.0%} | total MAE {m['total_mae']:.2f}  "
              f"(n={m['n']})")
    print(" Coarser markets (over/under) score far higher than exact score.")
    if intl.available():
        print(" +intl adj = opponent-adjusted attack/defence (best over/under).")
        print(" +KO factor = fixed knockout goal deflation (a small, robust correction).")
    else:
        print(" (Add the Kaggle international dataset — see README — for the +intl rows.)")
    print("=" * 60)


# ---- single-matchup CLI ----------------------------------------------------
def _resolve(name, meta):
    from difflib import get_close_matches
    by = {m["canonical"]: t for t, m in meta.items()}
    c = canonical_team(name)
    if c in by:
        return by[c]
    hit = get_close_matches(c, list(by), n=1, cutoff=0.6)
    return by[hit[0]] if hit else None


def show_matchup(home_name, away_name, kind="glm", knockout=False):
    from . import intl
    b = bracket.derive()
    meta = b["meta"]
    h, a = _resolve(home_name, meta), _resolve(away_name, meta)
    if h is None or a is None:
        print(f"Could not resolve: {home_name if h is None else away_name!r}")
        return
    use_intl = intl.available()   # auto-use recent international form if present
    model = PoissonScoreModel(kind, use_intl=use_intl,
                              ko_factor=KO_FACTOR).fit(_completed(b))
    ko = int(knockout)
    r = model.predict(h, a, ko)
    m = model.markets(h, a, ko)
    hn, an = meta[h]["name"], meta[a]["name"]
    tag = "Poisson-{}{}{}".format(kind.upper(), " +intl" if use_intl else "",
                                  " knockout" if knockout else "")
    print(f"\n  {hn} vs {an}   ({tag} score model)")
    print(f"  expected goals : {hn} {r['exp_home']:.2f} — {r['exp_away']:.2f} {an}"
          f"   (total {m['exp_total']:.2f})")
    print(f"  W / D / L      : {r['wdl'][0]:.0%} / {r['wdl'][1]:.0%} / {r['wdl'][2]:.0%}")
    print(f"  most likely    : {r['most_likely'][0]}–{r['most_likely'][1]}")
    print("\n  Total goals (over/under):")
    for ln, p in m["over"].items():
        print(f"     over {ln}: {p:5.0%}     under {ln}: {1-p:5.0%}")
    print(f"  Both teams to score: {m['btts']:.0%}")
    print("\n  Correct-score probabilities:")
    grid = m["grid"]
    top = sorted(((grid[x, y], x, y) for x in range(4) for y in range(4)),
                 reverse=True)[:8]
    for p, x, y in top:
        print(f"     {x}–{y}   {p:5.1%}")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    kind = "gbm" if "--gbm" in argv else "glm"
    knockout = any(f in argv for f in ("--knockout", "-k"))
    argv = [x for x in argv if x not in ("--gbm", "--knockout", "-k")]
    if len(argv) == 2:
        show_matchup(argv[0], argv[1], kind, knockout=knockout)
    else:
        run_backtest()


if __name__ == "__main__":
    main()
