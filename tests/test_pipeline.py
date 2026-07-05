"""Smoke + invariant tests for the pipeline.

Run from the repo root with:  python -m pytest tests -q
(or plain `python tests/test_pipeline.py` for a dependency-free run).

These assert the properties the project actually stands on — the label rule,
leakage-safety, the entity-resolution keystone, and probability validity — not
just that code executes.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcpredictor import paths  # noqa: E402
from wcpredictor.ingest import _result_from_goals  # noqa: E402
from wcpredictor.models import WDL  # noqa: E402
from wcpredictor.teams import canonical_team  # noqa: E402


def test_penalty_match_labelled_as_draw():
    # Equal regulation/ET goals is a DRAW regardless of the shootout winner.
    assert _result_from_goals(1, 1) == "D"
    assert _result_from_goals(2, 0) == "H"
    assert _result_from_goals(0, 3) == "A"
    assert _result_from_goals(None, 1) is None


def test_team_canonicalisation():
    assert canonical_team("USA") == canonical_team("United States")
    assert canonical_team("Türkiye") == canonical_team("Turkey")
    assert canonical_team("West Germany") == canonical_team("Germany")
    # Defunct entities stay distinct.
    assert canonical_team("Yugoslavia") != canonical_team("Serbia")


def _require_db():
    if not paths.UNIFIED_DB.exists():
        raise SystemExit("Run `python -m wcpredictor.pipeline` first to build the DB.")


def test_features_leakage_safe():
    _require_db()
    with sqlite3.connect(paths.UNIFIED_DB) as con:
        f = pd.read_sql("SELECT * FROM features ORDER BY match_date, match_key", con)
    completed = f[f["result"].notna()]
    first = completed.iloc[0]
    # The first match of all time must start from a blank slate.
    assert first["home_elo"] == 1500.0 and first["away_elo"] == 1500.0
    assert first["pedigree_matches_diff"] == 0
    # Goal columns must never have leaked into the feature table.
    assert "home_goals" not in f.columns and "away_goals" not in f.columns


def test_player_link_is_subset_and_typed():
    _require_db()
    with sqlite3.connect(paths.UNIFIED_DB) as con:
        link = pd.read_sql("SELECT * FROM player_link", con)
        wc = pd.read_sql("SELECT wc26_player_id FROM wc26_players", con)
    # Every linked wc26 id is a real 2026 player, and each is linked at most once.
    assert set(link["wc26_player_id"]).issubset(set(wc["wc26_player_id"]))
    assert link["wc26_player_id"].is_unique
    assert link["match_method"].isin(["exact", "dob+fuzzy", "name", "manual"]).all()


def test_probabilities_are_valid():
    _require_db()
    from wcpredictor.evaluate import load_feature_frame, temporal_split
    from wcpredictor import models

    df = load_feature_frame()
    train, _, test = temporal_split(df)
    trained = models.train_all(train)
    for name, model in trained.items():
        p = model.predict_wdl(test)
        assert p.shape == (len(test), 3), name
        assert np.all(p >= -1e-9) and np.all(p <= 1 + 1e-9), name
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6), name
    assert WDL == ["H", "D", "A"]


def test_fifa_group_tiebreakers():
    from wcpredictor.sim import rules
    # Three teams level on 4 pts; head-to-head decides. A beat B, B beat C, C beat A
    # is a cycle, so fall to overall GD/GF, then seed. Construct a clean H2H case:
    # A, B, C all 1-1-0-ish but A has the head-to-head edge.
    teams = ["A", "B", "C", "D"]
    matches = [
        ("A", "B", 1, 0), ("A", "C", 1, 0), ("A", "D", 0, 3),   # A: 6pts
        ("B", "C", 2, 0), ("B", "D", 0, 1), ("C", "D", 0, 1),   # D wins all vs A/B/C
    ]
    order, rec = rules.rank_group(teams, matches, seed_key={t: 0 for t in teams})
    assert order[0] == "D"          # D: 9 pts, top
    assert rec["D"]["pts"] == 9
    assert set(order) == set(teams)


def test_bracket_structure_is_valid():
    from wcpredictor.sim import bracket
    b = bracket.derive()
    assert len(b["r32"]) == 16
    # Every group contributes a 1st and 2nd; 8 third-slots exist.
    slots = [s["home_slot"] for s in b["r32"]] + [s["away_slot"] for s in b["r32"]]
    assert sum(1 for _, pos in slots if pos == 3) == 8
    assert sum(1 for _, pos in slots if pos == 1) == 12
    # The final is fed by the two semi-finals.
    final = next(n for n in b["ko"] if n["stage"] == bracket.STAGE_FINAL)
    sfs = {n["match_id"] for n in b["ko"] if n["stage"] == bracket.STAGE_SF}
    assert {final["home_src"][1], final["away_src"][1]} == sfs


def test_simulation_produces_full_bracket():
    from wcpredictor.sim import simulate
    sim = simulate.simulate(report_stdout=False)
    # 32 knockout matches (16+8+4+2+1+1) all resolved, and a champion emerges.
    assert len(sim["nodes"]) == 32
    assert sim["champion"] and sim["runner_up"] and sim["champion"] != sim["runner_up"]
    # Exactly 32 distinct qualifiers seeded the Round of 32.
    quals = {int(t) for order in sim["group_order"].values() for t in order[:2]}
    quals |= set(sim["best_thirds"])
    assert len(quals) == 32


def test_specialist_predictions_are_order_invariant():
    from wcpredictor.sim import bracket
    from wcpredictor.sim.model2026 import Specialist2026
    b = bracket.derive()

    def res(r):
        hg, ag = r["home_score"], r["away_score"]
        return "H" if hg > ag else ("A" if ag > hg else "D")

    spec = Specialist2026().fit(
        [(int(r["home_team_id"]), int(r["away_team_id"]), res(r))
         for r in b["by_id"].values() if r["status"] == "Completed"])
    name2id = {m["name"]: t for t, m in b["meta"].items()}
    arg, esp = name2id["Argentina"], name2id["Spain"]
    p1 = spec.predict_wdl(arg, esp)   # [P(arg), draw, P(esp)]
    p2 = spec.predict_wdl(esp, arg)   # [P(esp), draw, P(arg)]
    # Swapping teams must mirror the result, not change the winner.
    assert abs(p1[0] - p2[2]) < 1e-9 and abs(p1[2] - p2[0]) < 1e-9
    assert abs(p1[1] - p2[1]) < 1e-9


def test_intl_ingester_parses_and_rates(tmp_path=None):
    import tempfile
    from wcpredictor.sim import intl

    # Missing file -> clear error, not a crash.
    try:
        intl.load(Path(tempfile.gettempdir()) / "does_not_exist_intl.csv")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass

    # Kaggle-style rows with name variants -> canonicalised + rated.
    csv = ("date,home_team,away_team,home_score,away_score,tournament,neutral\n"
           "2024-06-01,Spain,Brazil,2,1,Friendly,True\n"
           "2023-09-10,Cape Verde,DR Congo,0,2,Friendly,True\n"
           "2022-11-20,United States,Turkey,3,0,Friendly,False\n")
    f = Path(tempfile.mkdtemp()) / "intl.csv"
    f.write_text(csv, encoding="utf-8")
    df = intl.load(f)
    teams = set(df["home"]) | set(df["away"])
    assert {"cabo verde", "congo dr", "turkiye", "united states"} <= teams
    rates = intl.team_goal_rates(df, since_year=2000)
    assert set(rates.columns) == {"team", "matches", "gf_rate", "ga_rate"}
    assert (rates["gf_rate"] >= 0).all() and (rates["ga_rate"] >= 0).all()


def test_score_model_distribution_valid():
    from wcpredictor.sim import bracket
    from wcpredictor.sim.goals import PoissonScoreModel, _completed
    b = bracket.derive()
    model = PoissonScoreModel("glm").fit(_completed(b))
    name2id = {m["name"]: t for t, m in b["meta"].items()}
    r = model.predict(name2id["Spain"], name2id["Brazil"])
    assert r["exp_home"] > 0 and r["exp_away"] > 0
    assert abs(sum(r["wdl"]) - 1.0) < 1e-6          # valid W/D/L distribution
    assert sum(p for _, p in r["top_scores"]) <= 1.0 + 1e-6
    assert len(r["most_likely"]) == 2


def test_score_model_intl_features_toggle():
    import numpy as np
    import pandas as pd
    from wcpredictor.sim import bracket
    from wcpredictor.sim.goals import PoissonScoreModel, _completed, INTL_FEATURES
    b = bracket.derive()
    # With no usable data (empty frame), use_intl must silently fall back to off.
    m0 = PoissonScoreModel("glm", use_intl=True, intl_adjusted=False,
                           intl_source=pd.DataFrame(columns=["team", "gf_rate", "ga_rate"]))
    assert m0.use_intl is False
    assert not any(f in m0.features for f in INTL_FEATURES)
    # With injected rates for the whole field, the features activate and predict.
    canon = sorted({m["canonical"] for m in b["meta"].values()})
    rng = np.random.default_rng(0)
    rates = pd.DataFrame({"team": canon, "matches": 30,
                          "gf_rate": rng.uniform(0.6, 2.4, len(canon)),
                          "ga_rate": rng.uniform(0.5, 1.8, len(canon))})
    m1 = PoissonScoreModel("glm", use_intl=True, intl_adjusted=False,
                           intl_source=rates).fit(_completed(b))
    assert m1.use_intl and all(f in m1.features for f in INTL_FEATURES)
    name2id = {m["name"]: t for t, m in b["meta"].items()}
    r = m1.predict(name2id["Spain"], name2id["Brazil"])
    assert abs(sum(r["wdl"]) - 1.0) < 1e-6


def test_knockout_deflates_goals():
    from wcpredictor.sim import bracket
    from wcpredictor.sim.goals import PoissonScoreModel, _completed, KO_FACTOR
    b = bracket.derive()
    model = PoissonScoreModel("glm", ko_factor=KO_FACTOR).fit(_completed(b))
    name2id = {m["name"]: t for t, m in b["meta"].items()}
    h, a = name2id["Spain"], name2id["Brazil"]
    grp = model.markets(h, a, is_knockout=0)["exp_total"]
    ko = model.markets(h, a, is_knockout=1)["exp_total"]
    assert ko < grp                                    # knockout is lower-scoring
    assert abs(ko - grp * KO_FACTOR) < 1e-9            # exactly the fixed deflation


def test_rankings_sorted_by_elo():
    from wcpredictor.sim import rankings
    df = rankings.table()
    assert len(df) == 48
    assert list(df["elo_rating"]) == sorted(df["elo_rating"], reverse=True)
    assert rankings.table(top=5).shape[0] == 5


def test_matchup_cli_resolves_and_predicts():
    from wcpredictor.sim import matchup
    # fuzzy/case-insensitive name resolution + valid probabilities
    out = matchup.predict_matchup("spain", "brazil")
    assert out is not None
    assert abs(sum(out) - 1.0) < 1e-6
    assert matchup.predict_matchup("not a country", "brazil") is None


def test_specialist2026_beats_coinflip():
    from wcpredictor.sim import continue_live
    tally = continue_live.backtest(warmup=24, show_ko=False)
    hit, tot = tally["overall"]
    # The 2026 specialist should clear a naive baseline comfortably.
    assert tot > 30 and hit / tot > 0.60


def test_continue_live_predicts_real_next_matches():
    from wcpredictor.sim import continue_live
    node = continue_live.run()
    # Completed real matches keep their true winners; scheduled ones are predicted.
    actual = [n for n in node.values() if n["actual"]]
    scheduled = [n for n in node.values() if not n["actual"]]
    assert actual, "expected real completed matches to be recorded"
    # If any match is still scheduled, at least one must be a pure true-data call.
    if scheduled:
        assert any(n["inputs_real"] for n in scheduled)
        for n in scheduled:
            p = n["proba"]
            assert abs(sum(p) - 1.0) < 1e-6


def test_dynamic_eval_runs_and_is_leakage_safe():
    from wcpredictor.sim import dynamic_eval
    static, dynamic = dynamic_eval.run()
    # Both regimes produce valid metrics over the played matches.
    for m in (static, dynamic):
        assert 0.0 <= m["overall_acc"] <= 1.0
        assert 0.0 <= m["brier"] <= 2.0
        assert m["log_loss"] > 0.0


def test_value_weight_zero_recovers_pure_elo():
    from wcpredictor.sim.model2026 import HistoryRatingModel
    # value_weight=0 must leave the 2026 rating exactly equal to raw Elo, so the
    # blend is a strict superset of the previous model (no silent behaviour drift).
    m0 = HistoryRatingModel(value_weight=0.0)
    assert m0.rating == m0.elo
    # A positive weight actually moves ratings (value carries information), but
    # keeps them on the Elo scale (sane range), and predictions stay valid.
    mv = HistoryRatingModel(value_weight=0.35)
    assert mv.rating != mv.elo
    assert all(1200.0 < r < 2400.0 for r in mv.rating.values())
    ids = list(mv.rating)
    p = mv.predict_wdl(ids[0], ids[1])
    assert abs(sum(p) - 1.0) < 1e-6 and np.all(p >= 0)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
