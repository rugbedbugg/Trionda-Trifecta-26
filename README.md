# Trionda-Trifecta-26: FIFA World Cup match & tournament predictor

A command-line toolkit that predicts FIFA Men's World Cup outcomes, single
matches (win / draw / loss), scorelines and the full 2026 bracket to a champion. 
Built on a hand-joined relational dataset spanning **1930–2026**.

Everything runs offline from local data; no service calls.

---

## Quick start

```bash
pip install -r requirements.txt        # pandas, numpy, scikit-learn (soccerdata optional)
cd src

python -m wcpredictor                  # help menu — lists every command
python -m wcpredictor.pipeline         # build the store + train + evaluate + live preds
python -m wcpredictor.sim              # simulate the whole 2026 tournament, then validate
python -m wcpredictor.sim.matchup "Spain" "France"       # predict any single matchup
python -m wcpredictor.sim.goals "Spain" "Brazil"         # scoreline + over/under + markets
```

`python -m wcpredictor` prints the full command menu at any time. Run any stage or
tool on its own (all are `python -m wcpredictor.<name>`). Tests:
`python tests/test_pipeline.py` (19 tests).

---

## Command surface

Every command is `python -m wcpredictor.<name>`. The last column names the model it
runs on.

| Command | What it does | Model it uses |
| --- | --- | --- |
| `analyst [--retrain]` | One-command prose briefing: state of play, model track record, remaining fixtures + scores, and what changed since last run | orchestrates the tools below |
| `pipeline` | Full core pipeline: ingest → resolve → enrich → features → evaluate → live predictions | all cross-era models |
| `evaluate` | Score every model on tournaments it never trained on (log loss, Brier, accuracy) | all cross-era models |
| `predict_live` | Predict the next scheduled fixtures with known teams | **ensemble** |
| `sim` | Simulate the entire tournament from scratch, then validate vs reality | history-rating model |
| `sim.simulate` / `sim.validate` | Just the prediction / just the scoring | history-rating model |
| `sim.continue_live [--scores]` | Predict the **real** remaining fixtures from all results so far (`--scores` adds a projected scoreline for each) | **2026 specialist** (+ Poisson score model for `--scores`) |
| `sim.dynamic_eval` | Walk-forward: predict each match from earlier real results (static vs dynamic) | 2026 specialist |
| `sim.matchup "A" "B" [--knockout]` | **W/D/L** for any two teams (order-invariant) | 2026 specialist |
| `sim.goals "A" "B" [--knockout] [--gbm]` | **Scoreline**: expected goals, over/under, BTTS, correct scores | Poisson score model |
| `sim.goals` (no args) | Score-model backtest (exact / within-1 / over-under / MAE) | Poisson score model |
| `sim.rankings [N]` | Teams ranked by current Elo (top N or all) | — (lookup) |
| `sim.market` | Model W/D/L next to prediction-market odds | 2026 specialist |
| `sim.intl` | Ingest the Kaggle international-results dataset (optional upgrade) | — (data prep) |

When new results land for: `git pull` (in the dataset repo) →
`python -m wcpredictor.pipeline` (retrain) → `python -m wcpredictor.sim.continue_live --scores`
(W/D/L **and** a projected scoreline for every remaining fixture, in one command).

---

## How it works

### Data sources

| Source | Role | Coverage | In repo |
| --- | --- | --- | --- |
| **Fjelstul World Cup DB** (`../worldcup/`) | Historical backbone + labels | Men's WC 1930–2022 | external (CC0) |
| **FIFA-World-Cup-2026-Dataset** (`../FIFA-World-Cup-2026-Dataset/`) | Live 2026 feed: squads, results, xG, current Elo, market value | 2026 | external (CC0) |
| **Kaggle international results** (`data/international_results.csv`) | ~49k internationals for goal-rate strengths | 1872–2026 | optional download |

The two World Cup sources are separate upstream repos placed alongside `predictor/`;
the Kaggle set is an optional drop-in (see *Setup*). `teams.py` canonicalises every
nation across all three ("USA"/"United States", "Türkiye"/"Turkey", "Cape Verde"/
"Cabo Verde", West Germany folded into Germany) so they join cleanly.

### The core pipeline (`src/wcpredictor/`)

| Stage | Module | What it does |
| --- | --- | --- |
| 1. Ingest | `ingest.py` | Fuse both WC sources into one cross-era `matches` fact table with **one W/D/L label rule** (from goals, so a penalty-shootout match is correctly a *draw*). |
| 2. Resolve | `resolve.py` | Link 2026 players to their Fjelstul history via normalised name + DOB with a fuzzy fallback; logs every miss; manual-override table. |
| 3. Enrich | `enrich.py` | Per-player ability signal. Optional FBref scrape (`soccerdata`) with **graceful fallback** to squad market value / caps / goals. |
| 4. Features | `features.py` | Rolling Elo, pedigree, recent form, match context — every column computed **strictly pre-kickoff**, with `_assert_leakage_safe()`. |
| 5. Models | `models.py` | Baselines, Poisson/Dixon-Coles, logistic regression, random forest, ensemble — one `predict_wdl → [H,D,A]` interface. |
| 6. Evaluate | `evaluate.py` | Temporal split (train ≤2018 · valid 2022 · test 2026); log loss, Brier, accuracy, calibration, draw sanity check. |
| 7. Live | `predict_live.py` | Predict remaining knockout fixtures; running Brier as results land. |

### The models behind the predictions

| Model | What it is, in one line | Predicts | Its job |
| --- | --- | --- | --- |
| **Base rate** | Always guesses the same long-run average split (~55% / 22% / 23% win / draw / loss), ignoring who is playing | W/D/L | The no-skill **floor** every other model has to beat. |
| **Elo** | Backs the higher-rated team, expressed as a probability | W/D/L | Simple **strength-only baseline**; its rating also feeds the models below. |
| **Logistic regression** | Weighs all the pre-match signals (Elo, form, history, rest) into one probability | W/D/L | The **main, interpretable** W/D/L predictor. |
| **Random forest** | Same inputs, but decision-tree "voting"; catches non-straight-line patterns | W/D/L | **Competitor** W/D/L model - sharpest on the honest no-leak test. |
| **Poisson + Dixon-Coles** | Predicts each side's goal count, then reads off the result; the Dixon-Coles tweak fixes draw/low-score odds | **Scoreline** → W/D/L | Where **honest draw probabilities** and a first scoreline come from. |
| **Ensemble** | The **average** of the models above | W/D/L | One steadier number; averaging cancels any single model's over-confidence. Used by `predict_live`. |
| **2026 specialist** | Logistic on each team's *current* Elo + total squad **market value**, trained on the real 2026 results so far | W/D/L | The **sharpest live** predictor once the tournament is underway (`matchup`, `continue_live`). |
| **History-rating model** | Learns "how a rating gap becomes a result" from every World Cup since 1930, applied to 2026 strength (Elo **blended with squad value**) | W/D/L | Drives the **from-scratch tournament sim** — the only model that works before any 2026 game is played. |
| **Poisson score model** (GLM / GBM) | A goals model tuned for 2026, optionally opponent-adjusted using 49k internationals | **Scoreline** + every betting market | Expected goals, most-likely score, over/under, both-teams-to-score, correct-score odds (`goals`). |

Two things that are **not** predictors but power the ones that are: the **Elo rating**
(the strength number most models read), and the **official FIFA tiebreaker rules**
(`sim/rules.py`) — deterministic group-standings logic, no machine learning.

> **Why several models instead of one?** With only ~1,000 World Cup matches ever
> played, a big fancy model just memorises noise. Simple models, averaged and scored
> on tournaments they never saw, are both more trustworthy and easier to explain.

### The tournament simulator (`sim/`)

`sim.simulate` predicts all 72 group matches, applies the **official FIFA
qualification rules** (`rules.py`), seeds the **official bracket** (`bracket.py` —
R32 template, feeder tree and semi-final pairing recovered from the dataset itself,
no hard-coded table), and predicts every knockout to a champion. `sim.validate`
scores the whole thing against reality; `sim.continue_live` does the same but from
the tournament's *actual* current state. Output is a shareable report with an ASCII
bracket of the final eight.

### The score model (`sim/goals.py`)

Models each side's goals as Poisson → full scoreline grid → every market. The λ
estimator is pluggable (`glm` default, `gbm`/XGBoost-style optional). Best variant
uses **opponent-adjusted attack/defence strengths** — a Dixon-Coles Poisson fit on
the 49k internationals (`intl.team_strengths()`) that de-conflates a team's scoring
from *who it played*. A `--knockout` flag applies the observed ~10% knockout goal
deflation.

---

## Accuracy (measured, walk-forward on the played 2026 matches)

Every number below is from predicting matches the model **had not seen**, either a
future tournament or the next real fixture. No cherry-picking.

**W/D/L — live prediction** (`continue_live` backtest: Each match predicted from only
the earlier ones):

| Metric | 2026 specialist |
| --- | -: |
| Overall result accuracy | **68.8%** (53/77) |
| Knockout result accuracy | **79.3%** (23/29) |
| Knockout **advancer** accuracy | **86.2%** (25/29) |

**Cross-era temporal test** (train ≤2018, test 2026):

| Model | log loss ↓ | Brier ↓ | accuracy ↑ |
| --- | -: | -: | -: |
| Random forest | **0.865** | **0.512** | **0.624** |
| Poisson (Dixon-Coles) | 0.913 | 0.540 | 0.594 |
| Base-rate baseline | 1.057 | 0.638 | 0.485 |

**Scores & markets**

| Score model | Exact | Within-1 | Over/Under 2.5 | Total-goals MAE |
| --- | -: | -: | -: | -: |
| GLM (Elo + value) | 16% | 67% | 64% | 1.29 |
| **GLM + opp-adjusted intl** | 16% | 67% | **70%** | 1.24 |

Exact scorelines top out at ~15–20% for any team. Coarser markets like
**over/under** are far more reliable.

---

## Setup

Two World Cup datasets live alongside this folder as `../worldcup/` and
`../FIFA-World-Cup-2026-Dataset/`. Generated artifacts (`data/unified.db`,
`outputs/`) are rebuilt by the pipeline and git-ignored.

**The international-results upgrade** (best score model): download
`results.csv` from Kaggle's
[International football results 1872–2024](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017),
save it as `./data/international_results.csv`, and run `python -m wcpredictor.sim.intl`.
Everything auto-activates once the file is present; until then it degrades cleanly.
