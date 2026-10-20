# Klara's entry: CatBoost blend

## Files for the comparison

- `../predictions/klara_val.csv` — predictions for `validation.csv`, from models trained on
  **`train.csv` only** (`05_compare_models.py`, final block: `model.fit(pool(tr))`, fixed
  tree counts from controller CV, no eval set and no early stopping on validation).
  Official scorer: **top-3 = 0.6142**.
- `../predictions/klara_test.csv` — the test-set entry (trained on train + validation).
- `outputs/experiment_conclusions.md` — every post-selection experiment with fold scores.

Run `python compare.py predictions/erna_val.csv predictions/klara_val.csv` from the repo root
(with `validation.csv` present).

## Model

Blend of `classifier_engineered` (CatBoost Logloss, engineered features) and `ranker_base`
(CatBoost YetiRank), softmax-normalised within each snapshot, weight 0.5, 600 trees each.
Selected by 3-fold **controller-held-out** CV on `train.csv` (0.6402); official validation was
kept as a separate diagnostic. Session, ATCO, sample id, row index, raw task name and
timestamp are excluded from the models.

`python 06_train_final.py --predict-only` reproduces the test entry from the saved models;
`python 05_compare_models.py` reproduces the validation predictions.

## One thing worth discussing before we submit

The two models are far apart (0.6142 vs 0.9353), and the gap comes from how each treats
later snapshots of a session, not from model strength.

`features.py` on the `erna` branch builds features from the whole session, past **and**
future — its own comment says so. The strongest of them, `t_until_task_end`
(`task_last_ts - ts`), is the time until the task leaves the session, and a task leaves the
session once the controller has handled it. Measured on `validation.csv`
(`13_leakage_audit.py`):

| | median time until the task leaves | within 3 s |
|---|---|---|
| target row | 9 s | 33.0% |
| other candidates | 300 s | 2.2% |

Sorting candidates by that one column, with no model at all, gives **top-3 = 0.6270** —
above this entire trained pipeline. The same applies to `t_next_input_flight`,
`next_input_is_flight`, `in_next` and all `tn_*` columns.

This is not a rules violation: `test.csv` contains whole sessions, so those columns are
computable for the submission and the leaderboard score should hold up. The problem is the
jury round. AWARE prioritises tasks inside a live 3-second snapshot, where it is not knowable
when a task will disappear or which flight receives the next input, and the brief puts
generalisation to an unseen controller (023) above public-leaderboard score.

So the choice is a real one, and it is yours as much as mine:
- `klara_val.csv` scores lower but every input is available at run time;
- `erna_val.csv` scores much higher and will likely rank higher on both leaderboards.

A third option: keep both, submit the stronger one for the leaderboard, and be ready to show
the operational model and this audit if we reach the jury. What I would avoid is arriving at
the jury with only the future-based model and no measurement of where its score comes from.

I tried to port the legitimate, past-only parts of your approach into my model —
`src/causal.py`, 23 features: time since the task appeared, presence in the previous snapshot,
time since the last state transition per flight, which flight holds the controller's attention,
per-aircraft task grouping, rank within task type. Also your z-score ensemble normalisation,
column/row subsampling and seed averaging. Measured on controller-held-out CV, none of them
cleared my adoption margin (details in `outputs/experiment_conclusions.md`) — which suggests
the gap really is the future information, not the rest of the feature engineering.
