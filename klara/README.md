# AWARE / ENGAGE KTN 2026 CatBoost solution

Each row is an open controller task; each sample is a simultaneous snapshot of candidates. Exactly one labelled candidate is active. The output ranks three distinct `row_index` values per sample. Evaluation is the fraction of samples whose true candidate appears in those three, so row accuracy is not an appropriate objective.

## Data and documentation

Original files remain in the parent directory, unchanged. The official presentation is `../../AWARE_prioritisation_challenge.pdf`; its extracted text is saved in `outputs/presentation.txt`. It confirms missing values mean not applicable, repeated scenarios, whole-controller evaluation, and permission to train on train plus validation.

- Train: 432,578 rows, 11,923 samples, 34 sessions, 16 controllers.
- Validation: 42,110 rows, 1,252 samples, 019sc1/019sc2/020sc1.
- Test: 66,409 rows, 1,863 samples, 020sc2/020sc3/023sc1/023sc2/023sc3.
- 96 labelled columns (95 test), including metadata. Full dtypes, missingness, distributions, session counts, task mapping and input hashes are in `outputs/`.

The audit checks one positive per labelled sample, unique row indices within a sample, sequence lengths, contiguous grouping, and disjoint sample/session sets. Controller 020 is present in validation and public test; controller 023 is entirely unseen. The private leaderboard cannot be estimated from public results.

## Leakage controls and validation

No random row or snapshot split is used. Three-fold GroupKFold holds out complete controllers from **train.csv only**, thereby holding out all of their sessions. Official validation is a separate diagnostic. Models and coarse classifier/ranker blend weights are selected using mean controller-CV top-3. A structural single model is preferred when it is within 0.3 percentage points of the best identity/time model; a blend must improve the current choice by more than 0.2 points. These small conservative margins were set before identity/time CV results. Per-fold sessions, controllers, sample counts, scores and timings appear in `outputs/experiments.csv`; mean/std and worst fold in `outputs/cv_summary.csv`.

These scores are model-selection estimates: checkpoints and configurations are compared on these folds, so they are not an unbiased nested-CV estimate or a guarantee of test performance. Snapshots are correlated; sample-level binomial confidence intervals would overstate certainty. Exploratory official runs select their own checkpoints. After CV selects the architecture and tree counts, a fresh train-only model is evaluated on official validation with fixed counts and no early stopping; this produces the delivered validation predictions. Earlier official results were visible during development, so it is still a diagnostic rather than a blind external test.

Session, ATCO, sample ID, row index, raw task name and timestamp are excluded from base/engineered models. Categorical task codes have no numeric ordering. No label-derived temporal or aircraft history is created. Existing action-history inputs are retained as supplied. Identity/time variants are explicit ablations, never silently added. Relative context uses only the candidates in the same snapshot. Time ablation uses unlabelled session timestamps. No test information is used to choose models.

## Features and models

Baseline uses all numeric predictive inputs plus categorical `task_index` and `scenario` and numeric candidate count (`seq_len`). Numeric NaNs remain NaNs. Engineered features add absolute sector/exit distances or times, log task age, snapshot-relative medians and percentile ranks, missing flags, task-specific operational interactions, and task/twin categories. Active-task interaction NaNs remain missing; an interaction outside its task family is zero by definition. No target encoding is implemented outside CatBoost.

CatBoostClassifier uses Logloss, ranking raw margins within each snapshot. A positive-weight-5 variant tests imbalance handling. CatBoostRanker uses YetiRank with contiguous sample groups; PrecisionAt:top=3 monitors early stopping (exactly one positive means this is hit rate divided by 3). Both use CPU, fixed seed, early stopping, and top-3 checkpoint selection. Classifier early stopping monitors Logloss, but saved checkpoints are selected using actual top-3. Ranker and classifier blends use within-snapshot softmax scores with a coarse, CV-only weight grid.

Defaults: up to 700 trees, depth 6 (plus one depth-8 capacity comparison), learning rate 0.06, L2=5, seed 42, at most 8 CPU threads. The installed runtime reports zero CUDA GPUs. Final tree counts are the median selected counts across controller folds, then models are retrained on train + validation. All ranking and submission ties break deterministically by row_index.

## Reproduce

Python 3.12 was used. An isolated `.venv` exists here (it inherits bundled runtime packages; the portable setup below creates a standalone environment). No global packages were modified. To reproduce elsewhere:

```bash
cd /Users/klarakatic/Desktop/engage2/engage-ktn-2026-competition/catboost_solution
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python check_pipeline.py
.venv/bin/python 01_explore.py
.venv/bin/python 02_baseline.py
.venv/bin/python 03_catboost_classifier.py
.venv/bin/python 04_catboost_ranker.py
.venv/bin/python 05_compare_models.py
.venv/bin/python 06_train_final.py
.venv/bin/python 07_report.py
```

`05_compare_models.py` resumes completed model/fold predictions. To run fresh experiments, use a fresh copy of the solution directory without generated `outputs/pred_*.csv` and models. Do not reuse prediction caches after changing features, training parameters, input data, or fold definitions. Experiment CSV replaces rows with matching model/scheme names.

To regenerate the submission from the saved final models:

```bash
cd /Users/klarakatic/Desktop/engage2/engage-ktn-2026-competition/catboost_solution
.venv/bin/python 06_train_final.py --predict-only
```

To independently score official validation predictions:

```bash
.venv/bin/python ../scorer.py --labelled ../validation.csv outputs/validation_predictions.csv
```

## Baselines and deliverables

Random (seed 42) official-validation top-3: 9.50%; analytic expectation: 9.79%. The presentation's nearest-absolute-time-to-sector rule scores 30.67%. Its predictions do not match the supplied sample submission, whose generating code was not supplied. The quoted 35.2% is a test reference, not an offline measurement. Test labels are unavailable.

`submission.csv` is validated against every test sample and candidate, then read back and revalidated. `outputs/submission_validation.json` records the check, including SHA-256 confirmation that original competition files did not change. The template controls column schema and sample ordering. Component-only alternatives are saved as `outputs/submission_classifier_engineered.csv` and `outputs/submission_ranker_base.csv` for the selected blend; `submission.csv` remains the CV-selected primary entry.

`outputs/error_analysis.csv` reports top-1/2/3 by task, scenario, controller, session, list size, twin flag, conflict and zone indicators. Selected detailed errors and confusion counts are also saved, including true rank 4 and which task types ranked above misses. `outputs/feature_importance.csv` describes the primary component; a blend also saves `ranker_feature_importance.csv`. Both use CatBoost PredictionValuesChange, a model dependence measure, **not causality**. OOF analysis provides broader coverage than the three official-validation sessions. SHAP and label-based smoothing are not required or used.

Final measured results are recorded in `outputs/results.md` after the experiments finish.

## Run results

Selected models: classifier_engineered, ranker_base; final trees: {'classifier_engineered': 600, 'ranker_base': 600}.

Official validation (fixed CV-chosen tree counts): top-1 35.6230%, top-2 51.9968%, **top-3 61.4217%** (1252 snapshots).

Controller CV: selected mean fold top-3 **64.0176%**. Pooled OOF top-3 64.0191%; equal-controller mean 64.5837%.

See `outputs/results.md` for all measured comparisons and errors.
