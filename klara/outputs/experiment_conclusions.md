# Post-selection experiments: SHAP, past-only session dynamics, regularisation

All numbers are mean top-3 over the same 3-fold controller-held-out CV on `train.csv`
used by `05_compare_models.py`. The selected configuration scores **0.640176**
(blend of `classifier_engineered` + `ranker_base`, softmax, weight 0.5). The
adoption margin is unchanged from the original protocol: a candidate must exceed
the current selection by more than 0.2 percentage points. Fold SD is about 0.006,
so smaller differences are not separable on three folds.

## 1. SHAP

CatBoost `ShapValues` were computed on each held-out fold from the saved fold
models (`08_shap_analysis.py`). Global mean |SHAP| was not the useful view: the
metric depends only on the ORDER of candidates inside a snapshot, so a
contribution identical for every candidate of a snapshot cannot change the
prediction. The audit therefore reports **within-snapshot SHAP spread**.

Findings: 26 of 118 features have zero within-snapshot spread in at least one
fold, and 54 features carry 90% of the ranking-relevant mass
(`shap_feature_summary.csv`).

Pruning to those subsets was tested (`09_shap_models.py`):

| model | features | mean top-3 | vs `classifier_engineered` (0.636587) |
|---|---|---|---|
| `classifier_shap_usable` | 92 | 0.630642 | -0.59 pp |
| `classifier_shap89` | 89 | 0.629409 | -0.72 pp |
| `classifier_shap67` | 67 | 0.628304 | -0.83 pp |
| `classifier_shap54` | 54 | 0.625994 | -1.06 pp |

Accuracy degrades monotonically as features are removed. **SHAP was not adopted
for feature selection.** It is retained as a diagnostic: it is what established
that the within-snapshot component, not global importance, is the quantity the
metric responds to. CatBoost's built-in SHAP was used; no external `shap`
package is required.

## 2. Past-only session dynamics (`src/causal.py`)

23 features describing how a task and its aircraft evolved **before** the
current snapshot: time since the task first appeared, snapshots seen so far,
presence in the previous snapshot, time since the last transition of nine flight
state columns, whether this flight received the controller's most recent input,
competing tasks of the same aircraft, and urgency rank within task type.

| model | mean top-3 | worst fold |
|---|---|---|
| `classifier_causal` | 0.631473 | 0.624360 |
| `ranker_causal` | 0.627388 | 0.625720 |
| best blend containing a causal component | 0.639224 | 0.635619 |

No variant reaches the threshold; the best is 0.10 pp below the current
selection. The existing `x_task_age`, `l_rank_*` and `a_since_*` inputs already
carry this information, so the additions saturate rather than add.

## 3. Blend normalisation (`11_blend_normalisation.py`)

Softmax, per-snapshot z-score and per-snapshot rank average, over four component
pairs and five weights, from cached fold predictions. Best: z-score,
`classifier_engineered` + `ranker_base`, weight 0.75 -> **0.641379** (+0.12 pp).

Below the threshold, and it is the maximum of 60 configurations evaluated on the
same folds, with a top band spanning 0.6385-0.6414 against a fold SD of 0.007.
**Not adopted**, because the margin exists to reject exactly this.

## 4. Subsampling and seed averaging (`12_regularisation.py`)

Column/row subsampling (`rsm=0.7`, `subsample=0.8`, Bernoulli) and averaging
three seeds. Both target variance, which is the relevant failure mode for an
unseen controller.

| configuration | mean top-3 |
|---|---|
| `seed_avg_classifier` + `ranker_base`, z-score, w=0.5 | 0.639710 |
| `classifier_subsample` (single) | 0.634409 |
| `ranker_subsample` (single) | 0.632860 |

Best 0.639710, i.e. 0.05 pp below the current selection. **Not adopted.**

## 5. Why future-derived features are excluded (`13_leakage_audit.py`)

The supplied files contain whole sessions, so later snapshots are readable for
the test set too. Features built from them are computable for a submission but
not by a system running in real time, because a task leaves the session only
after the controller has acted on it.

Measured on official validation, a single future-derived feature used as the
entire model, with no training at all:

| feature | information | top-3 |
|---|---|---|
| time until the task leaves the session, ascending | **future** | **0.626997** |
| `x_task_age`, descending | past | 0.118211 |
| time since the task appeared, descending | past | 0.092652 |
| `time_to_sector`, ascending | past | 0.078275 |

| | median time until the task leaves | within 3 s |
|---|---|---|
| target row | 9 s | 33.0% |
| other candidates | 300 s | 2.2% |

One such feature, sorted, beats this entire trained pipeline (0.6142). That is
the measure of how much label it carries, and the reason it is excluded.

## Outcome

No candidate cleared the margin. The selected configuration and `submission.csv`
are unchanged (SHA-256 `fceb21a1...f804c`, matching `reproducibility_check.txt`).
Every experiment above is kept with its fold scores so the negative results are
auditable rather than discarded.
