"""SHAP analysis of the selected classifier component.

Global mean |SHAP| ranks features by average effect on the raw margin. The
competition metric only depends on the ORDER of candidates inside a snapshot,
so a feature whose SHAP contribution is identical for every candidate of a
snapshot cannot change the prediction, however large its global importance.
Within-snapshot SHAP spread measures the part of a feature's effect that can
actually move a candidate into or out of the top three.
"""
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from src.training import ROOT, OUT, MODELS, CONFIGS, pool
from src.validation import controller_folds

spec = CONFIGS['classifier_engineered']
tr = pd.read_csv(ROOT.parent / 'train.csv')
rows = []
for fold, _, vi in controller_folds(tr):
    d = tr.iloc[vi].sort_values(['sample_id', 'row_index']).reset_index(drop=True)
    model = CatBoostClassifier()
    model.load_model(str(MODELS / f'classifier_engineered_cv{fold}.cbm'))
    sv = model.get_feature_importance(pool(d, spec), type='ShapValues')[:, :-1].astype('float32')
    names = model.feature_names_
    g = d.sample_id.to_numpy()
    sv_df = pd.DataFrame(sv, columns=names)
    # Part of each contribution that differs between candidates of the same snapshot.
    centred = sv_df - sv_df.groupby(g).transform('mean')
    rows.append(pd.DataFrame({
        'fold': fold, 'feature': names,
        'mean_abs_shap': np.abs(sv).mean(axis=0),
        'within_snapshot_spread': centred.abs().mean(axis=0).to_numpy(),
    }))
    del sv, sv_df, centred
    print('fold', fold, 'done', flush=True)

per_fold = pd.concat(rows, ignore_index=True)
per_fold.to_csv(OUT / 'shap_per_fold.csv', index=False)
agg = per_fold.groupby('feature').agg(
    mean_abs_shap=('mean_abs_shap', 'mean'),
    shap_fold_std=('mean_abs_shap', 'std'),
    within_snapshot_spread=('within_snapshot_spread', 'mean'),
    spread_fold_min=('within_snapshot_spread', 'min'),
).reset_index()
agg['ranking_share'] = agg.within_snapshot_spread / agg.within_snapshot_spread.sum()
# A feature is only useful for the metric if it separates candidates in EVERY fold.
agg['usable'] = agg.spread_fold_min > 0
agg = agg.sort_values('within_snapshot_spread', ascending=False)
agg.to_csv(OUT / 'shap_feature_summary.csv', index=False)
print(f'features: {len(agg)}  with zero within-snapshot spread in >=1 fold: {(~agg.usable).sum()}')
print(agg.head(25).to_string(index=False))
print('\nlowest within-snapshot spread:')
print(agg.tail(25).to_string(index=False))
cum = agg.ranking_share.cumsum()
for frac in [.9, .95, .99, .999]:
    print(f'features needed for {frac:.1%} of within-snapshot SHAP mass: {int((cum < frac).sum() + 1)}')
