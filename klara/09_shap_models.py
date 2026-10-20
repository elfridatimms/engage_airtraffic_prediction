"""Controller-CV test of the SHAP-pruned feature subsets.

Selection uses the same held-out-controller folds as 05_compare_models.py.
Official validation stays a diagnostic and is not used to choose a subset.
"""
import pandas as pd
from src.training import ROOT, OUT, CONFIGS, fit
from src.validation import controller_folds

MODELS = ['classifier_shap54', 'classifier_shap67', 'classifier_shap89', 'classifier_shap_usable']
tr = pd.read_csv(ROOT.parent / 'train.csv')
for name in MODELS:
    print('features kept:', name, len(CONFIGS[name]['keep']), flush=True)
    for f, ti, vi in controller_folds(tr):
        if not (OUT / f'pred_{name}_cv{f}.csv').exists():
            fit(name, tr.iloc[ti], tr.iloc[vi], f'cv{f}', 700)
exp = pd.read_csv(OUT / 'experiments.csv')
cv = exp[exp.validation_scheme.str.startswith('cv')]
s = cv.groupby('model').agg(mean_top3=('top3', 'mean'), std_top3=('top3', 'std'),
                            worst_fold=('top3', 'min'), folds=('top3', 'size')).sort_values('mean_top3', ascending=False)
s.to_csv(OUT / 'shap_cv_summary.csv')
print(s.to_string(), flush=True)
