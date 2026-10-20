"""Controller-CV test of the past-only session-dynamics features.

Selection margin follows 05_compare_models.py: a change must beat the current
selected configuration by more than 0.2 percentage points of mean fold top-3.
"""
import json
import numpy as np, pandas as pd
from src.training import ROOT, OUT, CONFIGS, fit
from src.validation import controller_folds
from src.metrics import topk

NEW = ['classifier_causal', 'ranker_causal']
tr = pd.read_csv(ROOT.parent / 'train.csv')
for name in NEW:
    for f, ti, vi in controller_folds(tr):
        if not (OUT / f'pred_{name}_cv{f}.csv').exists():
            fit(name, tr.iloc[ti], tr.iloc[vi], f'cv{f}', 700)

exp = pd.read_csv(OUT / 'experiments.csv'); cv = exp[exp.validation_scheme.str.startswith('cv')]
s = cv.groupby('model').agg(mean_top3=('top3','mean'), std_top3=('top3','std'),
                            worst_fold=('top3','min'), folds=('top3','size')).sort_values('mean_top3', ascending=False)
print(s.to_string(), flush=True)

def normalize(df, v):
    z = pd.Series(v, index=df.index); z = np.exp(z - z.groupby(df.sample_id).transform('max'))
    return (z / z.groupby(df.sample_id).transform('sum')).to_numpy()
def scores(name, frame, scheme):
    p = pd.read_csv(OUT / f'pred_{name}_{scheme}.csv')
    m = frame[['sample_id','row_index']].merge(p, on=['sample_id','row_index'], how='left', validate='one_to_one')
    assert m.score.notna().all(); return m.score.to_numpy()

rows = []
for c, r in [('classifier_causal','ranker_causal'), ('classifier_causal','ranker_base'),
             ('classifier_engineered','ranker_causal')]:
    for w in [.25,.5,.75]:
        fs = []
        for f, _, vi in controller_folds(tr):
            d = tr.iloc[vi].reset_index(drop=True)
            fs.append(topk(d, w*normalize(d, scores(c,d,f'cv{f}')) + (1-w)*normalize(d, scores(r,d,f'cv{f}'))))
        rows.append(dict(classifier=c, ranker=r, classifier_weight=w, mean_top3=float(np.mean(fs)),
                         std_top3=float(np.std(fs,ddof=1)), worst_fold=float(np.min(fs)), fold_scores=json.dumps(fs)))
b = pd.DataFrame(rows).sort_values('mean_top3', ascending=False)
b.to_csv(OUT / 'causal_ensemble_experiments.csv', index=False)
print(); print(b.to_string(index=False), flush=True)
cur = json.loads((OUT / 'selected_config.json').read_text())['cv_mean_top3']
best = b.iloc[0]
print(f"\ncurrent selected: {cur:.6f}   best causal blend: {best.mean_top3:.6f}   "
      f"delta: {best.mean_top3-cur:+.6f}   adopt (> +0.002): {best.mean_top3 > cur + .002}", flush=True)
