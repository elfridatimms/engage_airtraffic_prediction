"""Subsampling regularisation and seed averaging on controller-held-out folds.

Column and row subsampling, and averaging several seeds, are the parts of the
colleague's LightGBM setup that carry no future information. Both target
variance rather than fit, which is what an unseen controller needs.
"""
import json
import numpy as np, pandas as pd
from src.training import ROOT, OUT, CONFIGS, fit
from src.validation import controller_folds
from src.metrics import topk

NEW = ['classifier_subsample', 'ranker_subsample', 'classifier_engineered_s7', 'classifier_engineered_s13']
tr = pd.read_csv(ROOT.parent / 'train.csv')
for name in NEW:
    for f, ti, vi in controller_folds(tr):
        if not (OUT / f'pred_{name}_cv{f}.csv').exists():
            fit(name, tr.iloc[ti], tr.iloc[vi], f'cv{f}', 700)

def scores(name, frame, scheme):
    p = pd.read_csv(OUT / f'pred_{name}_{scheme}.csv')
    m = frame[['sample_id','row_index']].merge(p, on=['sample_id','row_index'], how='left', validate='one_to_one')
    assert m.score.notna().all(); return m.score.to_numpy()
def softmax(df, v):
    z = pd.Series(v, index=df.index); z = np.exp(z - z.groupby(df.sample_id).transform('max'))
    return (z / z.groupby(df.sample_id).transform('sum')).to_numpy()
def zscore(df, v):
    z = pd.Series(v, index=df.index); g = z.groupby(df.sample_id)
    return ((z - g.transform('mean')) / (g.transform('std') + 1e-9)).to_numpy()

exp = pd.read_csv(OUT / 'experiments.csv'); cv = exp[exp.validation_scheme.str.startswith('cv')]
print(cv.groupby('model').agg(mean_top3=('top3','mean'), worst=('top3','min'), folds=('top3','size'))
      .sort_values('mean_top3', ascending=False).to_string(), flush=True)

SEEDS = ['classifier_engineered', 'classifier_engineered_s7', 'classifier_engineered_s13']
rows = []
def evaluate(label, parts, norm):
    fs = []
    for f, _, vi in controller_folds(tr):
        d = tr.iloc[vi].reset_index(drop=True)
        s = np.zeros(len(d))
        for name, w in parts:
            s += w * norm(d, scores(name, d, f'cv{f}'))
        fs.append(topk(d, s))
    rows.append(dict(configuration=label, normalisation=norm.__name__, mean_top3=float(np.mean(fs)),
                     std_top3=float(np.std(fs, ddof=1)), worst_fold=float(np.min(fs)), fold_scores=json.dumps(fs)))

for norm in (softmax, zscore):
    evaluate('seed_avg_classifier(3 seeds)', [(n, 1/3) for n in SEEDS], norm)
    for w in [.5, .6, .75]:
        evaluate(f'seed_avg_classifier + ranker_base w={w}',
                 [(n, w/3) for n in SEEDS] + [('ranker_base', 1-w)], norm)
        evaluate(f'classifier_subsample + ranker_base w={w}',
                 [('classifier_subsample', w), ('ranker_base', 1-w)], norm)
        evaluate(f'classifier_subsample + ranker_subsample w={w}',
                 [('classifier_subsample', w), ('ranker_subsample', 1-w)], norm)
        evaluate(f'seed_avg + ranker_subsample w={w}',
                 [(n, w/3) for n in SEEDS] + [('ranker_subsample', 1-w)], norm)
b = pd.DataFrame(rows).sort_values('mean_top3', ascending=False)
b.to_csv(OUT / 'regularisation_experiments.csv', index=False)
print(); print(b.head(18).to_string(index=False), flush=True)
cur = json.loads((OUT / 'selected_config.json').read_text())['cv_mean_top3']
best = b.iloc[0]
print(f"\ncurrent selected: {cur:.6f}   best here: {best.mean_top3:.6f}   delta: {best.mean_top3-cur:+.6f}"
      f"   adopt (> +0.002): {best.mean_top3 > cur + .002}", flush=True)
