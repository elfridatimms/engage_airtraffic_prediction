"""Within-snapshot score normalisation for the blend, tested on cached CV predictions.

The current blend uses a softmax over each snapshot. A z-score per snapshot
(as in the colleague's ensemble) and a plain rank average are alternatives that
cost nothing to evaluate, since both components' fold predictions are saved.
"""
import json
import numpy as np, pandas as pd
from src.training import ROOT, OUT, CONFIGS
from src.validation import controller_folds
from src.metrics import topk

tr = pd.read_csv(ROOT.parent / 'train.csv')
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
def rankavg(df, v):
    z = pd.Series(v, index=df.index)
    return z.groupby(df.sample_id).rank(pct=True).to_numpy()

NORMS = {'softmax': softmax, 'zscore': zscore, 'rank': rankavg}
PAIRS = [('classifier_engineered','ranker_base'), ('classifier_engineered','ranker_causal'),
         ('classifier_engineered','classifier_weighted'), ('classifier_engineered','classifier_time')]
rows = []
for c, r in PAIRS:
    for nname, nf in NORMS.items():
        for w in [.25,.4,.5,.6,.75]:
            fs = []
            for f, _, vi in controller_folds(tr):
                d = tr.iloc[vi].reset_index(drop=True)
                fs.append(topk(d, w*nf(d, scores(c,d,f'cv{f}')) + (1-w)*nf(d, scores(r,d,f'cv{f}'))))
            rows.append(dict(component_a=c, component_b=r, normalisation=nname, weight_a=w,
                             mean_top3=float(np.mean(fs)), std_top3=float(np.std(fs,ddof=1)),
                             worst_fold=float(np.min(fs)), fold_scores=json.dumps(fs)))
b = pd.DataFrame(rows).sort_values('mean_top3', ascending=False)
b.to_csv(OUT / 'blend_normalisation_experiments.csv', index=False)
print(b.head(15).to_string(index=False))
cur = json.loads((OUT / 'selected_config.json').read_text())['cv_mean_top3']
best = b.iloc[0]
print(f"\ncurrent selected: {cur:.6f}   best here: {best.mean_top3:.6f}   delta: {best.mean_top3-cur:+.6f}"
      f"   adopt (> +0.002): {best.mean_top3 > cur + .002}")
