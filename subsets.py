"""Feature subsets by importance; strict splits:
   A) session + controller: test fold = all sessions of held-out controllers (no session, no controller shared)
   B) session + controller + scenario: test = held-out controllers in one scenario; train = other controllers, other scenarios"""
import numpy as np, pandas as pd
from pure import load, snapshot_features, fit_predict, hit3
df = snapshot_features(load()); lab = df[df.split < 2].reset_index(drop=True)
imp = pd.read_csv("feature_importance.csv", index_col=0).gain_pct
sets = {"all 88": list(imp.index), "drop lt0.1pct (57)": list(imp[imp >= 0.1].index),
        "top 40": list(imp.index[:40]), "top 20": list(imp.index[:20])}
ctrl_folds = np.array_split(np.random.RandomState(0).permutation(np.array(sorted(lab.atco.unique()))), 6)

def check(tr, te):
    assert not set(lab.session[tr]) & set(lab.session[te]), "session overlap"
    assert not set(lab.atco[tr]) & set(lab.atco[te]), "controller overlap"

def cv_A(cols):
    out = np.zeros(len(lab))
    for f in ctrl_folds:
        te = lab.atco.isin(f).values; tr = ~te; check(tr, te)
        out[te] = fit_predict(lab[tr], lab[te].reset_index(drop=True), cols, seeds=(0, 1))
    return hit3(lab, out)[0]

def cv_B(cols):
    out = np.full(len(lab), np.nan)
    for f in ctrl_folds:
        for s in [1, 2, 3]:
            te = (lab.atco.isin(f) & (lab.scenario == s)).values
            tr = (~lab.atco.isin(f) & (lab.scenario != s)).values
            if te.any():
                check(tr, te)
                out[te] = fit_predict(lab[tr], lab[te].reset_index(drop=True), cols, seeds=(0, 1))
    return hit3(lab, out)[0]

for k, c in sets.items():
    print("{:20s} session+controller={:.4f}  session+controller+scenario={:.4f}".format(k, cv_A(c), cv_B(c)), flush=True)
