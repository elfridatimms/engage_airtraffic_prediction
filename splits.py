"""Same model (raw AWARE columns), different validation splits -> how optimistic each one is."""
import numpy as np, pandas as pd
from pure import load, snapshot_features, fit_predict, hit3
df = snapshot_features(load()); lab = df[df.split < 2].reset_index(drop=True)
raw = [c for c in pd.read_csv("test.csv", nrows=1).columns
       if c not in ("sample_id", "row_index", "session", "atco", "scenario", "timestamp", "task_name")]
rng = np.random.RandomState(0)

def run(name, test_masks, train_masks=None):
    out = np.full(len(lab), np.nan)
    for i, te in enumerate(test_masks):
        tr = ~te if train_masks is None else train_masks[i]
        out[te] = fit_predict(lab[tr], lab[te].reset_index(drop=True), raw)
    m = ~np.isnan(out)
    print(f"{name:45s} top3=%.4f top1=%.4f" % hit3(lab[m].reset_index(drop=True), out[m]), flush=True)

def groups(key, n):
    g = np.array(sorted(lab[key].unique()))
    return [lab[key].isin(f).values for f in np.array_split(rng.permutation(g), n)]

# 1 random samples (WRONG: near-duplicate neighbours in train and test)
sid = lab.sample_id.unique(); rng.shuffle(sid)
run("1. random samples (wrong)", [lab.sample_id.isin(f).values for f in np.array_split(sid, 6)])
run("2. by session", groups("session", 6))
run("3. by controller", groups("atco", 6))
run("4. by scenario", groups("scenario", 3))
# 5 new controller AND new scenario: test = fold controllers in scenario s; train = other controllers, other scenarios
te_m, tr_m = [], []
for f in np.array_split(rng.permutation(np.array(sorted(lab.atco.unique()))), 6):
    for s in [1, 2, 3]:
        te = (lab.atco.isin(f) & (lab.scenario == s)).values
        if te.any():
            te_m.append(te); tr_m.append((~lab.atco.isin(f) & (lab.scenario != s)).values)
run("5. by controller AND scenario", te_m, tr_m)
# 6 official: train.csv -> validation.csv
run("6. official train -> validation", [(lab.split == 1).values], [(lab.split == 0).values])
