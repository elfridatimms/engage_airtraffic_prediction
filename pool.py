"""Run-level pooling of the look-ahead model's scores (controller-grouped OOF)."""
import lightgbm as lgb, numpy as np, pandas as pd
from runfeat import load, cols_of, hit
from train import BASE
df = load(); lab = df[df.split < 2].reset_index(drop=True)
# how often does the label change inside a run vs across a gap?
t = lab[lab.target_index == 1].sort_values(["session", "ts"])
same_sess = t.session.values[1:] == t.session.values[:-1]
same_run = t.run_rid.values[1:] == t.run_rid.values[:-1]
chg = t.task_name.values[1:] != t.task_name.values[:-1]
print("label change rate inside runs: %.3f  across gaps: %.3f" % (chg[same_run].mean(), chg[same_sess & ~same_run].mean()))
print("labels per run: mean %.2f, runs with 1 label: %.2f" % (t.groupby("run_rid").task_name.nunique().mean(),
      (t.groupby("run_rid").task_name.nunique() == 1).mean()))
cols = cols_of(lab)
folds = np.array_split(np.random.RandomState(0).permutation(np.array(sorted(lab.atco.unique()))), 6)
oof = np.zeros(len(lab))
for f in folds:
    te = lab.atco.isin(f).values; trn = lab[~te]
    ds = lgb.Dataset(trn[cols], trn.target_index, group=trn.groupby("sample_id", sort=False).size().values,
                     categorical_feature=["task_index"])
    oof[te] = lgb.train({**BASE, "seed": 0}, ds, 400).predict(lab.loc[te, cols])
lab["p"] = oof
g = lab.groupby("sample_id").p
lab["z"] = (lab.p - g.transform("mean")) / (g.transform("std") + 1e-9)
lab[["sample_id", "row_index", "p", "z"]].to_parquet("pool_oof.parquet")
print("no pooling: top3=%.4f top1=%.4f" % hit(lab, lab.z.values)[:2])
for rid in ["run_rid", "run7_rid"]:
    rm = lab.groupby([rid, "task_name"]).z.transform("mean")
    for b in [0.3, 0.5, 0.7, 1.0]:
        print(f"pool over {rid:8s} beta={b}: top3=%.4f top1=%.4f" % hit(lab, ((1 - b) * lab.z + b * rm).values)[:2])
# pool only over a window of +-k snapshots inside the run
lab = lab.sort_values(["session", "task_name", "ts"])
for k in [2, 4, 8]:
    gg = lab.groupby(["run_rid", "task_name"]).z
    w = gg.transform(lambda x: x.rolling(2 * k + 1, center=True, min_periods=1).mean())
    print(f"window +-{k} in run beta=0.5: top3=%.4f" % hit(lab.sort_index(), (0.5 * lab.z + 0.5 * w).sort_index().values)[0])
