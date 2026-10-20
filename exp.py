"""Fast experiment loop on cached features: OOF CV + error analysis."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import feature_cols

PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[3], learning_rate=0.05,
              num_leaves=31, min_data_in_leaf=50, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, lambdarank_truncation_level=10, verbose=-1, seed=0)
TYPES = {0: "entry_confl", 1: "exit_confl", 2: "entry_coord", 3: "exit_coord", 4: "assume", 5: "transfer",
         6: "conflict", 7: "nonconf", 8: "qos", 9: "entry_cond", 10: "exit_cond", 12: "zone", 13: "ret_route"}


def oof(lab, cols, params=PARAMS, rounds=400):
    sessions = np.array(sorted(lab.session.unique()))
    folds = np.array_split(np.random.RandomState(0).permutation(sessions), 6)
    out = np.zeros(len(lab))
    for f in folds:
        te = lab.session.isin(f).values
        trn = lab[~te]
        grp = trn.groupby("sample_id", sort=False).size().values
        ds = lgb.Dataset(trn[cols], trn["target_index"], group=grp, categorical_feature=["task_index"])
        out[te] = lgb.train(params, ds, rounds).predict(lab.loc[te, cols])
    return out


def ranks(lab, p):
    d = lab[["sample_id", "session", "target_index", "task_index"]].assign(p=p)
    d["rk"] = d.groupby("sample_id")["p"].rank(ascending=False, method="first")
    return d


def report(lab, p, name=""):
    d = ranks(lab, p)
    t = d[d.target_index == 1]
    h = t.groupby("session").apply(lambda x: (x.rk <= 3).mean())
    print(f"{name}: top3={np.mean(t.rk <= 3):.4f}  top1={np.mean(t.rk == 1):.4f}  "
          f"worst sessions: {h.sort_values().head(4).round(3).to_dict()}")
    return d


if __name__ == "__main__":
    df = pd.read_parquet("feats2.parquet")
    lab = df[df.split < 2].sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    cols = feature_cols(lab)
    d = report(lab, oof(lab, cols), "base")
    t = d[d.target_index == 1].copy()
    t["miss"] = t.rk > 3
    print(t.groupby(t.task_index.map(TYPES)).agg(n=("miss", "size"), miss=("miss", "mean")).sort_values("n", ascending=False))
    # what is ranked first when we miss?
    top1 = d[d.rk == 1].set_index("sample_id")["task_index"].map(TYPES)
    m = t[t.miss]
    print(pd.crosstab(m.task_index.map(TYPES), m.sample_id.map(top1)))
