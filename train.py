"""Session-grouped CV of a LightGBM ranker + temporal smoothing; writes submission.csv."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import build, feature_cols, load

SMOOTH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
SEEDS = 3
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[3], learning_rate=0.03,
              num_leaves=31, min_data_in_leaf=50, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, lambdarank_truncation_level=10, verbose=-1)
ROUNDS = 600


def fit_predict(trn, tst, cols):
    trn = trn.sort_values(["sample_id", "row_index"])
    grp = trn.groupby("sample_id", sort=False).size().values
    pred = np.zeros(len(tst))
    for s in range(SEEDS):
        ds = lgb.Dataset(trn[cols], trn["target_index"], group=grp, categorical_feature=["task_index"])
        m = lgb.train({**PARAMS, "seed": s}, ds, ROUNDS)
        pred += m.predict(tst[cols])
    return pred / SEEDS, m


def smooth(df, col, alpha):
    """blend each task's score with its scores in neighbouring snapshots (same session, same task)"""
    if alpha <= 0:
        return df[col]
    d = df[["st", "snap_k", col]].copy()
    d["z"] = d.groupby("snap_k")[col].transform(lambda x: x)  # placeholder keep shape
    # standardise scores within sample
    d["z"] = df.groupby("sample_id")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    lut = dict(zip(zip(d["st"], d["snap_k"]), d["z"]))
    out = d["z"].values.copy()
    for off, w in [(-2, alpha / 2), (-1, alpha), (1, alpha), (2, alpha / 2)]:
        nb = np.array([lut.get((s, k + off), np.nan) for s, k in zip(d["st"], d["snap_k"])])
        out += w * np.nan_to_num(nb, nan=-1.0)
    return out


def top3(df, col):
    df = df.sort_values(["sample_id", col], ascending=[True, False])
    return df.groupby("sample_id")["row_index"].apply(lambda x: " ".join(map(str, x.values[:3])))


def hit(df, col):
    df = df.sort_values(["sample_id", col], ascending=[True, False])
    df["rk"] = df.groupby("sample_id").cumcount()
    return df.loc[df.target_index == 1].assign(h=lambda x: x.rk < 3).groupby("session")["h"].mean()


def main():
    df = build(load())
    cols = feature_cols(df)
    print(len(cols), "features")
    lab = df[df.split < 2].copy()
    sessions = np.array(sorted(lab.session.unique()))
    rng = np.random.RandomState(0)
    folds = np.array_split(rng.permutation(sessions), 6)
    lab["oof"] = 0.0
    for f in folds:
        mtr = ~lab.session.isin(f)
        p, _ = fit_predict(lab[mtr], lab[~mtr], cols)
        lab.loc[~mtr, "oof"] = p
    for a in [0, 0.25, 0.5, 1.0]:
        lab["sm"] = smooth(lab, "oof", a)
        h = hit(lab, "sm")
        n = lab[lab.target_index == 1].groupby("session").size()
        print(f"smooth={a}: weighted top3={np.average(h, weights=n[h.index]):.4f}  "
              f"mean-session={h.mean():.4f}  ctrl020={h.get('020sc1', np.nan):.3f}")
    # final model on all labelled data
    te = df[df.split == 2].copy()
    te["p"], m = fit_predict(lab, te, cols)
    te["sm"] = smooth(te, "p", SMOOTH)
    sub = top3(te, "sm").reset_index().rename(columns={"row_index": "prediction"})
    sub.to_csv("submission.csv", index=False)
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print(imp.head(25).round(0))


if __name__ == "__main__":
    main()
