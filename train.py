"""Final pipeline: ensemble of LightGBM rankers/classifier on session-temporal features.

    python train.py          -> train on train+validation, write submission.csv for test.csv
    python train.py val      -> train on train.csv only, write val_submission.csv for validation.csv
                                (check with: python scorer.py --labelled validation.csv val_submission.csv)
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import build, build2, feature_cols, load

BASE = dict(objective="lambdarank", metric="ndcg", eval_at=[3], learning_rate=0.05,
            num_leaves=31, min_data_in_leaf=50, feature_fraction=0.7, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, lambdarank_truncation_level=10, verbose=-1,
            feature_pre_filter=False)
BINARY = {k: v for k, v in BASE.items() if k not in ("eval_at", "lambdarank_truncation_level")}
MODELS = [
    (BASE, 400),
    ({**BASE, "num_leaves": 15, "min_data_in_leaf": 100}, 600),
    ({**BASE, "num_leaves": 63, "min_data_in_leaf": 30}, 300),
    ({**BASE, "objective": "rank_xendcg"}, 400),
    ({**BINARY, "objective": "binary", "metric": "auc"}, 400),
]
SEEDS = 2


def features():
    if os.path.exists("feats2.parquet"):
        return pd.read_parquet("feats2.parquet")
    df = build2(build(load()))
    df.to_parquet("feats2.parquet")
    return df


def zscore(p, groups):
    s = pd.Series(p).groupby(groups)
    return ((pd.Series(p) - s.transform("mean")) / (s.transform("std") + 1e-9)).values


def ensemble(trn, tst, cols):
    trn = trn.sort_values(["sample_id", "row_index"])
    grp = trn.groupby("sample_id", sort=False).size().values
    ds = lgb.Dataset(trn[cols], trn["target_index"], group=grp, categorical_feature=["task_index"],
                     free_raw_data=False)
    score = np.zeros(len(tst))
    for params, rounds in MODELS:
        for s in range(SEEDS):
            p = lgb.train({**params, "seed": s}, ds, rounds).predict(tst[cols])
            score += zscore(p, tst["sample_id"].values)
    return score


def top3(df, col):
    df = df.sort_values(["sample_id", col], ascending=[True, False])
    return (df.groupby("sample_id")["row_index"].apply(lambda x: " ".join(map(str, x.values[:3])))
            .reset_index().rename(columns={"row_index": "prediction"}))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    df = features()
    cols = feature_cols(df)
    if mode == "val":
        trn, tst, out = df[df.split == 0], df[df.split == 1].copy(), "val_submission.csv"
    else:
        trn, tst, out = df[df.split < 2], df[df.split == 2].copy(), "submission.csv"
    tst = tst.reset_index(drop=True)
    tst["score"] = ensemble(trn, tst, cols)
    top3(tst, "score").to_csv(out, index=False)
    print("wrote", out, len(tst.sample_id.unique()), "samples")


if __name__ == "__main__":
    main()
