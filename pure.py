"""PURE real-time model: only what AWARE provides in ONE snapshot, no ids, no labels in features,
nothing that depends on which moments were annotated (no previous-snapshot features).
    python pure.py cv     -> strict CV grouped by controller AND leave-one-scenario-out
    python pure.py val    -> train on train.csv, predictions/erna_pure_val.csv
    python pure.py test   -> train on train+validation, submission_pure.csv"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ID = {"sample_id", "row_index", "target_index", "session", "atco", "scenario", "timestamp", "task_name"}
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[3], learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=100, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, lambdarank_truncation_level=10, verbose=-1, feature_pre_filter=False)


def load():
    tr, va, te = (pd.read_csv(f) for f in ["train.csv", "validation.csv", "test.csv"])
    te["target_index"] = -1
    df = pd.concat([tr.assign(split=0), va.assign(split=1), te.assign(split=2)], ignore_index=True)
    return df.sort_values(["sample_id", "row_index"]).reset_index(drop=True)


def snapshot_features(df):
    """features computed only from the current snapshot (rows with the same sample_id)"""
    df = df.copy()
    df["flight"] = df.task_name.str.rsplit("_", n=1).str[-1]
    conf = df.task_index.isin([0, 1, 6])
    df.loc[conf, "flight"] = df.loc[conf, "task_name"].str.rsplit("_", n=2).str[-2]
    df["abs_tts"] = df.time_to_sector.abs()
    sg = df.groupby("sample_id")
    for c in ["x_task_age", "a_since_flight", "abs_tts", "x_state_age", "distance_to_zone", "time_to_MD",
              "k_distanceToExitPoint"]:
        df["r_" + c] = sg[c].rank(method="min")
    df["n_flights"] = sg["flight"].transform("nunique")
    fg = df.groupby(["sample_id", "flight"])["x_task_age"]
    df["age_minus_flight_max"] = df.x_task_age - fg.transform("max")
    df["age_minus_flight_min"] = df.x_task_age - fg.transform("min")
    df["rt_x_task_age"] = df.groupby(["sample_id", "task_index"])["x_task_age"].rank(method="min")
    for w in [10, 30, 60]:
        df[f"inp_{w}"] = (df.a_since_flight <= w).astype(int)
    df["inp_after_task"] = (df.a_since_flight < df.x_task_age).astype(int)
    df["task_age_minus_inp"] = df.x_task_age - df.a_since_flight
    df["r_task_age_minus_inp"] = df.groupby("sample_id")["task_age_minus_inp"].rank(method="min")
    df["assume_ready"] = ((df.task_index == 4) & (df.x_transfer_in == 1)).astype(int)
    df["transfer_ready"] = ((df.task_index == 5) & (df.k_aircraftNeedsToBeTransferred == 1)).astype(int)
    df["n_inp_30_sample"] = df.groupby("sample_id")["inp_30"].transform("sum")
    return df


def cols_of(df):
    return [c for c in df.columns if c not in ID | {"split", "flight"}]


def zscore(p, g):
    s = pd.Series(p).groupby(g)
    return ((pd.Series(p) - s.transform("mean")) / (s.transform("std") + 1e-9)).values


def fit_predict(trn, tst, cols, seeds=(0,)):
    grp = trn.groupby("sample_id", sort=False).size().values
    ds = lgb.Dataset(trn[cols], trn.target_index, group=grp, categorical_feature=["task_index"],
                     free_raw_data=False)
    return sum(zscore(lgb.train({**PARAMS, "seed": s}, ds, 500).predict(tst[cols]), tst.sample_id.values)
               for s in seeds)


def hit3(d, p):
    r = pd.Series(p).groupby(d.sample_id.values).rank(ascending=False, method="first").values
    m = d.target_index.values == 1
    return (r[m] <= 3).mean(), (r[m] == 1).mean()


def group_cv(lab, cols, key, n):
    g = np.array(sorted(lab[key].unique()))
    out = np.zeros(len(lab))
    for f in np.array_split(np.random.RandomState(0).permutation(g), n):
        te = lab[key].isin(f).values
        out[te] = fit_predict(lab[~te], lab[te], cols)
    return out


def top3(d, p):
    d = d.assign(p=p).sort_values(["sample_id", "p"], ascending=[True, False])
    return (d.groupby("sample_id").row_index.apply(lambda x: " ".join(map(str, x.values[:3])))
            .reset_index().rename(columns={"row_index": "prediction"}))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    df = snapshot_features(load())
    cols = cols_of(df)
    if mode == "cv":
        lab = df[df.split < 2].reset_index(drop=True)
        print(len(cols), "features")
        raw = [c for c in cols if c in pd.read_csv("test.csv", nrows=1).columns]
        for name, cs in [("raw AWARE columns only", raw), ("pure (raw + snapshot features)", cols)]:
            a = group_cv(lab, cs, "atco", 6)
            s = group_cv(lab, cs, "scenario", 3)
            print(f"{name}: by controller top3=%.4f top1=%.4f | by scenario top3=%.4f top1=%.4f"
                  % (*hit3(lab, a), *hit3(lab, s)), flush=True)
    else:
        cols = [c for c in cols if c in pd.read_csv("test.csv", nrows=1).columns]  # raw AWARE columns only
        print(len(cols), "raw AWARE features")
        trn = df[df.split == 0] if mode == "val" else df[df.split < 2]
        tst = (df[df.split == 1] if mode == "val" else df[df.split == 2]).reset_index(drop=True)
        out = "predictions/erna_pure_val.csv" if mode == "val" else "submission_pure.csv"
        top3(tst, fit_predict(trn, tst, cols, seeds=(0, 1, 2, 3, 4))).to_csv(out, index=False)
        print("wrote", out)
