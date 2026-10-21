"""Look-ahead model v2: episode / run structure features on top of features.py (build + build2).

The data only has snapshots at labelled moments, so a run of snapshots without a gap is one or more consecutive
label episodes; an episode usually ends when the controller acts or the task disappears.
    python runfeat.py cv     -> controller-grouped CV: old features vs + run features
    python runfeat.py val    -> train on train.csv, val_submission_v2.csv (score with scorer.py)
    python runfeat.py test   -> train on train+validation, submission_v2.csv
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import feature_cols
from train import BASE, ensemble, top3


def build3(df, thr=4.5, p="run"):
    sn = (df[["session", "sample_id", "ts"]].drop_duplicates("sample_id")
          .sort_values(["session", "ts"]).reset_index(drop=True))
    gp = sn.groupby("session").ts.diff()
    sn["rid"] = (gp.isna() | (gp > thr)).cumsum()
    g = sn.groupby("rid").ts
    sn["r_start"], sn["r_end"] = g.transform("min"), g.transform("max")
    runs = sn.drop_duplicates("rid")[["rid", "session", "r_start", "r_end"]]
    runs["next_start"] = runs.groupby("session").r_start.shift(-1)
    runs["prev_end"] = runs.groupby("session").r_end.shift(1)
    sn = sn.merge(runs[["rid", "next_start", "prev_end"]], on="rid")
    f = pd.DataFrame({"sample_id": sn.sample_id, f"{p}_rid": sn.rid,
                      f"{p}_len": sn.r_end - sn.r_start,
                      f"{p}_to_end": sn.r_end - sn.ts,
                      f"{p}_from_start": sn.ts - sn.r_start,
                      f"{p}_gap_after": (sn.next_start - sn.r_end).fillna(3600),
                      f"{p}_gap_before": (sn.r_start - sn.prev_end).fillna(3600)})
    f[f"{p}_pos"] = f[f"{p}_from_start"] / (f[f"{p}_len"] + 3)
    df = df.merge(f, on="sample_id", how="left")
    to_end, from_start = df[f"{p}_to_end"], df[f"{p}_from_start"]
    df[f"{p}_input_minus_end"] = df.t_next_input_flight - to_end
    df[f"{p}_taskend_minus_end"] = df.t_until_task_end - to_end
    df[f"{p}_task_ends_in_run"] = (df.t_until_task_end < to_end - 0.5).astype(int)
    df[f"{p}_input_in_run"] = (df.t_next_input_flight <= to_end + 6).astype(int)
    df[f"{p}_prev_input_minus_start"] = df.a_since_flight - from_start
    df[f"{p}_task_new_in_run"] = (df.task_first_ts >= df.ts - from_start - 0.5).astype(int)
    pres = set(zip(df.st, df[f"{p}_rid"]))
    df[f"{p}_in_next_run"] = [int((s, r + 1) in pres) for s, r in zip(df.st, df[f"{p}_rid"])]
    df[f"{p}_in_prev_run"] = [int((s, r - 1) in pres) for s, r in zip(df.st, df[f"{p}_rid"])]
    sg = df.groupby("sample_id")
    df[f"{p}_n_end_in_run"] = sg[f"{p}_task_ends_in_run"].transform("sum")
    df[f"{p}_n_input_in_run"] = sg[f"{p}_input_in_run"].transform("sum")
    for c in [f"{p}_input_minus_end", f"{p}_taskend_minus_end"]:
        df["abs_" + c] = df[c].abs()
        df["r_abs_" + c] = sg["abs_" + c].rank(method="min")
    return df


def load():
    df = pd.read_parquet("feats2.parquet")
    df = build3(df, 4.5, "run")
    df = build3(df, 7.5, "run7")
    return df.sort_values(["sample_id", "row_index"]).reset_index(drop=True)


def cols_of(df, with_runs=True):
    c = [x for x in feature_cols(df) if not x.endswith("_rid")]
    return c if with_runs else [x for x in c if "run" not in x]


def hit(d, p):
    r = pd.Series(p).groupby(d.sample_id.values).rank(ascending=False, method="first").values
    m = d.target_index.values == 1
    return (r[m] <= 3).mean(), (r[m] == 1).mean(), r


TYPES = {0: "entry_confl", 1: "exit_confl", 2: "entry_coord", 3: "exit_coord", 4: "assume", 5: "transfer",
         6: "conflict", 7: "nonconf", 8: "qos", 9: "entry_cond", 10: "exit_cond", 12: "zone", 13: "ret_route"}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    df = load()
    if mode in ("val", "test"):
        from penalty import chain_adjust  # next episode is a different task (tau=2, lam=1 from controller CV)
        cols = cols_of(df)
        trn = df[df.split == 0] if mode == "val" else df[df.split < 2]
        tst = (df[df.split == 1] if mode == "val" else df[df.split == 2]).reset_index(drop=True)
        s = pd.Series(ensemble(trn, tst, cols))
        g = s.groupby(tst.sample_id.values)
        tst["z"] = ((s - g.transform("mean")) / (g.transform("std") + 1e-9)).values
        tst[["sample_id", "row_index", "session", "ts", "run_rid", "task_name", "flight", "task_index", "z"]].to_parquet(
            "v2_%s_scores.parquet" % mode)  # for decoding variants without retraining
        tst["score"] = tst.z + chain_adjust(tst, "z", 2.0, "fb") / 2.0
        out = "val_submission_v2.csv" if mode == "val" else "submission_v2.csv"
        top3(tst, "score").to_csv(out, index=False)
        top3(tst, "z").to_csv(out.replace(".csv", "_nochain.csv"), index=False)
        print(len(cols), "features, wrote", out)
        return
    lab = df[df.split < 2].reset_index(drop=True)
    folds = np.array_split(np.random.RandomState(0).permutation(np.array(sorted(lab.atco.unique()))), 6)

    def cv(cols, params=BASE, rounds=400):
        out = np.zeros(len(lab))
        for f in folds:
            te = lab.atco.isin(f).values
            trn = lab[~te]
            grp = trn.groupby("sample_id", sort=False).size().values
            ds = lgb.Dataset(trn[cols], trn.target_index, group=grp, categorical_feature=["task_index"])
            out[te] = lgb.train({**params, "seed": 0}, ds, rounds).predict(lab.loc[te, cols])
        return out

    res = {}
    for name, cols in [("old features", cols_of(lab, False)), ("+ run features", cols_of(lab, True))]:
        p = cv(cols)
        t3, t1, r = hit(lab, p)
        res[name] = r
        print(f"{name:16s} {len(cols)} feats  top3=%.4f top1=%.4f" % (t3, t1), flush=True)
    t = lab[lab.target_index == 1][["task_index", "session"]].copy()
    for name, r in res.items():
        t[name] = r[lab.target_index.values == 1] > 3
    print(t.groupby(t.task_index.map(TYPES))[list(res)].mean().round(3).assign(n=t.groupby(t.task_index.map(TYPES)).size())
          .sort_values("n", ascending=False).to_string())
    s = t.groupby("session")[list(res)].mean()
    d = s["old features"] - s["+ run features"]
    print("sessions better: %d, worse: %d" % ((d > 0).sum(), (d < 0).sum()))


if __name__ == "__main__":
    main()
