"""Real-time (causal) model: only information available at the snapshot time.

No future snapshots, no features that depend on which moments were labelled.
    python causal.py cv      -> session-grouped CV
    python causal.py val     -> train on train.csv, write predictions/erna_causal_val.csv
    python causal.py test    -> train on train+validation, write submission_causal.csv
"""
import re
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import feature_cols

FUTURE = re.compile(r"(t_until|t_next|tn_|in_next|in_prev|gap_next|gap_prev|task_ends_before|"
                    r"next_input_is|end_minus|t_since_task_start)")
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[3], learning_rate=0.03,
              num_leaves=31, min_data_in_leaf=100, feature_fraction=0.6, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, lambdarank_truncation_level=10, verbose=-1,
              feature_pre_filter=False)
ROUNDS = 500
MAX_LAG = 15.0  # seconds: a past snapshot older than this is treated as missing


def past_features(df):
    """lags from the most recent earlier snapshot of the same task (within MAX_LAG s)"""
    df = df.sort_values(["st", "ts"]).copy()
    g = df.groupby("st")
    dt = df["ts"] - g["ts"].shift()
    ok = dt <= MAX_LAG
    df["lag_dt"] = dt.where(ok)
    for c in ["time_to_sector", "a_since_flight", "a_n60", "x_state_age", "l_rank_absf1_all",
              "conflict_ind", "time_to_MD", "distance_to_zone", "k_cfl_minus_fl",
              "non_conformance_agreed_route_deviation_horizontally"]:
        df["d_" + c] = (df[c] - g[c].shift()).where(ok)
    # input on this flight within the last 10 / 30 / 60 s
    for w in [10, 30, 60]:
        df[f"inp_{w}"] = (df["a_since_flight"] <= w).astype(int)
    # input on this flight came after the task appeared (controller already started on it)
    df["inp_after_task"] = (df["a_since_flight"] < df["x_task_age"]).astype(int)
    df["task_age_minus_inp"] = df["x_task_age"] - df["a_since_flight"]
    # type-specific readiness signals
    df["assume_ready"] = ((df.task_index == 4) & (df.x_transfer_in == 1)).astype(int)
    df["transfer_ready"] = ((df.task_index == 5) & (df.k_aircraftNeedsToBeTransferred == 1)).astype(int)
    sg = df.groupby("sample_id")
    for c in ["task_age_minus_inp", "x_state_age", "k_distanceToExitPoint", "lag_dt"]:
        df["r_" + c] = sg[c].rank(method="min")
    df["n_inp_30_sample"] = sg["inp_30"].transform("sum")
    return df


def crowd_prior(df, bin_s=60, k=5.0, max_split=1):
    """how often OTHER controllers worked on this task in the same scenario at the same aircraft position.
    Leave-one-session-out for labelled sessions, so a session never sees its own labels."""
    df = df.copy()
    df["tbin"] = (df["time_to_sector"].clip(-1800, 1800) // bin_s).astype(int)
    rel = df["ts"] - df.groupby("session")["ts"].transform("min")
    df["rbin"] = (rel // 90).astype(int)
    lab = df[df.split <= max_split]  # only labels allowed for this run
    out = {}
    for name, key in [("cp_task", ["scenario", "task_name", "tbin"]),
                      ("cp_task_all", ["scenario", "task_name"]),
                      ("cp_flight", ["scenario", "flight", "tbin"]),
                      ("cp_type", ["task_index", "tbin"]),
                      ("cp_time", ["scenario", "task_name", "rbin"]),
                      ("cp_time_flight", ["scenario", "flight", "rbin"])]:
        tot = lab.groupby(key)["target_index"].agg(["sum", "count"])
        per = lab.groupby(["session"] + key)["target_index"].agg(["sum", "count"])
        r = df[["session"] + key].join(tot, on=key).join(per, on=["session"] + key, rsuffix="_own")
        h = r["sum"].fillna(0) - r["sum_own"].fillna(0)
        n = r["count"].fillna(0) - r["count_own"].fillna(0)
        base = h.sum() / max(n.sum(), 1)
        out[name] = ((h + k * base) / (n + k)).values
        out[name + "_n"] = n.values
    for name, v in out.items():
        df[name] = v
    sg = df.groupby("sample_id")
    for c in ["cp_task", "cp_flight", "cp_task_all", "cp_time", "cp_time_flight"]:
        df["r_" + c] = sg[c].rank(ascending=False, method="min")
        df["z_" + c] = df[c] / (sg[c].transform("sum") + 1e-9)
    return df.drop(columns=["tbin", "rbin"])


def scenario_time(df):
    """align sessions of the same scenario: offset from when each aircraft reaches the sector boundary"""
    x = df[(df.within_sector == 0) & (df.approaching_sector == 1) & df.time_to_sector.between(0, 1800)]
    c = (x.ts + x.time_to_sector).groupby([x.session, x.scenario, x.flight]).median().rename("c").reset_index()
    c["d"] = c.c - c.groupby(["scenario", "flight"]).c.transform("median")
    off = c.groupby("session").d.median()
    return df["ts"] - df["session"].map(off)


def aligned_prior(df, max_split=1, bin_s=30, k=3.0):
    """LOSO rate at which OTHER controllers worked on this task/flight at the same aligned scenario time,
    and at shifted times (what others did shortly before / after this point of the scenario)"""
    df = df.copy()
    df["abin"] = (scenario_time(df) // bin_s).astype(int)
    lab = df[df.split <= max_split]
    for name, key in [("ap_task", ["scenario", "task_name"]), ("ap_flight", ["scenario", "flight"]),
                      ("ap_type", ["scenario", "task_index"])]:
        tot = lab.groupby(key + ["abin"])["target_index"].agg(["sum", "count"])
        per = lab.groupby(["session"] + key + ["abin"])["target_index"].agg(["sum", "count"])
        base = lab["target_index"].mean()
        for sh in [-4, -2, -1, 0, 1, 2, 4, 8]:
            q = df[["session"] + key].copy()
            q["abin"] = df["abin"] + sh
            r = q.join(tot, on=key + ["abin"]).join(per, on=["session"] + key + ["abin"], rsuffix="_own")
            h = r["sum"].fillna(0) - r["sum_own"].fillna(0)
            n = r["count"].fillna(0) - r["count_own"].fillna(0)
            df[f"{name}_{sh}"] = ((h + k * base) / (n + k)).values
            if sh == 0:
                df[f"{name}_n"] = n.values
        cols = [f"{name}_{sh}" for sh in [-1, 0, 1, 2]]
        df[f"{name}_near"] = df[cols].mean(axis=1)
        df[f"{name}_ahead"] = df[[f"{name}_{sh}" for sh in [1, 2, 4]]].mean(axis=1)
    sg = df.groupby("sample_id")
    for c in ["ap_task_0", "ap_task_near", "ap_task_ahead", "ap_flight_near"]:
        df["r_" + c] = sg[c].rank(ascending=False, method="min")
        df["z_" + c] = df[c] / (sg[c].transform("sum") + 1e-9)
    return df.drop(columns="abin")


def list_events(df, max_gap=15.0):
    """past-only list dynamics between consecutive snapshots (<= max_gap s apart):
    tasks that just disappeared (likely finished) and tasks that just appeared"""
    snaps = df[["session", "sample_id", "ts"]].drop_duplicates("sample_id").sort_values(["session", "ts"])
    snaps["prev"] = snaps.groupby("session")["sample_id"].shift()
    snaps["gap"] = snaps["ts"] - snaps.groupby("session")["ts"].shift()
    snaps = snaps[snaps.gap <= max_gap]
    tasks = df.groupby("sample_id")["task_name"].apply(set).to_dict()
    gone = []
    for sid, prev, ts, sess in zip(snaps.sample_id, snaps.prev, snaps.ts, snaps.session):
        for t in tasks[int(prev)] - tasks[sid]:
            gone.append((sess, t.rsplit("_", 1)[-1], t, ts))
    gone = pd.DataFrame(gone, columns=["session", "flight", "task_name", "ts"])
    gone["type"] = gone.task_name.str.rsplit("_", n=1).str[0]
    out = df[["session", "flight", "ts", "sample_id", "task_name"]].copy()
    sf = out.session + "|" + out.flight
    for name, sel in [("gone_any", gone), ("gone_assume", gone[gone.type == "assume_aircraft"]),
                      ("gone_coord", gone[gone.type.str.contains("coordination")]),
                      ("gone_transfer", gone[gone.type == "transfer_aircraft"])]:
        ev = {k: np.sort(v.values) for k, v in sel.groupby(sel.session + "|" + sel.flight)["ts"]}
        res = np.full(len(out), 3600.0)
        for i, (k, t) in enumerate(zip(sf.values, out.ts.values)):
            e = ev.get(k)
            if e is not None:
                j = np.searchsorted(e, t, side="right")
                if j > 0:
                    res[i] = t - e[j - 1]
        df["ts_" + name] = res
    # any task of the session gone in the last 15 s (controller just finished something)
    ev_s = {k: np.sort(v.values) for k, v in gone.groupby("session")["ts"]}
    res = np.full(len(out), 3600.0)
    for i, (k, t) in enumerate(zip(out.session.values, out.ts.values)):
        e = ev_s.get(k)
        if e is not None:
            j = np.searchsorted(e, t, side="right")
            if j > 0:
                res[i] = t - e[j - 1]
    df["ts_gone_session"] = res
    # task newly appeared since previous close snapshot
    prev_of = dict(zip(snaps.sample_id, snaps.prev))
    df["is_new_task"] = [(-1 if sid not in prev_of else int(tn not in tasks[int(prev_of[sid])]))
                         for sid, tn in zip(df.sample_id, df.task_name)]
    sg = df.groupby("sample_id")
    df["r_ts_gone_any"] = sg["ts_gone_any"].rank(method="min")
    return df


def causal_cols(df):
    return [c for c in feature_cols(df) if not FUTURE.search(c)]


def zscore(p, groups):
    s = pd.Series(p).groupby(groups)
    return ((pd.Series(p) - s.transform("mean")) / (s.transform("std") + 1e-9)).values


def past_scores(df, col):
    """stage-2 inputs: this task's stage-1 score in the previous snapshots (past only)"""
    d = df[["st", "ts", "sample_id"]].copy()
    d["z"] = zscore(df[col].values, df["sample_id"].values)
    d["rk"] = d.groupby("sample_id")["z"].rank(ascending=False, method="min")
    d = d.sort_values(["st", "ts"])
    g = d.groupby("st")
    out = pd.DataFrame(index=d.index)
    out["ps_z0"] = d["z"]
    out["ps_rk0"] = d["rk"]
    for k in [1, 2, 3]:
        dt = d["ts"] - g["ts"].shift(k)
        ok = dt <= MAX_LAG * k
        out[f"ps_z{k}"] = g["z"].shift(k).where(ok)
        out[f"ps_rk{k}"] = g["rk"].shift(k).where(ok)
    out["ps_zmean"] = out[["ps_z0", "ps_z1", "ps_z2", "ps_z3"]].mean(axis=1)
    out["ps_top1_streak"] = (out[["ps_rk1", "ps_rk2", "ps_rk3"]] == 1).sum(axis=1)
    return out.loc[df.index]


def train(trn, cols, params=PARAMS, rounds=ROUNDS, seeds=(0,)):
    trn = trn.sort_values(["sample_id", "row_index"])
    grp = trn.groupby("sample_id", sort=False).size().values
    ds = lgb.Dataset(trn[cols], trn["target_index"], group=grp, categorical_feature=["task_index"],
                     free_raw_data=False)
    return [lgb.train({**params, "seed": s}, ds, rounds) for s in seeds]


def predict(models, df, cols):
    return sum(zscore(m.predict(df[cols]), df["sample_id"].values) for m in models)


def folds(lab, n=6):
    s = np.array(sorted(lab.session.unique()))
    return np.array_split(np.random.RandomState(0).permutation(s), n)


def oof(lab, cols, **kw):
    out = np.zeros(len(lab))
    for f in folds(lab):
        te = lab.session.isin(f).values
        out[te] = predict(train(lab[~te], cols, **kw), lab[te], cols)
    return out


def hit3(lab, p):
    d = lab[["sample_id", "target_index"]].assign(p=p)
    d["rk"] = d.groupby("sample_id")["p"].rank(ascending=False, method="first")
    t = d[d.target_index == 1]
    return (t.rk <= 3).mean(), (t.rk == 1).mean()


def two_stage(trn, tst, cols):
    """stage-1 OOF on trn (session folds) -> stage-2 on past scores; returns tst scores"""
    trn = trn.copy()
    trn["p1"] = oof(trn, cols)
    tst = tst.copy()
    tst["p1"] = predict(train(trn, cols, seeds=(0, 1, 2)), tst, cols)
    trn = pd.concat([trn, past_scores(trn, "p1")], axis=1)
    tst = pd.concat([tst, past_scores(tst, "p1")], axis=1)
    c2 = cols + [c for c in trn.columns if c.startswith("ps_")]
    return predict(train(trn, c2, rounds=300, seeds=(0, 1, 2)), tst, c2)


USE_PRIOR = "--noprior" not in sys.argv
VAL_MODE = len(sys.argv) > 1 and sys.argv[1] == "val"


def load():
    df = pd.read_parquet("feats2.parquet")
    df = past_features(df)
    df = list_events(df)
    if USE_PRIOR:
        df = crowd_prior(df, max_split=0 if VAL_MODE else 1)
        df = aligned_prior(df, max_split=0 if VAL_MODE else 1)
    return df.sort_values(["sample_id", "row_index"]).reset_index(drop=True)


def top3(df, col):
    df = df.sort_values(["sample_id", col], ascending=[True, False])
    return (df.groupby("sample_id")["row_index"].apply(lambda x: " ".join(map(str, x.values[:3])))
            .reset_index().rename(columns={"row_index": "prediction"}))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    df = load()
    cols = causal_cols(df)
    lab = df[df.split < 2].reset_index(drop=True)
    if mode == "cv":
        print(len(cols), "causal features")
        p1 = oof(lab, cols)
        print("stage-1  top3=%.4f top1=%.4f" % hit3(lab, p1))
        p2 = np.zeros(len(lab))
        for f in folds(lab):
            te = lab.session.isin(f).values
            p2[te] = two_stage(lab[~te], lab[te], cols)
        print("stage-2  top3=%.4f top1=%.4f" % hit3(lab, p2))
        return
    if mode == "val":
        trn, tst, out = df[df.split == 0], df[df.split == 1], "predictions/erna_causal_val.csv"
    else:
        trn, tst, out = lab, df[df.split == 2], "submission_causal.csv"
    tst = tst.reset_index(drop=True)
    tst["score"] = two_stage(trn.reset_index(drop=True), tst, cols)
    top3(tst, "score").to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
