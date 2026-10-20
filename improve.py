"""Real-time ideas, tested with the strict session+controller split (whole sessions, no controller in train and test).

B) controller's personal timing so far in THIS session, from the input log (only inputs before the snapshot):
   how long this controller lets an offered aircraft wait before assuming it, how early before the exit
   point they transfer. Compared with each open task (e.g. this aircraft has waited as long as this
   controller usually lets them wait).
C) controller's current 'mode': time since their last assume / transfer / level / lateral input, last input type.
D) causal smoothing: blend the model's score with its own score for the same task 3-4 s earlier.

AWARE logs every input, so at snapshot time a real system knows all inputs made before it. The dataset has no
raw log, so it is rebuilt from the a_ columns (time of last input on each aircraft + its type); only inputs at
or before the snapshot time are used for that snapshot.

    python improve.py cv
    python improve.py test <B|C|BC|none> <alpha>   -> submission_improved.csv
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from pure import PARAMS, hit3, load, snapshot_features, top3, zscore

ID = ["sample_id", "row_index", "session", "atco", "scenario", "timestamp", "task_name"]
TYPES = ["a_last_assume_control", "a_last_force_assume_control", "a_last_assign_level_clearance",
         "a_last_assign_lateral_clearance", "a_last_initiate_transfer_of_control", "a_last_decontrol"]
B = ["p_n_assume", "p_assume_delay", "p_wait_rel", "p_n_transfer", "p_transfer_tte", "p_tte_rel"]
C = ["since_own_assume", "since_own_transfer", "since_own_level", "since_own_lateral", "last_own_type"]


def input_log(df):
    """(session, flight, t, type) of every input, rebuilt from the a_ columns"""
    e = df.loc[df.a_since_flight < 3600, ["session", "flight", "ts", "a_since_flight"] + TYPES]
    e = e.drop_duplicates(["session", "flight", "ts"]).copy()
    e["t"] = e.ts - e.a_since_flight
    f = e[TYPES].fillna(0).values
    e["type"] = np.where(f.sum(1) > 0, f.argmax(1) + 1, 0)  # 1 assume 2 force 3 level 4 lateral 5 transfer 6 decontrol
    e = e.sort_values(["session", "flight", "t"])
    new = (e.session != e.session.shift()) | (e.flight != e.flight.shift()) | (e.t - e.t.shift() > 2.5)
    e["eid"] = new.cumsum()
    ev = e.groupby("eid").agg(session=("session", "first"), flight=("flight", "first"),
                              t=("t", "min"), type=("type", "max")).reset_index(drop=True)
    # state of the aircraft just before the input (last snapshot at least 1 s earlier), to tell this
    # controller's inputs from the neighbouring sectors' (offers in, assumes by the next sector)
    st = (df[["session", "flight", "ts", "is_assumed", "k_timeToExitPoint"]]
          .drop_duplicates(["session", "flight", "ts"]).sort_values("ts"))
    ev["tq"] = ev.t - 1.0
    ev = pd.merge_asof(ev.sort_values("tq"), st, left_on="tq", right_on="ts", by=["session", "flight"],
                       direction="backward", tolerance=120.0)
    ev["own_assume"] = ev.type.isin([1, 2]) & (ev.is_assumed == 0)
    ev["own_transfer"] = (ev.type == 5) & (ev.is_assumed == 1)
    ev["own_level"] = (ev.type == 3) & (ev.is_assumed == 1)
    ev["own_lateral"] = (ev.type == 4) & (ev.is_assumed == 1)
    # personal timing values
    off = df.loc[df.x_transfer_in == 1, ["session", "flight", "ts", "x_state_age"]].drop_duplicates(
        ["session", "flight", "ts"])
    off = (off.ts - off.x_state_age).groupby([off.session, off.flight]).median().rename("offer_t")
    ev = ev.join(off, on=["session", "flight"])
    ev["assume_delay"] = (ev.t - ev.offer_t).where(ev.own_assume)
    ev.loc[~ev.assume_delay.between(0, 1800), "assume_delay"] = np.nan
    ev["transfer_tte"] = (ev.k_timeToExitPoint - (ev.t - ev.ts)).where(ev.own_transfer)
    ev.loc[~ev.transfer_tte.between(-600, 3600), "transfer_tte"] = np.nan
    return ev.drop(columns=["tq", "ts"])


def running_mean(snaps, ev, col):
    n = np.zeros(len(snaps))
    m = np.full(len(snaps), np.nan)
    for s, idx in snaps.groupby("session").indices.items():
        e = ev[(ev.session == s) & ev[col].notna()].sort_values("t")
        if e.empty:
            continue
        k = np.searchsorted(e.t.values, snaps.ts.values[idx], side="right")  # inputs at or before snapshot
        cs = np.concatenate([[0.0], np.cumsum(e[col].values)])
        n[idx] = k
        m[idx] = np.where(k > 0, cs[k] / np.maximum(k, 1), np.nan)
    return n, m


def since_last(snaps, ev, flag):
    out = np.full(len(snaps), 3600.0)
    for s, idx in snaps.groupby("session").indices.items():
        t = np.sort(ev.loc[(ev.session == s) & ev[flag], "t"].values)
        if len(t) == 0:
            continue
        ts = snaps.ts.values[idx]
        k = np.searchsorted(t, ts, side="right")
        ok = k > 0
        out[idx[ok]] = np.minimum(3600.0, ts[ok] - t[k[ok] - 1])
    return out


def add_features(df):
    df = df.copy()
    df["ts"] = pd.to_timedelta(df.timestamp).dt.total_seconds()
    ev = input_log(df)
    snaps = df[["sample_id", "session", "ts"]].drop_duplicates("sample_id").reset_index(drop=True)
    snaps["p_n_assume"], snaps["p_assume_delay"] = running_mean(snaps, ev, "assume_delay")
    snaps["p_n_transfer"], snaps["p_transfer_tte"] = running_mean(snaps, ev, "transfer_tte")
    names = {"own_assume": 1, "own_transfer": 2, "own_level": 3, "own_lateral": 4}
    for f in names:
        snaps["since_" + f] = since_last(snaps, ev, f)
    since = snaps[["since_" + f for f in names]].values
    snaps["last_own_type"] = np.where(since.min(1) < 3600, since.argmin(1) + 1, 0)
    df = df.merge(snaps.drop(columns=["session", "ts"]), on="sample_id", how="left")
    df["p_wait_rel"] = (df.x_state_age - df.p_assume_delay).where(df.x_transfer_in == 1)
    df["p_tte_rel"] = (df.k_timeToExitPoint - df.p_transfer_tte).where(df.is_assumed == 1)
    return df.sort_values(["sample_id", "row_index"]).reset_index(drop=True), ev


def causal_smooth(d, z, alpha, max_gap=4.5):
    """EMA of a task's score over its previous snapshots (only the past, gap <= max_gap s)"""
    if alpha <= 0:
        return z
    o = np.lexsort((d.ts.values, d.task_name.values, d.session.values))
    sess, tn, ts, zz = d.session.values[o], d.task_name.values[o], d.ts.values[o], z[o]
    out = np.empty(len(o))
    for i in range(len(o)):
        if i and sess[i] == sess[i - 1] and tn[i] == tn[i - 1] and ts[i] - ts[i - 1] <= max_gap:
            out[i] = (1 - alpha) * zz[i] + alpha * out[i - 1]
        else:
            out[i] = zz[i]
    res = np.empty(len(o))
    res[o] = out
    return res


BINARY = {**{k: v for k, v in PARAMS.items() if k not in ("eval_at", "lambdarank_truncation_level")},
          "objective": "binary", "metric": "auc"}


def fit(trn, tst, cols, params=PARAMS, rounds=500, seeds=(0, 1)):
    grp = trn.groupby("sample_id", sort=False).size().values
    cat = ["task_index"] + (["last_own_type"] if "last_own_type" in cols else [])
    ds = lgb.Dataset(trn[cols], trn.target_index, group=grp, categorical_feature=cat, free_raw_data=False)
    return sum(zscore(lgb.train({**params, "seed": s}, ds, rounds).predict(tst[cols]), tst.sample_id.values)
               for s in seeds)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    df, ev = add_features(snapshot_features(load()))
    raw = [c for c in pd.read_csv("test.csv", nrows=1).columns if c not in ID]
    sets = {"none": raw, "C": raw + C, "B": raw + B, "BC": raw + B + C}
    if mode == "test":
        cols, alpha = sets[sys.argv[2]], float(sys.argv[3])
        tst = df[df.split == 2].reset_index(drop=True)
        z = fit(df[df.split < 2], tst, cols, seeds=(0, 1, 2, 3, 4))
        top3(tst, causal_smooth(tst, zscore(z, tst.sample_id.values), alpha)).to_csv("submission_improved.csv",
                                                                                     index=False)
        print("wrote submission_improved.csv", sys.argv[2:], len(cols), "features")
        return
    own = ev[ev.own_assume]
    print("own assumes per session: %.1f, own transfers: %.1f" % (
        own.groupby("session").size().mean(), ev[ev.own_transfer].groupby("session").size().mean()))
    pc = ev.groupby("session")[["assume_delay", "transfer_tte"]].mean()
    print("per-session mean assume delay: mean %.0f s, spread (std) %.0f s; transfer time-to-exit: mean %.0f s, "
          "std %.0f s" % (pc.assume_delay.mean(), pc.assume_delay.std(), pc.transfer_tte.mean(),
                          pc.transfer_tte.std()), flush=True)
    lab = df[df.split < 2].reset_index(drop=True)
    folds = np.array_split(np.random.RandomState(0).permutation(np.array(sorted(lab.atco.unique()))), 6)

    def cv(cols, params=PARAMS, rounds=500):
        out = np.zeros(len(lab))
        for f in folds:
            te = lab.atco.isin(f).values
            assert not set(lab.session[te]) & set(lab.session[~te]) and not set(lab.atco[te]) & set(lab.atco[~te])
            out[te] = fit(lab[~te], lab[te], cols, params, rounds)
        return zscore(out, lab.sample_id.values)

    def per_session(p):
        r = pd.Series(p).groupby(lab.sample_id.values).rank(ascending=False, method="first").values
        return lab.assign(h=r <= 3)[lab.target_index == 1].groupby("session").h.mean()

    P = {}
    for name, cols in sets.items():
        P[name] = cv(cols)
        print(f"{name:12s} top3=%.4f top1=%.4f" % hit3(lab, P[name]), flush=True)
    best = max(["C", "B", "BC"], key=lambda k: hit3(lab, P[k])[0])
    P[best + "_big"] = cv(sets[best], {**PARAMS, "num_leaves": 63, "min_data_in_leaf": 50}, 600)
    P[best + "_binary"] = cv(sets[best], BINARY, 500)
    for k in [best + "_big", best + "_binary"]:
        print(f"{k:12s} top3=%.4f top1=%.4f" % hit3(lab, P[k]), flush=True)
    P[best + "+binary"] = zscore(P[best] + P[best + "_binary"], lab.sample_id.values)
    print(f"{best + '+binary':12s} top3=%.4f top1=%.4f" % hit3(lab, P[best + "+binary"]))
    print("causal smoothing:")
    for k in ["none", best, best + "+binary"]:
        for a in [0.3, 0.5, 0.7]:
            print(f"  {k:12s} alpha={a}: top3=%.4f top1=%.4f" % hit3(lab, causal_smooth(lab, P[k], a)), flush=True)
    base = per_session(P["none"])
    for k in [best, best + "+binary"]:
        d = per_session(P[k]) - base
        print(f"{k} vs none: sessions better {int((d > 0).sum())}, worse {int((d < 0).sum())}, "
              f"mean diff {d.mean():+.4f}")
    pd.DataFrame(P).assign(sample_id=lab.sample_id, row_index=lab.row_index).to_parquet("improve_oof.parquet")


if __name__ == "__main__":
    main()
