"""Episode-to-episode transitions learned from TRAINING labels, used to decode a whole test session jointly.

Across a gap the next episode is a different task; how often it is the same aircraft, and which task type
follows which, is learned from training sessions only (never from the session being predicted).
Decoding: forward-backward over the runs of a session, emissions = softmax of the model's pooled run scores.
    python transitions.py cv     -> controller-grouped evaluation on pool_oof.parquet
"""
import sys

import numpy as np
import pandas as pd

NT = 14


def episodes(d):
    """consecutive run pairs in labelled data: (prev label, next run candidates)"""
    t = d[d.target_index == 1].sort_values(["session", "ts"])
    last = t.groupby("run_rid").tail(1).set_index("run_rid")
    first = t.groupby("run_rid").head(1).set_index("run_rid")
    cand = d.sort_values("ts").groupby("run_rid").head(10 ** 6)
    first_ts = cand.groupby("run_rid").ts.transform("min")
    cand = cand[cand.ts == first_ts]  # tasks in the first snapshot of each run
    sess = d.drop_duplicates("run_rid").set_index("run_rid").session
    return last, first, cand, sess


def fit_lifts(d, smooth=20.0):
    """lift = observed / chance for 'same aircraft as previous episode' and for (prev type -> next type)"""
    last, first, cand, sess = episodes(d)
    obs_same = chance_same = 0.0
    obs_t = np.zeros((NT, NT))
    chance_t = np.zeros((NT, NT))
    cg = {r: g for r, g in cand.groupby("run_rid")}
    for r in first.index:
        if r - 1 not in last.index or sess.get(r - 1) != sess.get(r) or r not in cg:
            continue
        y = last.loc[r - 1]
        g = cg[r]
        g = g[g.task_name != y.task_name]
        if len(g) == 0:
            continue
        x = first.loc[r]
        obs_same += float(x.flight == y.flight)
        chance_same += (g.flight == y.flight).mean()
        obs_t[y.task_index, x.task_index] += 1
        chance_t[y.task_index] += np.bincount(g.task_index, minlength=NT) / len(g)
    lift_same = obs_same / max(chance_same, 1e-9)
    lift_t = (obs_t + smooth * chance_t.sum(1, keepdims=True) / NT * 0 + 1.0) / (chance_t + 1.0)
    return lift_same, lift_t, obs_same / max(len(first), 1), chance_same / max(len(first), 1)


def decode(d, zcol, lift_same, lift_t, tau=2.0, use_type=True, use_same=True):
    """forward-backward over runs; returns log(posterior / emission) per row"""
    corr = np.zeros(len(d))
    for s, ds in d.groupby("session"):
        runs = []
        for rid, g in ds.groupby("run_rid"):
            pz = g.groupby("task_name")[zcol].mean()
            meta = g.drop_duplicates("task_name").set_index("task_name").loc[pz.index]
            runs.append((rid, np.asarray(pz.index, dtype=object), np.asarray(meta.flight, dtype=object),
                         meta.task_index.values.astype(int), pz.values.astype(float)))
        n = len(runs)
        q = []
        for _, _, _, _, z in runs:
            e = np.exp(tau * (z - z.max()))
            q.append(e / e.sum())
        T = []
        for k in range(n - 1):
            ny, fy, ty = runs[k][1], runs[k][2], runs[k][3]
            nx, fx, tx = runs[k + 1][1], runs[k + 1][2], runs[k + 1][3]
            w = np.ones((len(ny), len(nx)))
            if use_same:
                w *= np.where(fy[:, None] == fx[None, :], lift_same, 1.0)
            if use_type:
                w *= lift_t[ty][:, tx]
            w[ny[:, None] == nx[None, :]] = 1e-3  # same task again across a gap: almost never
            T.append(w / w.sum(1, keepdims=True))
        a = [q[0]]
        for k in range(n - 1):
            v = q[k + 1] * (a[k] @ T[k])
            a.append(v / v.sum())
        b = [None] * n
        b[n - 1] = np.ones(len(q[n - 1]))
        for k in range(n - 2, -1, -1):
            v = T[k] @ (q[k + 1] * b[k + 1])
            b[k] = v / v.sum()
        for k in range(n):
            post = a[k] * b[k]
            post /= post.sum()
            c = dict(zip(runs[k][1], np.log(np.maximum(post, 1e-12) / np.maximum(q[k], 1e-12))))
            m = (ds.run_rid == runs[k][0]).values
            corr[np.where(d.session.values == s)[0][m]] = [c[x] for x in ds.task_name.values[m]]
    return corr


if __name__ == "__main__":
    from runfeat import hit, load
    df = load()
    lab = df[df.split < 2].reset_index(drop=True)
    lab = (lab.merge(pd.read_parquet("pool_oof.parquet")[["sample_id", "row_index", "z"]], on=["sample_id", "row_index"])
           .sort_values(["sample_id", "row_index"]).reset_index(drop=True))
    ls, lt, po, pc = fit_lifts(lab)
    print("next episode on the same aircraft: observed %.3f vs chance %.3f -> lift %.1f" % (po, pc, ls))
    names = {0: "entry_confl", 1: "exit_confl", 2: "entry_coord", 3: "exit_coord", 4: "assume", 5: "transfer",
             6: "conflict", 7: "nonconf", 8: "qos", 10: "exit_cond", 12: "zone", 13: "ret_route"}
    top = [(names.get(i, i), names.get(j, j), lt[i, j]) for i in names for j in names if lt[i, j] > 2.5]
    print("strong type transitions (lift>2.5):", sorted(top, key=lambda x: -x[2])[:12])
    print("no decoding: top3=%.4f top1=%.4f" % hit(lab, lab.z.values)[:2], flush=True)
    folds = np.array_split(np.random.RandomState(0).permutation(np.array(sorted(lab.atco.unique()))), 6)
    lifts = []
    for f in folds:  # transitions learned only on the other controllers
        te = lab.atco.isin(f).values
        lifts.append((te,) + fit_lifts(lab[~te])[:2])
    for name, kw in [("uniform (v2 rule)", dict(use_type=False, use_same=False)),
                     ("+ same aircraft", dict(use_type=False, use_same=True)),
                     ("+ same aircraft + types", dict(use_type=True, use_same=True))]:
        for tau in [2.0, 3.0]:
            corr = np.zeros(len(lab))
            for te, ls_f, lt_f in lifts:
                corr[te] = decode(lab[te], "z", ls_f, lt_f, tau, **kw)
            for lam in [0.5, 1.0, 1.5]:
                print(f"{name:26s} tau={tau} lam={lam}: top3=%.4f top1=%.4f" % hit(lab, lab.z.values + lam * corr / tau)[:2], flush=True)
