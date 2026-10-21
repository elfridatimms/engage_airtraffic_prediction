"""Exploit 'the next episode is a different task': across a gap the label changes 99.8% of the time.
Run-level chain decoding on the controller-grouped OOF scores of the run-feature model (pool_oof.parquet)."""
import numpy as np
import pandas as pd

from runfeat import hit, load


def run_tables(d, zcol):
    """per session: ordered runs, each a dict task -> pooled score (mean z over the run's snapshots)"""
    pz = d.groupby(["session", "run_rid", "task_name"])[zcol].mean()
    out = {}
    for (s, rid), g in pz.groupby(level=[0, 1]):
        out.setdefault(s, []).append((rid, dict(zip(g.index.get_level_values(2), g.values))))
    return out


def softmax(dct, tau):
    k = list(dct)
    v = np.array([dct[x] for x in k]) * tau
    v = np.exp(v - v.max())
    return dict(zip(k, v / v.sum()))


def chain_adjust(d, zcol, tau, mode="fb"):
    """returns log-correction per row: log(posterior with constraint / posterior without), per (run, task)"""
    corr = {}
    for s, runs in run_tables(d, zcol).items():
        q = [softmax(t, tau) for _, t in runs]
        n = len(q)
        if mode == "penalty":  # simple: penalise tasks likely to be the neighbouring runs' answer
            for k in range(n):
                for x in q[k]:
                    p = (q[k - 1].get(x, 0) if k > 0 else 0) + (q[k + 1].get(x, 0) if k < n - 1 else 0)
                    corr[(runs[k][0], x)] = np.log(max(1 - p, 1e-6))
            continue
        alpha = []  # forward: label_k != label_{k-1}
        for k in range(n):
            a = {x: q[k][x] * (1 - (alpha[k - 1].get(x, 0) if k else 0)) for x in q[k]}
            z = sum(a.values())
            alpha.append({x: v / z for x, v in a.items()})
        beta = [None] * n
        beta[n - 1] = {x: 1.0 for x in q[n - 1]}
        for k in range(n - 2, -1, -1):
            w = {y: q[k + 1][y] * beta[k + 1][y] for y in q[k + 1]}
            z = sum(w.values())
            beta[k] = {x: 1 - w.get(x, 0) / z for x in q[k]}
        for k in range(n):
            post = {x: alpha[k][x] * beta[k][x] for x in q[k]}
            z = sum(post.values())
            for x in q[k]:
                corr[(runs[k][0], x)] = np.log(max(post[x] / z, 1e-9) / max(q[k][x], 1e-9))
    return np.array([corr[(r, t)] for r, t in zip(d.run_rid, d.task_name)])


if __name__ == "__main__":
    df = load()
    lab = df[df.split < 2].reset_index(drop=True)
    lab = lab.merge(pd.read_parquet("pool_oof.parquet")[["sample_id", "row_index", "z"]], on=["sample_id", "row_index"])
    lab = lab.sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    t3, t1, r = hit(lab, lab.z.values)
    print("no chain:        top3=%.4f top1=%.4f" % (t3, t1))
    # diagnostic: how often is the previous run's true answer wasting a slot in the current top-3?
    tl = lab[lab.target_index == 1].drop_duplicates("run_rid").set_index("run_rid").task_name
    lab["prev_ans"] = (lab.run_rid - 1).map(tl)
    lab["same_sess_prev"] = (lab.run_rid - 1).map(lab.drop_duplicates("run_rid").set_index("run_rid").session) == lab.session
    m = (lab.task_name == lab.prev_ans) & lab.same_sess_prev & (r <= 3)
    print("samples whose top-3 contains the previous episode's answer: %.3f" % (lab.loc[m, "sample_id"].nunique() / lab.sample_id.nunique()))
    for mode in ["penalty", "fb"]:
        for tau in [1, 2, 3, 5]:
            c = chain_adjust(lab, "z", tau, mode)
            for lam in [0.5, 1.0, 2.0]:
                print(f"{mode:7s} tau={tau} lam={lam}: top3=%.4f top1=%.4f" % hit(lab, lab.z.values + lam * c / tau)[:2], flush=True)
