"""Stage-2: add neighbouring-snapshot stage-1 scores of the same task, re-rank."""
import numpy as np
import pandas as pd


def neighbour_feats(df, col="p1"):
    d = df[["sample_id", "st", "snap_k", col]].copy()
    sg = d.groupby("sample_id")[col]
    d["z"] = (d[col] - sg.transform("mean")) / (sg.transform("std") + 1e-9)
    d["rk"] = sg.rank(ascending=False, method="min")
    d["gap"] = d[col] - sg.transform("max")
    lut_z = dict(zip(zip(d.st, d.snap_k), d.z))
    lut_r = dict(zip(zip(d.st, d.snap_k), d.rk))
    out = {"s2_z": d.z.values, "s2_rk": d.rk.values, "s2_gap": d.gap.values}
    keys = list(zip(d.st, d.snap_k))
    for off in [-3, -2, -1, 1, 2, 3]:
        out[f"s2_z{off}"] = np.array([lut_z.get((s, k + off), np.nan) for s, k in keys])
        out[f"s2_r{off}"] = np.array([lut_r.get((s, k + off), np.nan) for s, k in keys])
    o = pd.DataFrame(out, index=df.index)
    o["s2_zmean"] = o[[f"s2_z{i}" for i in [-2, -1, 1, 2]]].mean(axis=1)
    o["s2_top1_prev"] = (o["s2_r-1"] == 1).astype(int)
    o["s2_top1_next"] = (o["s2_r1"] == 1).astype(int)
    return o
