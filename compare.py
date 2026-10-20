"""Compare two validation submissions on the same samples.

    python compare.py val_submission.csv kolegica_val.csv
Both must be predictions for validation.csv from models trained on train.csv only.
"""
import sys

import numpy as np
import pandas as pd

from scorer import _ids


def hits(path, sol):
    sub = pd.read_csv(path, dtype={"prediction": str}).set_index("sample_id")["prediction"]
    missing = set(sol.sample_id) - set(sub.index)
    if missing:
        sys.exit(f"{path}: {len(missing)} validation samples missing")
    return np.array([int(t in _ids(sub[s])[:3]) for s, t in zip(sol.sample_id, sol.target)])


def main():
    a, b = sys.argv[1], sys.argv[2]
    lab = pd.read_csv("validation.csv", usecols=["sample_id", "row_index", "target_index", "session"])
    sol = lab[lab.target_index == 1].rename(columns={"row_index": "target"}).reset_index(drop=True)
    ha, hb = hits(a, sol), hits(b, sol)
    print(f"{a}: {ha.mean():.4f}   {b}: {hb.mean():.4f}   ({len(sol)} samples)")
    for s, g in sol.groupby("session"):
        print(f"  {s}: {ha[g.index].mean():.4f} vs {hb[g.index].mean():.4f}  ({len(g)})")
    print(f"only {a} right: {np.sum(ha > hb)}   only {b} right: {np.sum(hb > ha)}   "
          f"both wrong: {np.sum((ha == 0) & (hb == 0))}")
    # bootstrap over sessions-blocks of 20 consecutive samples (samples are not independent)
    blocks = np.arange(len(sol)) // 20
    nb = blocks.max() + 1
    d = pd.Series(ha - hb).groupby(blocks).sum().values
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, nb, nb)].sum() / len(sol) for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"difference {ha.mean() - hb.mean():+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
