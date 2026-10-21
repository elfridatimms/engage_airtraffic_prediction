"""v3 = v2 scores + joint decoding of the session's episodes with transitions learned from training labels.
    python apply_v3.py <tau> <lam> <use_type 0/1> <use_same 0/1>
writes val_submission_v3.csv (transitions from train.csv only) and submission_v3.csv (from train+validation)."""
import sys

import pandas as pd

from runfeat import load
from train import top3
from transitions import decode, fit_lifts

tau, lam = float(sys.argv[1]), float(sys.argv[2])
use_type, use_same = sys.argv[3] == "1", sys.argv[4] == "1"
df = load()
for mode, lab_mask in [("val", df.split == 0), ("test", df.split < 2)]:
    sc = pd.read_parquet(f"v2_{mode}_scores.parquet").sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    ls, lt, _, _ = fit_lifts(df[lab_mask].reset_index(drop=True))
    sc["score"] = sc.z + lam * decode(sc, "z", ls, lt, tau, use_type=use_type, use_same=use_same) / tau
    out = "val_submission_v3.csv" if mode == "val" else "submission_v3.csv"
    top3(sc, "score").to_csv(out, index=False)
    print("wrote", out)
