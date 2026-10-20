import numpy as np, pandas as pd
from exp import oof, report, PARAMS
from features import feature_cols
from stage2 import neighbour_feats
df = pd.read_parquet("feats2.parquet")
lab = df[df.split < 2].sort_values(["sample_id", "row_index"]).reset_index(drop=True)
cols = feature_cols(lab)
lab["p1"] = oof(lab, cols)
report(lab, lab.p1, "stage1")
nf = neighbour_feats(lab)
lab = pd.concat([lab, nf], axis=1)
c2 = ["task_index"] + list(nf.columns)
report(lab, oof(lab, c2, rounds=200), "stage2 small")
report(lab, oof(lab, cols + list(nf.columns), rounds=300), "stage2 full")
lab[["sample_id","row_index","p1"]].to_parquet("oof_p1.parquet")
