import numpy as np, pandas as pd
from exp import oof, report, PARAMS
from features import feature_cols
df = pd.read_parquet("feats2.parquet")
lab = df[df.split < 2].sort_values(["sample_id", "row_index"]).reset_index(drop=True)
cols = feature_cols(lab)
def z(p):
    s = pd.Series(p).groupby(lab.sample_id.values)
    return ((pd.Series(p) - s.transform("mean")) / (s.transform("std") + 1e-9)).values
P = {}
P["rank"] = oof(lab, cols)
P["rank_l15"] = oof(lab, cols, {**PARAMS, "num_leaves": 15, "min_data_in_leaf": 100}, 600)
P["rank_l63"] = oof(lab, cols, {**PARAMS, "num_leaves": 63, "min_data_in_leaf": 30}, 300)
P["xendcg"] = oof(lab, cols, {**PARAMS, "objective": "rank_xendcg"}, 400)
P["binary"] = oof(lab, cols, {**{k: v for k, v in PARAMS.items() if k not in ("eval_at","lambdarank_truncation_level")}, "objective": "binary", "metric": "auc"}, 400)
for k, v in P.items(): report(lab, v, k)
report(lab, sum(z(v) for v in P.values()), "ENSEMBLE all")
report(lab, z(P["rank"]) + z(P["binary"]) + z(P["xendcg"]), "ENS rank+bin+xe")
pd.DataFrame(P).to_parquet("oof_models.parquet")
