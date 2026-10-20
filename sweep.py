import numpy as np, pandas as pd
from causal import causal_cols, oof, hit3, PARAMS, zscore
lab = pd.read_parquet("causal_cache.parquet"); cols = causal_cols(lab)
B = {k: v for k, v in PARAMS.items() if k not in ("eval_at", "lambdarank_truncation_level")}
C = {
 "base": (PARAMS, 500),
 "big": ({**PARAMS, "num_leaves": 63, "min_data_in_leaf": 50}, 600),
 "small": ({**PARAMS, "num_leaves": 15, "min_data_in_leaf": 200, "learning_rate": 0.05}, 500),
 "ff04_extra": ({**PARAMS, "feature_fraction": 0.4, "extra_trees": True}, 900),
 "trunc30": ({**PARAMS, "lambdarank_truncation_level": 30}, 500),
 "xendcg": ({**PARAMS, "objective": "rank_xendcg"}, 500),
 "binary": ({**B, "objective": "binary", "metric": "auc", "scale_pos_weight": 3.0}, 500),
}
P = {}
for k, (p, r) in C.items():
    P[k] = zscore(oof(lab, cols, params=p, rounds=r), lab.sample_id.values)
    print(k, "top3=%.4f top1=%.4f" % hit3(lab, P[k]), flush=True)
pd.DataFrame(P).assign(sample_id=lab.sample_id.values, row_index=lab.row_index.values).to_parquet("sweep_oof.parquet")
print("ENS all", "top3=%.4f top1=%.4f" % hit3(lab, sum(P.values())))
best = sorted(P, key=lambda k: -hit3(lab, P[k])[0])
for n in [2, 3, 4, 5]:
    print(f"ENS top{n}", best[:n], "top3=%.4f" % hit3(lab, sum(P[k] for k in best[:n]))[0])
