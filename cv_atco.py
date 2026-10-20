"""Strict controller-grouped CV (as in the challenge slides: split by session AND controller).
Crowd priors are recomputed per fold from the training controllers only."""
import sys
sys.argv = [sys.argv[0], "--noprior"]  # build base frame without priors
import numpy as np
import pandas as pd
import causal
from causal import PARAMS, aligned_prior, causal_cols, crowd_prior, hit3, predict, train

base = causal.load()
base = base[base.split < 2].reset_index(drop=True)
atcos = np.array(sorted(base.atco.unique()))
folds = np.array_split(np.random.RandomState(0).permutation(atcos), 6)
CFG = {"no_prior": None,
       "base": (PARAMS, 500),
       "big": ({**PARAMS, "num_leaves": 63, "min_data_in_leaf": 50}, 600)}
oofs = {k: np.zeros(len(base)) for k in CFG}
for f in folds:
    te = base.atco.isin(f).values
    d = base.copy()
    d.loc[te, "split"] = 9  # held-out controllers' labels never feed the priors
    d = aligned_prior(crowd_prior(d, max_split=1), max_split=1)
    cols_p = causal_cols(d)
    cols_n = [c for c in cols_p if not c.startswith(("cp_", "ap_", "r_cp", "z_cp", "r_ap", "z_ap"))]
    for k, cfg in CFG.items():
        if cfg is None:
            oofs[k][te] = predict(train(d[~te], cols_n), d[te], cols_n)
        else:
            oofs[k][te] = predict(train(d[~te], cols_p, params=cfg[0], rounds=cfg[1]), d[te], cols_p)
    print("fold", list(f), "done", flush=True)
for k, o in oofs.items():
    h = {a: round(hit3(base[base.atco == a], o[(base.atco == a).values])[0], 3) for a in atcos}
    print(k, "top3=%.4f top1=%.4f" % hit3(base, o), "| ctrl020=%.3f" % h[20], flush=True)
pd.DataFrame(oofs).assign(sample_id=base.sample_id, row_index=base.row_index).to_parquet("cv_atco_oof.parquet")
