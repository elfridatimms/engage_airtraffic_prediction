import numpy as np, pandas as pd
from causal import PARAMS, hit3, past_scores, zscore, predict, train
from scen_cv import scen_cols, scen_oof, per_scen
lab = pd.read_parquet("causal_cache2.parquet"); cols = scen_cols(lab)
R = lambda n, o: print(n, "top3=%.4f top1=%.4f" % hit3(lab, o), per_scen(lab, o), flush=True)
noev = [c for c in cols if not (c.startswith("ts_gone") or c in ("is_new_task", "r_ts_gone_any"))]
R("no list_events", scen_oof(lab, noev))
REG = {**PARAMS, "num_leaves": 15, "min_data_in_leaf": 300, "lambda_l2": 20.0, "feature_fraction": 0.5}
R("regularised", scen_oof(lab, cols, REG, 600))
REG2 = {**PARAMS, "num_leaves": 7, "min_data_in_leaf": 500, "lambda_l2": 50.0, "feature_fraction": 0.5, "learning_rate": 0.05}
R("very regularised", scen_oof(lab, cols, REG2, 600))
# drop features that identify session / traffic level
gen = [c for c in cols if c not in ("seq_len", "n_flights", "a_n300", "a_n60")]
R("no load feats", scen_oof(lab, gen))
# stage-2 with past scores, scenario folds (stage-1 OOF inside training scenarios)
out = np.zeros(len(lab))
for s in sorted(lab.scenario.unique()):
    te = (lab.scenario == s).values
    trn, tst = lab[~te].copy(), lab[te].copy()
    trn["p1"] = scen_oof(trn, cols)
    tst["p1"] = predict(train(trn, cols), tst, cols)
    trn = pd.concat([trn, past_scores(trn, "p1")], axis=1); tst = pd.concat([tst, past_scores(tst, "p1")], axis=1)
    c2 = cols + [c for c in trn.columns if c.startswith("ps_")]
    out[te] = predict(train(trn, c2, rounds=300), tst, c2)
R("stage-2", out)
