"""Leave-one-scenario-out CV: train on 2 scenarios, test on the unseen 3rd.
Scenario-specific priors (keyed by flight/task name or scenario) are not allowed: an unseen scenario has none."""
import re
import sys

import numpy as np
import pandas as pd

from causal import PARAMS, causal_cols, hit3, predict, train

SCEN_SPECIFIC = re.compile(r"^(r_|z_)?(cp_|ap_)")


def scen_cols(df):
    return [c for c in causal_cols(df) if not SCEN_SPECIFIC.match(c) and c != "scenario"]


def scen_oof(lab, cols, params=PARAMS, rounds=500, seeds=(0,)):
    out = np.zeros(len(lab))
    for s in sorted(lab.scenario.unique()):
        te = (lab.scenario == s).values
        out[te] = predict(train(lab[~te], cols, params=params, rounds=rounds, seeds=seeds), lab[te], cols)
    return out


def per_scen(lab, p):
    return {s: round(hit3(lab[lab.scenario == s], p[(lab.scenario == s).values])[0], 4)
            for s in sorted(lab.scenario.unique())}


if __name__ == "__main__":
    lab = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "causal_cache2.parquet")
    cols = scen_cols(lab)
    print(len(cols), "features (no scenario-specific priors)")
    for name, (p, r) in {"base": (PARAMS, 500),
                         "big": ({**PARAMS, "num_leaves": 63, "min_data_in_leaf": 50}, 600)}.items():
        o = scen_oof(lab, cols, p, r)
        print(name, "top3=%.4f top1=%.4f" % hit3(lab, o), per_scen(lab, o), flush=True)
