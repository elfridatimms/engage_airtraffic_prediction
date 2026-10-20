import sys
from causal import load, causal_cols, oof, hit3
df = load(); lab = df[df.split < 2].reset_index(drop=True); cols = causal_cols(lab)
print(len(cols), "features")
print("stage-1 top3=%.4f top1=%.4f" % hit3(lab, oof(lab, cols, seeds=(0, 1))))
