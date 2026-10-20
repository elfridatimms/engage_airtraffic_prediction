"""Final real-time model: causal features, no scenario prior (best under strict controller-grouped CV).
    python final_causal.py val   -> train on train.csv, predictions/erna_realtime_val.csv
    python final_causal.py test  -> train on train+validation, submission_realtime.csv"""
import sys
mode = sys.argv[1] if len(sys.argv) > 1 else "test"
sys.argv = [sys.argv[0], "--noprior"]
import causal
from causal import causal_cols, predict, top3, train

df = causal.load()
cols = [c for c in causal_cols(df) if not c.startswith(("cp_", "ap_", "r_cp", "z_cp", "r_ap", "z_ap"))]
if mode == "val":
    trn, tst, out = df[df.split == 0], df[df.split == 1], "predictions/erna_realtime_val.csv"
else:
    trn, tst, out = df[df.split < 2], df[df.split == 2], "submission_realtime.csv"
tst = tst.reset_index(drop=True)
tst["score"] = predict(train(trn, cols, seeds=(0, 1, 2, 3, 4)), tst, cols)
top3(tst, "score").to_csv(out, index=False)
print(len(cols), "features, wrote", out)
