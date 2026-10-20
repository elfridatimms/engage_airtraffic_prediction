# AWARE task prediction (ENGAGE KTN 2026)

Each person works on their own branch (`erna`, ...). `main` holds only the shared tools.

## Fair comparison on validation.csv
1. Train **only on train.csv** and predict validation.csv.
2. Save as `predictions/<name>_val.csv` (`sample_id,prediction`) and push it on your branch.
3. Compare: `python compare.py predictions/erna_val.csv predictions/<name>_val.csv`

Data files (train/validation/test.csv) are not in git; copy them into the folder yourself.
