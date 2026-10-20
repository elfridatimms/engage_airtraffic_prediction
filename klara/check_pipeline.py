"""Small adversarial checks for submission ordering, alignment, and official scoring parity."""
from pathlib import Path
import numpy as np,pandas as pd
from src.metrics import verified_score,topk
from src.prediction import make_top3_submission,validate_submission
r=Path(__file__).resolve().parent
x=pd.DataFrame({'sample_id':[9,3,9,3,9,3,9,3],'row_index':[7,8,2,5,4,3,1,1],'target_index':[0,0,1,0,0,0,0,1]})
s=np.array([1,0,1,0,1,0,1,1.])
sub=make_top3_submission(x,s)
assert sub.set_index('sample_id').loc[9,'prediction']=='1 2 4'
assert verified_score(x,s,r.parent/'scorer.py')==1
assert topk(x,s,1)==.5
for bad in [sub.assign(prediction='1 1 1'),sub.assign(prediction='999 2 3'),pd.concat([sub,sub.iloc[:1]])]:
 try: validate_submission(x,bad)
 except AssertionError: pass
 else: raise AssertionError('Invalid submission accepted')
print('Submission/metric tests passed')
