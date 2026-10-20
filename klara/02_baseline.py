from pathlib import Path
import numpy as np,pandas as pd
from src.metrics import verified_score
from src.prediction import make_top3_submission
R=Path(__file__).resolve().parent; D=R.parent
rows=[]
for split in ['train','validation']:
 d=pd.read_csv(D/f'{split}.csv')
 for name,s in [('random',np.random.default_rng(42).random(len(d))),('nearest_sector',-d.time_to_sector.abs().fillna(float('inf')))]:
  score=verified_score(d,s,D/'scorer.py'); rows.append(dict(split=split,model=name,top3=score)); print(rows[-1])
 print('Expected random', (3/d.groupby('sample_id').size()).mean())
pd.DataFrame(rows).to_csv(R/'outputs/baselines.csv',index=False)
t=pd.read_csv(D/'test.csv'); s=make_top3_submission(t,-t.time_to_sector.abs().fillna(1e20)); provided=pd.read_csv(D/'sample_submission.csv'); match=s.merge(provided,on='sample_id'); print('Nearest rule exact prediction match',match.prediction_x.eq(match.prediction_y).mean())
