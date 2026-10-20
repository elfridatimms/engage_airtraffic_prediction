import importlib.util
import numpy as np
from .prediction import make_top3_submission

def topk(df, scores, k=3):
    r = df.assign(score=np.asarray(scores)).sort_values(['sample_id','score','row_index'],ascending=[True,False,True])
    return float(r.groupby('sample_id').head(k).groupby('sample_id').target_index.max().mean())

def verified_score(df, scores, scorer_path):
    spec=importlib.util.spec_from_file_location('official_scorer',scorer_path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sol=df.loc[df.target_index.eq(1),['sample_id','row_index']].rename(columns={'row_index':'target'})
    score=topk(df,scores)
    assert score == mod.score(sol,make_top3_submission(df,scores))
    return score
