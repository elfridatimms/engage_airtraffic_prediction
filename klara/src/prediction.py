import numpy as np
import pandas as pd

def make_top3_submission(df, scores):
    scores = np.asarray(scores).reshape(-1)
    assert len(scores) == len(df) and np.isfinite(scores).all()
    assert not df.duplicated(['sample_id', 'row_index']).any()
    ranked = df[['sample_id','row_index']].assign(score=scores).sort_values(
        ['sample_id','score','row_index'], ascending=[True,False,True], kind='stable')
    sub = ranked.groupby('sample_id', sort=False).head(3).groupby('sample_id').row_index.apply(
        lambda x: ' '.join(str(int(v)) for v in x)).rename('prediction').reset_index()
    validate_submission(df, sub)
    return sub

def validate_submission(df, sub):
    assert list(sub.columns) == ['sample_id','prediction']
    assert sub.sample_id.is_unique and set(sub.sample_id) == set(df.sample_id)
    assert sub.prediction.notna().all()
    allowed = df.groupby('sample_id').row_index.agg(set)
    for sid, pred in sub.itertuples(index=False):
        tokens = str(pred).split()
        assert all(t.lstrip('-').isdigit() for t in tokens)
        ids = list(map(int, tokens))
        assert len(ids) == len(set(ids)) == min(3,len(allowed[sid]))
        assert set(ids) <= allowed[sid]
