from sklearn.model_selection import GroupKFold

def controller_folds(df,n_splits=3):
    for fold,(tr,va) in enumerate(GroupKFold(n_splits=n_splits).split(df,groups=df.atco)):
        assert not set(df.iloc[tr].atco)&set(df.iloc[va].atco)
        assert not set(df.iloc[tr].session)&set(df.iloc[va].session)
        yield fold,tr,va

def check_data(df,labelled=True):
    assert not df.duplicated(['sample_id','row_index']).any()
    g=df.groupby('sample_id')
    for c in ['session','atco','scenario','timestamp','seq_len']:
        assert g[c].nunique().eq(1).all()
    assert g.size().eq(g.seq_len.first()).all()
    assert g.size().min()>=3
    if labelled:
        assert df.target_index.isin([0,1]).all()
        assert g.target_index.sum().eq(1).all()
