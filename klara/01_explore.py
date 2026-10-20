from pathlib import Path
import json, hashlib
import pandas as pd
from src.validation import check_data
from src.features import task_type
ROOT=Path(__file__).resolve().parent; DATA=ROOT.parent; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
summary={}; frames={}
for name in ['train','validation','test']:
    d=pd.read_csv(DATA/f'{name}.csv'); frames[name]=d; check_data(d,name!='test')
    summary[name]={'shape':d.shape,'samples':d.sample_id.nunique(),'sessions':sorted(d.session.unique()),'controllers':sorted(map(int,d.atco.unique())),'scenarios':sorted(map(int,d.scenario.unique())),'positive_count':int(d.target_index.sum()) if 'target_index' in d else None,'contiguous_groups':not d.sample_id.loc[d.sample_id.ne(d.sample_id.shift())].duplicated().any()}
    pd.DataFrame({'dtype':d.dtypes.astype(str),'missing_count':d.isna().sum(),'missing_fraction':d.isna().mean(),'unique':d.nunique()}).to_csv(OUT/f'{name}_columns.csv')
    d.groupby(['task_index']).size().to_csv(OUT/f'{name}_task_distribution.csv')
    if name!='test': d[d.target_index.eq(1)].groupby(['task_index']).size().to_csv(OUT/f'{name}_target_distribution.csv')
    d.groupby(['session','atco','scenario']).agg(rows=('sample_id','size'),samples=('sample_id','nunique')).to_csv(OUT/f'{name}_sessions.csv')
    print(name,json.dumps(summary[name],indent=2))
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert not set(frames[a].session)&set(frames[b].session)
    assert not set(frames[a].sample_id)&set(frames[b].sample_id)
(OUT/'dataset_summary.json').write_text(json.dumps(summary,indent=2))
d=frames['train']; d[['task_index']].assign(task_type=task_type(d)).drop_duplicates().to_csv(OUT/'task_mapping.csv',index=False)
(OUT/'original_checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.iterdir() if p.is_file()},indent=2))
