from pathlib import Path
import json,time,os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,CatBoostRanker,Pool
from .features import build_features,task_type
from .metrics import topk,verified_score
from .prediction import make_top3_submission
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; MODELS=ROOT/'models'
CONFIGS={
 'classifier_base':dict(kind='classifier',features='base'),
 'ranker_base':dict(kind='ranker',features='base'),
 'classifier_engineered':dict(kind='classifier',features='engineered'),
 'classifier_weighted':dict(kind='classifier',features='engineered',scale_pos_weight=5),
 'ranker_engineered':dict(kind='ranker',features='engineered'),
 'classifier_identity':dict(kind='classifier',features='identity'),
 'classifier_time':dict(kind='classifier',features='time'),
 'classifier_depth8':dict(kind='classifier',features='engineered',depth=8),
 # Past-only session-dynamics features; operationally computable in a live snapshot.
 'classifier_causal':dict(kind='classifier',features='causal'),
 'ranker_causal':dict(kind='ranker',features='causal'),
 'classifier_subsample':dict(kind='classifier',features='engineered',rsm=.7,subsample=.8),
 'ranker_subsample':dict(kind='ranker',features='base',rsm=.7,subsample=.8),
 'classifier_engineered_s7':dict(kind='classifier',features='engineered',seed=7),
 'classifier_engineered_s13':dict(kind='classifier',features='engineered',seed=13),
}

def shap_keep(n):
 """Top-n features by mean within-snapshot SHAP spread: the part of each
 contribution that can actually reorder candidates inside a snapshot."""
 s=pd.read_csv(OUT/'shap_feature_summary.csv').sort_values('within_snapshot_spread',ascending=False)
 return list(s.feature.head(n))

if (OUT/'shap_feature_summary.csv').exists():
 for _n in [54,67,89]:
  CONFIGS[f'classifier_shap{_n}']=dict(kind='classifier',features='engineered',keep=shap_keep(_n))
 CONFIGS['classifier_shap_usable']=dict(kind='classifier',features='engineered',keep=shap_keep(10**6) and list(pd.read_csv(OUT/'shap_feature_summary.csv').query('usable').feature))

def params(cfg,iterations=700):
 p=dict(iterations=iterations,depth=cfg.get('depth',6),learning_rate=.06,l2_leaf_reg=5,random_seed=cfg.get('seed',42),
        thread_count=min(8,os.cpu_count() or 4),task_type='CPU',allow_writing_files=False,verbose=200)
 # Column/row subsampling: regularisation aimed at generalising to an unseen controller.
 if 'rsm' in cfg: p['rsm']=cfg['rsm']
 if 'subsample' in cfg: p.update(subsample=cfg['subsample'],bootstrap_type='Bernoulli')
 if cfg['kind']=='ranker': p.update(loss_function='YetiRank',eval_metric='PrecisionAt:top=3')
 else: p.update(loss_function='Logloss',eval_metric='Logloss')
 if 'scale_pos_weight' in cfg: p['scale_pos_weight']=cfg['scale_pos_weight']
 return p

def pool(df,cfg):
 x,cats=build_features(df,cfg['features'],cfg.get('keep'))
 return Pool(x,label=df.target_index if 'target_index' in df else None,
             cat_features=cats,group_id=df.sample_id.to_numpy() if cfg['kind']=='ranker' else None)

def predict(model,df,cfg):
 p=pool(df,cfg)
 return model.predict(p,prediction_type='RawFormulaVal') if cfg['kind']=='classifier' else model.predict(p)

def fit(name,tr,va,scheme='official',iterations=700):
 cfg=CONFIGS[name]; tr=tr.sort_values(['sample_id','row_index']).reset_index(drop=True); va=va.sort_values(['sample_id','row_index']).reset_index(drop=True)
 assert not set(tr.session)&set(va.session)
 assert not set(tr.atco)&set(va.atco)
 p=params(cfg,iterations); model=(CatBoostClassifier if cfg['kind']=='classifier' else CatBoostRanker)(**p)
 t=time.time(); train_pool=pool(tr,cfg); val_pool=pool(va,cfg)
 model.fit(train_pool,eval_set=val_pool,early_stopping_rounds=100,use_best_model=False)
 # Select checkpoint with actual top-3, including classifier whose stopping monitor is logloss.
 checkpoints=sorted(set(list(range(50,model.tree_count_+1,50))+[model.tree_count_]))
 best=(-1,0,None)
 for n in checkpoints:
  s=model.predict(val_pool,ntree_end=n,prediction_type='RawFormulaVal') if cfg['kind']=='classifier' else model.predict(val_pool,ntree_end=n)
  metric=topk(va,s)
  if metric>best[0]: best=(metric,n,s)
 score,n,s=best; model.shrink(n); model.save_model(str(MODELS/f'{name}_{scheme}.cbm'))
 verified_score(va,s,ROOT.parent/'scorer.py')
 pred=va[['sample_id','row_index']].assign(score=s); pred.to_csv(OUT/f'pred_{name}_{scheme}.csv',index=False)
 row=dict(model=name,model_type=cfg['kind'],feature_set=cfg['features'],parameters=json.dumps(p),validation_scheme=scheme,top3=score,top1=topk(va,s,1),top2=topk(va,s,2),trees=n,held_out_sessions=' '.join(sorted(va.session.unique())),held_out_atcos=' '.join(map(str,sorted(va.atco.unique()))),samples=va.sample_id.nunique(),seconds=round(time.time()-t,1),notes='Checkpoint selected on this validation fold; model-selection estimate, not untouched test performance.')
 path=OUT/'experiments.csv'; old=pd.read_csv(path) if path.exists() else pd.DataFrame(); old=pd.concat([old,pd.DataFrame([row])],ignore_index=True); old.drop_duplicates(['model','validation_scheme'],keep='last').to_csv(path,index=False)
 print('RESULT',json.dumps(row),flush=True)
 analysis(va,s,f'{name}_{scheme}')
 if scheme=='official':
  imp=model.get_feature_importance(train_pool,type='PredictionValuesChange'); pd.DataFrame({'feature':model.feature_names_,'importance':imp}).sort_values('importance',ascending=False).to_csv(OUT/f'importance_{name}.csv',index=False)
 return row

def analysis(df,s,prefix):
 r=df.assign(score=s,task_type=task_type(df)).sort_values(['sample_id','score','row_index'],ascending=[True,False,True]); r['rank']=r.groupby('sample_id').cumcount()+1
 truth=r[r.target_index.eq(1)].copy(); truth['seq_len_bin']=pd.cut(truth.seq_len,[0,20,30,40,50,100]).astype(str)
 for k in [1,2,3]: truth[f'hit{k}']=truth['rank'].le(k).astype(int)
 truth.to_csv(OUT/f'errors_{prefix}.csv',index=False)
 tables=[]
 for c in ['task_index','task_type','scenario','atco','session','seq_len_bin','l_twin_AE','conflict_ind','zone_conflict_ind']:
  tab=truth.groupby(c).agg(samples=('sample_id','size'),top1=('hit1','mean'),top2=('hit2','mean'),top3=('hit3','mean'),rank4=('rank',lambda x:x.eq(4).sum())).reset_index().rename(columns={c:'value'}); tab.insert(0,'dimension',c); tables.append(tab)
 pd.concat(tables).to_csv(OUT/f'analysis_{prefix}.csv',index=False)
 misses=truth[truth['rank'].gt(3)][['sample_id','task_type','task_name','rank']].rename(columns={'task_type':'true_type','task_name':'true_task_name','rank':'true_rank'})
 above=r.merge(misses,on='sample_id'); above=above[above['rank']<above.true_rank]
 above.groupby(['true_type','task_type']).agg(count=('sample_id','size')).reset_index().to_csv(OUT/f'confusion_{prefix}.csv',index=False)
