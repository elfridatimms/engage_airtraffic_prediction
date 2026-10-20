"""Controller CV uses train only, preserving official validation as a separate diagnostic."""
import argparse,json
import numpy as np,pandas as pd
from src.training import ROOT,OUT,CONFIGS,fit,analysis
from src.validation import controller_folds
from src.metrics import topk,verified_score
from src.prediction import make_top3_submission
p=argparse.ArgumentParser(); p.add_argument('--models',nargs='+',default=list(CONFIGS)); p.add_argument('--iterations',type=int,default=700); p.add_argument('--select-only',action='store_true'); a=p.parse_args()
tr=pd.read_csv(ROOT.parent/'train.csv'); va=pd.read_csv(ROOT.parent/'validation.csv')
if not a.select_only:
 for name in a.models:
  if not (OUT/f'pred_{name}_official.csv').exists(): fit(name,tr,va,iterations=a.iterations)
  for f,ti,vi in controller_folds(tr):
   if not (OUT/f'pred_{name}_cv{f}.csv').exists(): fit(name,tr.iloc[ti],tr.iloc[vi],f'cv{f}',a.iterations)
exp=pd.read_csv(OUT/'experiments.csv'); cv=exp[exp.validation_scheme.str.startswith('cv')]
complete=[n for n in a.models if len(cv[cv.model.eq(n)])==3]
assert complete,'No completed CV models'
summary=cv[cv.model.isin(complete)].groupby('model').agg(mean_top3=('top3','mean'),std_top3=('top3','std'),worst_fold=('top3','min'),total_samples=('samples','sum')).sort_values('mean_top3',ascending=False)
summary.to_csv(OUT/'cv_summary.csv'); print(summary.to_string(),flush=True)
# Coarse blend of best classifier and ranker. Within-snapshot softmax places margins on comparable scale.
def normalize(df,s):
 z=pd.Series(s,index=df.index); z=np.exp(z-z.groupby(df.sample_id).transform('max')); return (z/z.groupby(df.sample_id).transform('sum')).to_numpy()
def scores(name,frame,scheme):
 pred=pd.read_csv(OUT/f'pred_{name}_{scheme}.csv'); merged=frame[['sample_id','row_index']].merge(pred,on=['sample_id','row_index'],how='left',validate='one_to_one'); assert merged.score.notna().all(); return merged.score.to_numpy()
best=summary.index[0]
structural=[n for n in summary.index if CONFIGS[n]['features'] in ['base','engineered']]
if structural and summary.loc[best,'mean_top3']-summary.loc[structural[0],'mean_top3'] < .003:
 best=structural[0]
chosen=[best]; weight=1.; best_metric=summary.loc[best,'mean_top3']
classifiers=[n for n in summary.index if CONFIGS[n]['kind']=='classifier']; rankers=[n for n in summary.index if CONFIGS[n]['kind']=='ranker']; blends=[]
if classifiers and rankers:
 c,r=classifiers[0],rankers[0]
 for w in [.25,.5,.75]:
  fold_scores=[]
  for f,_,vi in controller_folds(tr):
   d=tr.iloc[vi].reset_index(drop=True); s=w*normalize(d,scores(c,d,f'cv{f}'))+(1-w)*normalize(d,scores(r,d,f'cv{f}')); fold_scores.append(topk(d,s))
  mean=float(np.mean(fold_scores)); blends.append(dict(classifier=c,ranker=r,classifier_weight=w,mean_top3=mean,std_top3=float(np.std(fold_scores,ddof=1)),fold_scores=json.dumps(fold_scores)))
  # A modest improvement threshold avoids selecting a blend for a few lucky samples.
  if mean>best_metric+.002: chosen=[c,r]; weight=w; best_metric=mean
pd.DataFrame(blends).to_csv(OUT/'ensemble_experiments.csv',index=False)
config={'models':chosen,'classifier_weight':weight,'cv_mean_top3':best_metric,'selection':'3-fold controller-held-out CV on train; official validation diagnostic only','iterations':{n:int(cv[cv.model.eq(n)].trees.median()) for n in chosen}}
(OUT/'selected_config.json').write_text(json.dumps(config,indent=2))
s=scores(chosen[0],va,'official') if len(chosen)==1 else weight*normalize(va,scores(chosen[0],va,'official'))+(1-weight)*normalize(va,scores(chosen[1],va,'official'))
print('SELECTED',config,'official',verified_score(va,s,ROOT.parent/'scorer.py'),flush=True)
make_top3_submission(va,s).to_csv(OUT/'validation_predictions.csv',index=False); analysis(va,s,'selected')
pd.read_csv(OUT/'analysis_selected.csv').to_csv(OUT/'error_analysis.csv',index=False)
# Store selected OOF predictions for independent sample-weighted and controller-weighted reporting.
parts=[]
for f,_,vi in controller_folds(tr):
 d=tr.iloc[vi].reset_index(drop=True)
 s=scores(chosen[0],d,f'cv{f}') if len(chosen)==1 else weight*normalize(d,scores(chosen[0],d,f'cv{f}'))+(1-weight)*normalize(d,scores(chosen[1],d,f'cv{f}'))
 parts.append(d.assign(score=s,fold=f))
oof=pd.concat(parts).sort_values(['sample_id','row_index']); analysis(oof,oof.score.to_numpy(),'oof_selected')
oof[['sample_id','row_index','score','fold']].to_csv(OUT/'oof_predictions.csv',index=False)
# Final official diagnostic: fixed CV-chosen tree counts, no early stopping or
# checkpoint selection on official validation. Earlier official runs are exploratory.
from catboost import CatBoostClassifier,CatBoostRanker
from src.training import params,pool,predict,MODELS
fixed=[]
for i,name in enumerate(chosen):
 spec=CONFIGS[name]; model=(CatBoostClassifier if spec['kind']=='classifier' else CatBoostRanker)(**params(spec,config['iterations'][name]))
 model.fit(pool(tr.sort_values(['sample_id','row_index']),spec))
 model.save_model(str(MODELS/f'selected_validation_{i}.cbm'))
 fixed.append(predict(model,va,spec))
s=fixed[0] if len(fixed)==1 else weight*normalize(va,fixed[0])+(1-weight)*normalize(va,fixed[1])
fixed_score=verified_score(va,s,ROOT.parent/'scorer.py')
va[['sample_id','row_index']].assign(score=s).to_csv(OUT/'validation_row_scores.csv',index=False)
make_top3_submission(va,s).to_csv(OUT/'validation_predictions.csv',index=False)
analysis(va,s,'selected'); pd.read_csv(OUT/'analysis_selected.csv').to_csv(OUT/'error_analysis.csv',index=False)
(OUT/'official_fixed_result.json').write_text(json.dumps({'top1':topk(va,s,1),'top2':topk(va,s,2),'top3':fixed_score,'iterations':config['iterations'],'checkpoint_selected_on_official':False},indent=2))
print('FINAL FIXED-ITERATION OFFICIAL TOP3',fixed_score,flush=True)
