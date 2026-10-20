import argparse,json,hashlib
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,CatBoostRanker
from src.training import ROOT,OUT,MODELS,CONFIGS,params,pool,predict
from src.prediction import make_top3_submission,validate_submission
p=argparse.ArgumentParser(); p.add_argument('--predict-only',action='store_true'); a=p.parse_args()
cfg=json.loads((OUT/'selected_config.json').read_text()); tr=pd.concat([pd.read_csv(ROOT.parent/'train.csv'),pd.read_csv(ROOT.parent/'validation.csv')]).sort_values(['sample_id','row_index']).reset_index(drop=True); te=pd.read_csv(ROOT.parent/'test.csv').sort_values(['sample_id','row_index']).reset_index(drop=True)
preds=[]
for i,name in enumerate(cfg['models']):
 spec=CONFIGS[name]; model=(CatBoostClassifier if spec['kind']=='classifier' else CatBoostRanker)(**params(spec,cfg['iterations'][name])); path=MODELS/('best_model.cbm' if i==0 else 'best_ranker.cbm')
 if a.predict_only: model.load_model(str(path))
 else:
  tp=pool(tr,spec); model.fit(tp); model.save_model(str(path))
  imp=pd.DataFrame({'feature':model.feature_names_,'importance':model.get_feature_importance(tp,type='PredictionValuesChange')}).sort_values('importance',ascending=False)
  imp.to_csv(OUT/('feature_importance.csv' if i==0 else 'ranker_feature_importance.csv'),index=False)
 preds.append(predict(model,te,spec))
 make_top3_submission(te,preds[-1]).to_csv(OUT/f'submission_{name}.csv',index=False)
if len(preds)==1: s=preds[0]
else:
 norm=[]
 for v in preds:
  z=pd.Series(v); z=np.exp(z-z.groupby(te.sample_id).transform('max')); norm.append((z/z.groupby(te.sample_id).transform('sum')).to_numpy())
 w=cfg['classifier_weight']; s=w*norm[0]+(1-w)*norm[1]
sub=make_top3_submission(te,s); template=pd.read_csv(ROOT.parent/'sample_submission.csv'); sub=template[['sample_id']].merge(sub,on='sample_id',validate='one_to_one'); validate_submission(te,sub); assert list(sub.columns)==list(template.columns); assert len(sub)==te.sample_id.nunique()==1863
sub.to_csv(ROOT/'submission.csv',index=False); reread=pd.read_csv(ROOT/'submission.csv',dtype={'prediction':str}); validate_submission(te,reread)
te[['sample_id','row_index']].assign(score=s).to_csv(OUT/'test_scores.csv',index=False)
checks=json.loads((OUT/'original_checksums.json').read_text()); assert all(hashlib.sha256((ROOT.parent/n).read_bytes()).hexdigest()==v for n,v in checks.items())
(OUT/'submission_validation.json').write_text(json.dumps({'valid':True,'samples':len(sub),'rows':len(te),'original_files_unchanged':True,'config':cfg},indent=2)); print('VALID SUBMISSION:',len(sub),'samples',flush=True)
