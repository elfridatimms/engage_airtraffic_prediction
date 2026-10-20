import numpy as np
import pandas as pd

EXCLUDE={'sample_id','row_index','target_index','session','atco','timestamp','task_name'}
TASKS=['entry_conflict_resolution','exit_conflict_resolution','non_conformance_resolution','conflict_resolution','entry_coordination','exit_coordination','assume_aircraft','transfer_aircraft','qos_improvement','entry_conditions','exit_conditions','zone_conflict','return_to_route','aircraft_request']
def task_type(df):
    return df.task_name.str.extract('^('+'|'.join(TASKS)+')_',expand=False).fillna('unknown')

def build_features(df, feature_set='base', keep=None):
    x=df[[c for c in df if c not in EXCLUDE]].copy()
    cats=['task_index','scenario']
    for c in cats: x[c]=x[c].astype(str)
    if feature_set != 'base':
        for c in ['time_to_sector','distance_to_sector','k_timeToExitPoint']:
            x['abs_'+c]=df[c].abs()
        x['log_task_age']=np.log1p(df.x_task_age.clip(lower=0))
        for c in ['time_to_sector','x_task_age','time_to_MD','k_timeToExitPoint','distance_to_zone']:
            g=df.groupby('sample_id')[c]
            x[c+'_relative_median']=df[c]-g.transform('median')
            x[c+'_percentile']=g.rank(pct=True)
            x[c+'_missing']=df[c].isna().astype(int)
        typ=task_type(df)
        for name,c in [('assume_aircraft','x_transfer_in'),('assume_aircraft','a_last_assume_control'),('transfer_aircraft','k_aircraftNeedsToBeTransferred'),('transfer_aircraft','k_transferToolActivation'),('entry_coordination','x_entry_coord_not_done'),('conflict_resolution','conflict_ind'),('zone_conflict','zone_conflict_ind'),('qos_improvement','qos_candidate'),('non_conformance_resolution','monitoring_rule_activated')]:
            x[name+'__'+c]=df[c].where(typ.eq(name),0)
        x['task_twin']=x.task_index+'_'+df.l_twin_AE.astype(str); cats+=['task_twin']
    if feature_set == 'identity':
        x['aircraft_identity']=df.task_name.str.replace('^('+'|'.join(TASKS)+')_','',regex=True)
        cats+=['aircraft_identity']
    if feature_set == 'causal':
        from .causal import build as causal_build
        x=pd.concat([x,causal_build(df)],axis=1)
    if feature_set == 'time':
        t=pd.to_timedelta(df.timestamp).dt.total_seconds()
        x['seconds_from_midnight']=t
        x['seconds_from_session_start']=t-t.groupby(df.session).transform('min')
    if keep is not None:
        # SHAP-pruned subset; a kept name must exist so a stale list cannot silently shrink the model.
        missing=[c for c in keep if c not in x.columns]
        assert not missing, f'unknown kept features: {missing}'
        x=x[[c for c in x.columns if c in set(keep)]]; cats=[c for c in cats if c in x.columns]
    for c in x.columns.difference(cats): x[c]=x[c].astype('float32')
    return x,cats
