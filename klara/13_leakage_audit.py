"""Audit of future-derived features: how much of the label do they carry?

A snapshot is a live 3-second picture, but the supplied files contain whole
sessions, so later snapshots are readable at training and at prediction time.
Features built from them are therefore computable for the test set, yet they
cannot be computed by a system running in an operations room. This script
measures how much of the target such a feature recovers on its own.
"""
import numpy as np, pandas as pd
from src.training import ROOT, OUT

df = pd.read_csv(ROOT.parent / 'validation.csv')
ts = pd.to_timedelta(df.timestamp).dt.total_seconds()
tn = df.task_name
# Conflict tasks (indices 0, 1, 6) name two aircraft; the first is the subject.
flight = np.where(df.task_index.isin([0, 1, 6]), tn.str.rsplit('_', n=2).str[-2], tn.str.split('_').str[-1])
st = df.session + '|' + df.task_name
g = ts.groupby(st)
df['f_t_until_task_end'] = g.transform('max') - ts        # future: task disappears once handled
df['c_t_since_task_start'] = ts - g.transform('min')      # past: first appearance precedes now

def rule_top3(col, ascending):
    r = df.sort_values(['sample_id', col, 'row_index'], ascending=[True, ascending, True])
    return float(r.groupby('sample_id').head(3).groupby('sample_id').target_index.max().mean())

rows = [
    dict(feature='f_t_until_task_end', information='future', rule='ascending', top3=rule_top3('f_t_until_task_end', True)),
    dict(feature='c_t_since_task_start', information='past', rule='descending', top3=rule_top3('c_t_since_task_start', False)),
    dict(feature='x_task_age', information='past', rule='descending', top3=rule_top3('x_task_age', False)),
    dict(feature='time_to_sector', information='past', rule='ascending', top3=rule_top3('time_to_sector', True)),
]
out = pd.DataFrame(rows)
t, o = df[df.target_index.eq(1)], df[df.target_index.eq(0)]
stats = pd.DataFrame([
    dict(group='target row', median_s=t.f_t_until_task_end.median(), mean_s=t.f_t_until_task_end.mean(),
         fraction_within_3s=t.f_t_until_task_end.le(3).mean()),
    dict(group='other candidates', median_s=o.f_t_until_task_end.median(), mean_s=o.f_t_until_task_end.mean(),
         fraction_within_3s=o.f_t_until_task_end.le(3).mean()),
])
out.to_csv(OUT / 'leakage_audit.csv', index=False)
stats.to_csv(OUT / 'leakage_audit_distribution.csv', index=False)
print('Single feature used as the entire model, official validation (1252 snapshots):')
print(out.to_string(index=False))
print('\nTime until the task leaves the session:')
print(stats.round(3).to_string(index=False))
print('\nA future-derived feature alone outranks the full trained pipeline (0.6142),'
      '\nwhich is why it is excluded rather than adopted.')
