"""Operationally valid (past-only) features.

Every column here is computable inside a live 3-second snapshot: it uses the
current snapshot, earlier snapshots of the same session, and the supplied
"time since" columns. Nothing reads a later snapshot. The excluded
counterparts -- time until the task disappears, time until the next controller
input, presence in the next snapshot, time until the next state change -- are
strong predictors only because the task disappears after the controller acts
on it, so they encode the label through the future and cannot be computed at
run time.
"""
import numpy as np
import pandas as pd

# Conflict tasks name two aircraft; the first one is the subject of the task.
CONFLICT_TASKS = {0, 1, 6}
# Flight-state columns whose last transition marks recent operational change.
CHANGE_COLS = ['x_isc_coordinated', 'x_entry_coord_not_done', 'x_transfer_in',
               'k_out_COORDINATED', 'is_assumed', 'conflict_ind', 'zone_conflict_ind',
               'qos_candidate', 'monitoring_rule_activated']
COORD_COLS = ['x_isc_coordinated', 'x_entry_coord_not_done', 'k_out_COORDINATED']


def flight_id(df):
    """Aircraft the task refers to. Used only to group candidates, never as an identity category."""
    tn = df.task_name
    return np.where(df.task_index.isin(CONFLICT_TASKS),
                    tn.str.rsplit('_', n=2).str[-2], tn.str.split('_').str[-1])


def build(df):
    out = pd.DataFrame(index=df.index)
    ts = pd.to_timedelta(df.timestamp).dt.total_seconds()
    flight = pd.Series(flight_id(df), index=df.index)
    st = df.session + '|' + df.task_name
    sf = df.session + '|' + flight

    # --- task persistence: the first appearance of a live task is always in the past ---
    out['c_t_since_task_start'] = ts - ts.groupby(st).transform('min')
    order = np.lexsort((df.row_index.to_numpy(), ts.to_numpy()))
    seen = pd.Series(1, index=df.index[order]).groupby(st.iloc[order]).cumsum()
    out['c_task_snapshots_so_far'] = seen.reindex(df.index)

    # --- snapshot timeline of the session ---
    snap = pd.DataFrame({'sample_id': df.sample_id, 'session': df.session, 'ts': ts}) \
        .drop_duplicates('sample_id').sort_values(['session', 'ts'])
    snap['k'] = snap.groupby('session').cumcount()
    snap['gap_prev'] = snap.groupby('session').ts.diff()
    k = df.sample_id.map(snap.set_index('sample_id').k)
    out['c_gap_prev'] = df.sample_id.map(snap.set_index('sample_id').gap_prev)
    key = set(zip(st.to_numpy(), k.to_numpy()))
    out['c_in_prev'] = [int((s, kk - 1) in key) for s, kk in zip(st.to_numpy(), k.to_numpy())]

    # --- time since the last transition of each flight state (past snapshots only) ---
    fl = pd.DataFrame({'sf': sf, 'sample_id': df.sample_id, 'ts': ts}) \
        .join(df[CHANGE_COLS]).drop_duplicates(['sf', 'sample_id']).sort_values(['sf', 'ts'])
    tp = {}
    for c in CHANGE_COLS:
        v = fl[c].fillna(-999)
        changed = v.ne(v.groupby(fl.sf).shift()) & fl.sf.eq(fl.sf.shift())
        # ffill carries only earlier rows forward, so no future value can reach this row.
        last = fl.ts.where(changed).groupby(fl.sf).ffill()
        tp['c_tp_' + c] = fl.ts - last
    tp = pd.DataFrame(tp, index=fl.index)
    tp['c_tp_any_change'] = tp.min(axis=1)
    tp['c_tp_coord_change'] = tp[['c_tp_' + c for c in COORD_COLS]].min(axis=1)
    tp[['sf', 'sample_id']] = fl[['sf', 'sample_id']]
    merged = pd.DataFrame({'sf': sf, 'sample_id': df.sample_id}).merge(
        tp, on=['sf', 'sample_id'], how='left', validate='many_to_one')
    for c in tp.columns.difference(['sf', 'sample_id']):
        out[c] = merged[c].to_numpy()

    # --- which flight currently holds the controller's attention (supplied recency columns) ---
    out['c_input_recency_gap'] = df.a_since_flight - df.a_since_any
    out['c_last_input_is_flight'] = (df.a_since_flight <= df.a_since_any + 1e-6).astype(int)

    # --- competing tasks of the same aircraft inside this snapshot ---
    out['c_n_flights'] = df.assign(f=flight).groupby('sample_id').f.transform('nunique').to_numpy()
    fg = df.assign(f=flight).groupby(['sample_id', 'f'])
    out['c_tasks_this_flight'] = fg.task_name.transform('size').to_numpy()
    age = fg.x_task_age
    out['c_age_minus_flight_max'] = (df.x_task_age - age.transform('max')).to_numpy()
    out['c_age_minus_flight_min'] = (df.x_task_age - age.transform('min')).to_numpy()

    # --- urgency rank among candidates of the SAME task type in this snapshot ---
    stg = df.assign(abs_tts=df.time_to_sector.abs()).groupby(['sample_id', 'task_index'])
    out['c_rt_task_age'] = stg.x_task_age.rank(method='min').to_numpy()
    out['c_rt_abs_time_to_sector'] = stg.abs_tts.rank(method='min').to_numpy()
    return out
