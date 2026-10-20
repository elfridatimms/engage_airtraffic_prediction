"""Feature engineering for the AWARE task-prioritisation challenge."""
import numpy as np
import pandas as pd

BIG = 3600.0


def load():
    tr = pd.read_csv("train.csv")
    va = pd.read_csv("validation.csv")
    te = pd.read_csv("test.csv")
    te["target_index"] = -1
    df = pd.concat([tr, va, te], ignore_index=True)
    df["split"] = np.r_[np.zeros(len(tr)), np.ones(len(va)), np.full(len(te), 2)].astype(int)
    return df


def _next_event_time(times_by_key, keys, ts):
    """for each row: seconds until next event (>= ts) of its key, and since previous (< ts)"""
    nxt = np.full(len(ts), BIG)
    prv = np.full(len(ts), BIG)
    for i, (k, t) in enumerate(zip(keys, ts)):
        ev = times_by_key.get(k)
        if ev is None:
            continue
        j = np.searchsorted(ev, t - 1.5)
        if j < len(ev):
            nxt[i] = min(BIG, ev[j] - t)
        if j > 0:
            prv[i] = min(BIG, t - ev[j - 1])
    return nxt, prv


def build(df):
    df = df.copy()
    df["ts"] = pd.to_timedelta(df["timestamp"]).dt.total_seconds()
    df["flight"] = df["task_name"].str.split("_").str[-1]
    # conflict tasks name two aircraft; first one is the subject
    tn = df["task_name"]
    df["flight"] = np.where(df["task_index"].isin([0, 1, 6]),
                            tn.str.rsplit("_", n=2).str[-2], df["flight"])
    df["sf"] = df["session"] + "|" + df["flight"]
    df["st"] = df["session"] + "|" + df["task_name"]

    # ---- session-level temporal features (use full session, past and future) ----
    g = df.groupby("st")["ts"]
    df["task_first_ts"] = g.transform("min")
    df["task_last_ts"] = g.transform("max")
    df["t_until_task_end"] = df["task_last_ts"] - df["ts"]
    df["t_since_task_start"] = df["ts"] - df["task_first_ts"]
    sess_end = df.groupby("session")["ts"].transform("max")
    df["task_ends_before_session"] = (df["task_last_ts"] < sess_end - 1).astype(int)

    # controller input events per flight: time = ts - a_since_flight
    ev = df.loc[df["a_since_flight"] < BIG - 1, ["sf", "ts", "a_since_flight"]].copy()
    ev["evt"] = (ev["ts"] - ev["a_since_flight"]).round()
    evs = {k: np.unique(v.values) for k, v in ev.groupby("sf")["evt"]}
    df["t_next_input_flight"], _ = _next_event_time(evs, df["sf"].values, df["ts"].values)

    # input type at next input: look at snapshot right after the event
    for col in ["a_last_assume_control", "a_last_assign_level_clearance",
                "a_last_assign_lateral_clearance", "a_last_initiate_transfer_of_control"]:
        e = df.loc[df[col] == 1, ["sf", "ts", "a_since_flight"]]
        e = e[e["a_since_flight"] < BIG - 1]
        t = (e["ts"] - e["a_since_flight"]).round()
        d = {k: np.unique(v.values) for k, v in t.groupby(e["sf"])}
        nx, _ = _next_event_time(d, df["sf"].values, df["ts"].values)
        df["t_next_" + col[7:]] = nx

    # state transitions of the flight
    for col in ["is_assumed", "x_transfer_in", "x_isc_coordinated", "k_out_COORDINATED", "within_sector"]:
        on = df.loc[df[col] == 1].groupby("sf")["ts"].min()
        df["t_until_" + col] = (df["sf"].map(on) - df["ts"]).fillna(BIG).clip(-BIG, BIG)

    # session-wide busy / any input next
    ev_any = df[["session", "ts", "a_since_any"]].drop_duplicates()
    ev_any = ev_any[ev_any.a_since_any < BIG - 1]
    ev_any["evt"] = (ev_any["ts"] - ev_any["a_since_any"]).round()
    d = {k: np.unique(v.values) for k, v in ev_any.groupby("session")["evt"]}
    df["t_next_input_any"], _ = _next_event_time(d, df["session"].values, df["ts"].values)
    # is this flight the one receiving the next input of the session
    df["next_input_is_flight"] = (np.abs(df["t_next_input_flight"] - df["t_next_input_any"]) < 1.5).astype(int)

    # previous / next snapshot presence of the task
    df = df.sort_values(["session", "ts", "row_index"]).reset_index(drop=True)
    sid_order = df[["session", "sample_id", "ts"]].drop_duplicates("sample_id").sort_values(["session", "ts"])
    sid_order["k"] = sid_order.groupby("session").cumcount()
    df["snap_k"] = df["sample_id"].map(sid_order.set_index("sample_id")["k"])
    key = set(zip(df["st"], df["snap_k"]))
    df["in_prev"] = [int((s, k - 1) in key) for s, k in zip(df["st"], df["snap_k"])]
    df["in_next"] = [int((s, k + 1) in key) for s, k in zip(df["st"], df["snap_k"])]
    gap = sid_order.groupby("session")["ts"].diff().fillna(BIG)
    df["gap_prev"] = df["sample_id"].map(dict(zip(sid_order.sample_id, gap)))
    gapn = -sid_order.groupby("session")["ts"].diff(-1).fillna(-BIG)
    df["gap_next"] = df["sample_id"].map(dict(zip(sid_order.sample_id, gapn)))

    # ---- within-sample relative features ----
    sg = df.groupby("sample_id")
    for c in ["x_task_age", "t_next_input_flight", "t_until_task_end", "a_since_flight",
              "t_until_is_assumed", "t_until_x_transfer_in", "distance_to_zone", "time_to_MD"]:
        df["r_" + c] = sg[c].rank(method="min")
    df["abs_tts"] = df["time_to_sector"].abs()
    df["r_abs_tts"] = sg["abs_tts"].rank(method="min")
    df["n_flights"] = sg["flight"].transform("nunique")
    # rank within (sample, type) for the strongest temporal signals
    stg = df.groupby(["sample_id", "task_index"])
    for c in ["t_next_input_flight", "t_until_task_end", "x_task_age"]:
        df["rt_" + c] = stg[c].rank(method="min")
    # within flight: is this the oldest/newest task of the aircraft
    fg = df.groupby(["sample_id", "flight"])["x_task_age"]
    df["age_minus_flight_max"] = df["x_task_age"] - fg.transform("max")
    df["age_minus_flight_min"] = df["x_task_age"] - fg.transform("min")
    return df


DROP = {"sample_id", "target_index", "session", "atco", "timestamp", "task_name", "flight",
        "sf", "st", "split", "ts", "task_first_ts", "task_last_ts", "snap_k", "row_index"}


def feature_cols(df):
    return [c for c in df.columns if c not in DROP]
