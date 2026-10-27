"""Interval staffing requirements with the Erlang C queueing model.

For each 30-minute interval: arrival rate from the daily forecast x intraday
profile, average handle time (AHT), and a service-level target (80% of calls
answered within 30 seconds). Returns the minimum agents on the phones.
"""
import math
import numpy as np, pandas as pd
from data import intraday_profile

AHT = 360          # seconds
INTERVAL = 1800    # seconds
SL_TARGET, ANSWER_SEC = 0.80, 30


def erlang_c(n, a):
    """Probability a call has to wait (n agents, a Erlangs of traffic)."""
    if n <= a:
        return 1.0
    s, term = 0.0, 1.0
    for k in range(n):
        if k > 0:
            term *= a / k
        s += term
    top = term * a / n * n / (n - a)
    return top / (s + top)


def service_level(n, a, aht=AHT, t=ANSWER_SEC):
    return 1 - erlang_c(n, a) * math.exp(-(n - a) * t / aht)


def agents_needed(calls, aht=AHT, sl=SL_TARGET, t=ANSWER_SEC):
    a = calls * aht / INTERVAL
    if a == 0:
        return 0
    n = max(1, math.ceil(a))
    while service_level(n, a, aht, t) < sl:
        n += 1
    return n


def interval_requirements(daily_fc, sl=SL_TARGET, volume_mult=1.0):
    """daily_fc: DataFrame(date, forecast). Returns long table of interval calls and required agents."""
    prof = intraday_profile()
    rows = []
    for _, r in daily_fc.iterrows():
        for iv, share in prof.items():
            c = r["forecast"] * share * volume_mult
            rows.append({"date": r["date"], "interval": iv, "calls": c, "agents": agents_needed(c, sl=sl)})
    return pd.DataFrame(rows)
