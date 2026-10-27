"""Synthetic but realistic contact-center volume data.

Real bank operations volumes are confidential, so this generator builds daily
call volumes with the patterns seen in operations centers: weekly cycle,
annual seasonality, growth trend, federal-holiday dips with next-day catch-up,
and over-dispersed noise. Because the true patterns are known, we can check
that the forecasting models recover them.
"""
import numpy as np, pandas as pd, holidays

SEED = 7
OPEN_HOUR, CLOSE_HOUR = 8, 20                     # 8:00-20:00, 30-min intervals
INTERVALS = pd.date_range("2000-01-01 08:00", "2000-01-01 19:30", freq="30min").strftime("%H:%M")


def daily_volume(start="2023-01-01", end="2026-09-30"):
    rng = np.random.default_rng(SEED)
    d = pd.date_range(start, end, freq="D")
    t = np.arange(len(d)) / 365.25
    dow_factor = np.array([1.22, 1.08, 1.02, 0.99, 0.95, 0.45, 0.30])[d.dayofweek]   # Mon..Sun
    annual = 1 + 0.08 * np.sin(2 * np.pi * (d.dayofyear - 30) / 365.25) + 0.05 * np.cos(4 * np.pi * d.dayofyear / 365.25)
    trend = 1.04 ** t                                                               # +4% per year
    us = holidays.US(years=range(d.year.min(), d.year.max() + 1))
    hol = np.array([x in us for x in d.date])
    after = np.r_[False, hol[:-1]] & (d.dayofweek < 5)
    hol_factor = np.where(hol, 0.25, 1.0) * np.where(after, 1.25, 1.0)
    mu = 5200 * dow_factor * annual * trend * hol_factor
    k = 150                                                                         # negative-binomial dispersion
    y = rng.negative_binomial(k, k / (k + mu))
    return pd.DataFrame({"date": d, "calls": y, "holiday": hol.astype(int), "after_holiday": after.astype(int)})


def intraday_profile():
    """Share of a day's calls in each 30-minute interval (morning peak, lunch dip, afternoon shoulder)."""
    x = np.arange(len(INTERVALS))
    shape = 0.6 + 1.0 * np.exp(-((x - 5) / 3.0) ** 2) + 0.7 * np.exp(-((x - 13) / 4.0) ** 2) - 0.15 * np.exp(-((x - 9) / 1.2) ** 2)
    shape = np.clip(shape, 0.15, None) * np.where(x > 19, 0.6, 1.0)
    return pd.Series(shape / shape.sum(), index=INTERVALS, name="share")


if __name__ == "__main__":
    df = daily_volume(); print(df.tail()); print(df.calls.describe())
