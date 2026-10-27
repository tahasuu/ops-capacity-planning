"""Daily volume forecasting with rolling-origin backtesting.

Models compared (all forecast 28 days ahead):
  1. Seasonal naive  - same weekday last week (the benchmark to beat)
  2. Holt-Winters    - additive trend + weekly seasonality on log volume
  3. Linear regression - day-of-week, month, trend and holiday features (log volume)
  4. SARIMAX         - (1,0,1)x(1,1,1,7) with holiday regressors
Backtest: 6 rolling origins, each forecasting the next 28 days.
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from data import daily_volume

H = 28


def features(df):
    X = pd.get_dummies(df["date"].dt.dayofweek.astype(str), prefix="dow", drop_first=True)
    X = X.join(pd.get_dummies(df["date"].dt.month.astype(str), prefix="m", drop_first=True))
    X["trend"] = (df["date"] - pd.Timestamp("2023-01-01")).dt.days / 365.25
    X[["holiday", "after_holiday"]] = df[["holiday", "after_holiday"]].values
    return sm.add_constant(X.astype(float), has_constant="add")


def fc_snaive(tr, te):
    last = tr.set_index(tr.date.dt.dayofweek)["calls"].iloc[-7:]
    p = te.date.dt.dayofweek.map(last).values.astype(float)
    return np.where(te.holiday == 1, p * 0.25, p)       # planners already know holidays are low


def fc_hw(tr, te):
    m = ExponentialSmoothing(np.log(tr.calls.values), trend="add", seasonal="add", seasonal_periods=7).fit()
    p = np.exp(m.forecast(len(te)))
    return p * np.where(te.holiday == 1, 0.25, 1) * np.where(te.after_holiday == 1, 1.25, 1)


def fc_reg(tr, te):
    Xa = features(pd.concat([tr, te], ignore_index=True))
    m = sm.OLS(np.log(tr.calls.values), Xa.iloc[:len(tr)]).fit()
    return np.exp(m.predict(Xa.iloc[len(tr):]).values + m.mse_resid / 2)


def fc_sarimax(tr, te):
    ex = ["holiday", "after_holiday"]
    m = SARIMAX(np.log(tr.calls.values), exog=tr[ex].values, order=(1, 0, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False, maxiter=200)
    return np.exp(m.forecast(len(te), exog=te[ex].values))


MODELS = {"Seasonal naive": fc_snaive, "Holt-Winters": fc_hw, "Linear regression": fc_reg, "SARIMAX": fc_sarimax}


def metrics(y, p):
    return {"MAPE_%": np.mean(np.abs(p - y) / y) * 100, "WAPE_%": np.abs(p - y).sum() / y.sum() * 100,
            "bias_%": (p.sum() - y.sum()) / y.sum() * 100}


def backtest(df, origins=6):
    rows = []
    end = len(df) - H
    for i in range(origins):
        cut = end - i * H
        tr, te = df.iloc[:cut], df.iloc[cut:cut + H]
        for name, f in MODELS.items():
            p = f(tr, te)
            rows.append({"origin": te.date.iloc[0].date(), "model": name, **metrics(te.calls.values, p)})
    return pd.DataFrame(rows)


def run(out="results"):
    df = daily_volume()
    bt = backtest(df)
    summ = bt.groupby("model")[["MAPE_%", "WAPE_%", "bias_%"]].mean().round(2).sort_values("WAPE_%")
    bt.round(2).to_csv(f"{out}/backtest_by_origin.csv", index=False); summ.to_csv(f"{out}/backtest_summary.csv")
    # champion model: forecast the next 28 days after the data ends (Oct 2026)
    best = summ.index[0]
    fut = pd.DataFrame({"date": pd.date_range(df.date.max() + pd.Timedelta(days=1), periods=H)})
    import holidays; us = holidays.US(years=[2026, 2027])
    fut["holiday"] = fut.date.dt.date.map(lambda x: int(x in us))
    fut["after_holiday"] = (fut.holiday.shift(1, fill_value=0).astype(bool) & (fut.date.dt.dayofweek < 5)).astype(int)
    fut["forecast"] = np.round(MODELS[best](df, fut)).astype(int)
    fut.to_csv(f"{out}/forecast_next_28_days.csv", index=False)
    return summ, best, fut


if __name__ == "__main__":
    s, b, f = run(); print(s); print("champion:", b); print(f.head(10))
