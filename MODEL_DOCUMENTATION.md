# Model Documentation: Contact-Center Capacity Model

*Written in the structure of a Model Risk Management (MRM) development document. This is a portfolio project, so it has not been through independent validation.*

| Item | Detail |
|---|---|
| Model name | Contact-Center Capacity Model (CCM) v1.0 |
| Owner / developer | Taha Su |
| Model type | Forecasting (statistical) + queueing + optimization + simulation |
| Use | Weekly staffing and shift planning; service-level trade-off analysis |
| Run frequency | Weekly (forecast refresh), with monthly monitoring review |
| Last run | October 2026, planning week of Oct 5, 2026 |

## 1. Purpose and scope
- **Purpose:** estimate how many agents are needed, and when, to meet a service-level target at minimum labor cost.
- **In scope:** daily volume forecast for 28 days, interval staffing requirement, weekly full-time/part-time shift plan, and a stress test.
- **Out of scope:** hiring and attrition planning, skills-based routing, multi-channel work (chat, email), and intraday re-forecasting.

## 2. Data
- **Source:** `data.py` creates synthetic daily volume from Jan 1, 2023 to Sep 30, 2026 (1,369 days). It was used because real volumes are confidential.
- **Built-in structure:**
  - day-of-week cycle (Monday highest, Sunday lowest);
  - annual seasonality;
  - +4% growth per year;
  - U.S. federal holidays at 25% of normal volume, with +25% catch-up the next business day;
  - negative-binomial noise.
- **Intraday profile:** fixed share of daily calls per 30-minute interval, 8:00–20:00.
- **Data-quality checks for production use:** missing days, duplicate dates, zero-volume days outside holidays, and outliers more than 4 standard deviations from the model residual.

## 3. Methodology
1. **Forecast.** Four candidate models, all on log volume:
   - seasonal naive (benchmark);
   - Holt-Winters (additive trend and weekly seasonality);
   - OLS linear regression (day-of-week, month, linear trend, holiday and after-holiday flags);
   - SARIMAX (1,0,1)(1,1,1)₇ with holiday regressors.

   The champion is chosen by mean WAPE over a rolling-origin backtest.
2. **Staffing (Erlang C).** Interval traffic = calls × AHT ÷ 1,800 s. Required agents = the smallest N with service level ≥ 80% in 30 s.
3. **Shift optimization (MIP, Gurobi).**
   - Decision variables: integer counts of 8-hour full-time and 4-hour part-time starts per slot and day.
   - Constraints: shrinkage-adjusted coverage ≥ requirement in every interval; part-time hours ≤ 30% of paid hours.
   - Objective: minimize labor cost (full-time $30/h, part-time $33/h).
4. **Simulation (SimPy).** Discrete-event model of Monday:
   - piecewise-Poisson arrivals;
   - lognormal handle times (CV 0.8);
   - exponential caller patience (mean 180 s);
   - interval-varying staff, where agents finish their current call before leaving.

   20 replications per scenario.

## 4. Key assumptions
| # | Assumption | Impact if wrong | Mitigation |
|---|---|---|---|
| A1 | AHT = 360 s and stable | Staffing scales almost linearly with AHT | Monitor weekly AHT; re-run if it shifts more than 5% |
| A2 | Shrinkage = 30% | Under-coverage if actual shrinkage is higher | Sensitivity run; same-day call-in trigger |
| A3 | Intraday profile is the same every day | Interval mismatches on Mondays and weekends | Estimate a day-specific profile from interval data |
| A4 | Holidays move volume by fixed factors | Error around holidays | Holiday flags in the regression; planner override |
| A5 | Erlang C has no abandonment | Overstates required staff | The simulation measures the cushion (97% simulated service level vs. 80% target) |

## 5. Performance and validation (developer testing)
- **Backtest:** 6 origins × 28 days. Champion linear regression: MAPE 6.8%, WAPE 6.6%, bias −0.8%. Benchmark seasonal naive: WAPE 13.1%. Regression had the lowest WAPE in 4 of 6 windows; SARIMAX won the other 2.
- **Benchmarking:** the champion beats the naive benchmark in every window.
- **Optimization:** solved to proven optimality (MIP gap 0%). The plan is 12% above the continuous lower bound.
- **Outcome analysis (simulation):**
  - plan as forecast: 97.3% service level;
  - +15% volume: 87.1%;
  - −10% agents: 89.5%;
  - both shocks together: 69.6%, which is below target.
- **Known bias:** the synthetic data has a multiplicative structure, which favors log-linear regression. Champion selection must be repeated on real data.

## 6. Limitations
- Synthetic data, so production accuracy is not established.
- Single queue, single skill, and no callbacks or retrials.
- Weekly plan only; there are no intraday re-planning or overtime decisions.
- Forecast prediction intervals are not yet passed into staffing. A planned enhancement is to staff to a forecast percentile.

## 7. Ongoing monitoring plan
| Metric | Frequency | Green | Amber | Red (escalate / re-develop) |
|---|---|---|---|---|
| Weekly WAPE (28-day) | Weekly | ≤ 8% | 8–12% | > 12% for 3 consecutive weeks |
| Forecast bias | Weekly | within ±3% | ±3–6% | beyond ±6% |
| Achieved vs. planned service level | Weekly | ≥ target | 5 pts below | 10+ pts below |
| AHT and shrinkage drift | Monthly | < 5% | 5–10% | > 10% |
| Champion vs. challenger WAPE | Quarterly | champion best | within 1 pt | challenger better by > 1 pt |

## 8. Change log
| Version | Date | Change |
|---|---|---|
| 1.0 | Oct 2026 | Initial development: forecast, Erlang C, MIP schedule, simulation, documentation |
