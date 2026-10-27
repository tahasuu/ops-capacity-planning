"""End-to-end capacity plan: forecast -> staffing -> shift optimization -> simulation check."""
import json, warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
import forecast, staffing, schedule, simulate
from data import INTERVALS

OUT = "results"


def on_phone(shifts, n_iv=24):
    cov = np.zeros((7, n_iv))
    for _, s in shifts.iterrows():
        L = schedule.FT_LEN if s.type == "FT" else schedule.PT_LEN
        cov[s.day, s.start_iv:s.start_iv + L] += s.agents
    return np.floor(cov * (1 - schedule.SHRINKAGE)).astype(int)


def main():
    summ, best, fut = forecast.run(OUT)
    week = fut[(fut.date >= "2026-10-05") & (fut.date <= "2026-10-11")].reset_index(drop=True)   # Mon-Sun
    req = staffing.interval_requirements(week)
    req["day"] = req.date.dt.dayofweek; req["iv"] = req.groupby("date").cumcount()
    req.round(1).to_csv(f"{OUT}/interval_requirements.csv", index=False)

    plan, shifts = schedule.optimize_schedule(req[["day", "iv", "agents"]])
    ft_only, _ = schedule.optimize_schedule(req[["day", "iv", "agents"]], max_pt_share=0.0)
    plan["ft_only_cost_$"] = ft_only["weekly_cost_$"]; plan["ft_only_paid_hours"] = ft_only["paid_hours"]
    plan["savings_vs_ft_only_%"] = round(100 * (1 - plan["weekly_cost_$"] / ft_only["weekly_cost_$"]), 1)
    shifts["start_time"] = shifts.start_iv.map(dict(enumerate(INTERVALS)))
    shifts.to_csv(f"{OUT}/optimal_shifts.csv", index=False)

    # service-level target sensitivity: what does 90/30 cost vs 80/30?
    sens = []
    for sl in [0.70, 0.80, 0.90]:
        r = staffing.interval_requirements(week, sl=sl); r["day"] = r.date.dt.dayofweek; r["iv"] = r.groupby("date").cumcount()
        p, _ = schedule.optimize_schedule(r[["day", "iv", "agents"]])
        sens.append({"service_level_target": f"{int(sl*100)}% in 30s", "weekly_cost_$": p["weekly_cost_$"], "paid_hours": p["paid_hours"], "fte_40h": p["fte_40h"]})
    sens = pd.DataFrame(sens); sens.to_csv(f"{OUT}/sl_target_sensitivity.csv", index=False)

    # simulation check of Monday (busiest day)
    mon = req[req.day == 0]
    calls = mon.calls.values
    staff = on_phone(shifts)[0]
    scen = {"Plan as forecast": (calls, staff),
            "Plan with 10% fewer agents on phones": (calls, np.floor(staff * 0.9).astype(int)),
            "Volume 15% above forecast": (calls * 1.15, staff),
            "Volume +15% and 10% of agents absent": (calls * 1.15, np.floor(staff * 0.9).astype(int))}
    rows = []
    for name, (c, a) in scen.items():
        d, per = simulate.replicate(c, a, reps=20)
        rows.append({"scenario": name, "service_level_%": round(100 * d.service_level, 1), "abandon_rate_%": round(100 * d.abandon_rate, 1),
                     "worst_interval_sl_%": round(100 * per.service_level.min(), 1)})
        if name == "Plan as forecast":
            per.index = [INTERVALS[i] for i in per.index]; per.round(3).to_csv(f"{OUT}/simulation_monday_by_interval.csv")
    sim = pd.DataFrame(rows); sim.to_csv(f"{OUT}/simulation_scenarios.csv", index=False)

    out = {"champion_model": best, "backtest": summ.reset_index().to_dict("records"), "schedule": plan,
           "sl_sensitivity": sens.to_dict("records"), "simulation": sim.to_dict("records"),
           "week_forecast": week[["date", "forecast"]].assign(date=week.date.dt.strftime("%a %b %d")).to_dict("records"),
           "monday_required_agents": mon.agents.tolist(), "monday_scheduled_on_phone": staff.tolist()}
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=2, default=str)
    print(summ); print(json.dumps(plan, indent=1)); print(sens); print(sim.to_string())
    print("Mon req", mon.agents.tolist()); print("Mon staff", staff.tolist())


if __name__ == "__main__":
    main()
