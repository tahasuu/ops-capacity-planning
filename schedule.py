"""Shift-scheduling optimization (Gurobi MIP).

Decide how many full-time (8h) and part-time (4h) agents start at each
30-minute slot on each day so that, after shrinkage (breaks, training,
absence), on-phone agents cover the Erlang C requirement in every interval,
at minimum labor cost.
"""
import numpy as np, pandas as pd
import gurobipy as gp
from gurobipy import GRB

SHRINKAGE = 0.30        # share of paid time not available for calls
FT_LEN, PT_LEN = 16, 8  # intervals (8h, 4h)
FT_RATE, PT_RATE = 30.0, 33.0   # $ per paid hour (part-time premium: hiring/training overhead)
MAX_PT_SHARE = 0.30     # cap on part-time share of paid hours


def optimize_schedule(req, n_iv=24, time_limit=60, max_pt_share=None):
    """req: DataFrame with columns day (0-6), iv (0..n_iv-1), agents. Returns (summary dict, shifts DataFrame)."""
    days = sorted(req.day.unique())
    R = req.set_index(["day", "iv"]).agents
    ft_starts = range(0, n_iv - FT_LEN + 1)
    pt_starts = range(0, n_iv - PT_LEN + 1)
    m = gp.Model("shifts"); m.Params.OutputFlag = 0; m.Params.TimeLimit = time_limit
    x = m.addVars(days, ft_starts, vtype=GRB.INTEGER, name="ft")
    y = m.addVars(days, pt_starts, vtype=GRB.INTEGER, name="pt")
    for d in days:
        for i in range(n_iv):
            on = gp.quicksum(x[d, s] for s in ft_starts if s <= i < s + FT_LEN) + \
                 gp.quicksum(y[d, s] for s in pt_starts if s <= i < s + PT_LEN)
            m.addConstr((1 - SHRINKAGE) * on >= R[d, i], name=f"cover_{d}_{i}")
    ft_hours = gp.quicksum(x[d, s] for d in days for s in ft_starts) * FT_LEN / 2
    pt_hours = gp.quicksum(y[d, s] for d in days for s in pt_starts) * PT_LEN / 2
    cap = MAX_PT_SHARE if max_pt_share is None else max_pt_share
    m.addConstr(pt_hours <= cap * (ft_hours + pt_hours), name="pt_cap")
    m.setObjective(FT_RATE * ft_hours + PT_RATE * pt_hours, GRB.MINIMIZE)
    m.optimize()
    shifts = [{"day": d, "type": "FT", "start_iv": s, "agents": int(round(x[d, s].X))} for d in days for s in ft_starts if x[d, s].X > 0.5] + \
             [{"day": d, "type": "PT", "start_iv": s, "agents": int(round(y[d, s].X))} for d in days for s in pt_starts if y[d, s].X > 0.5]
    paid_hours = ft_hours.getValue() + pt_hours.getValue()
    req_hours = req.agents.sum() / 2 / (1 - SHRINKAGE)  # theoretical minimum paid hours
    summary = {"status": m.Status, "mip_gap": m.MIPGap, "weekly_cost_$": round(m.ObjVal),
               "paid_hours": round(paid_hours), "ft_hours": round(ft_hours.getValue()), "pt_hours": round(pt_hours.getValue()),
               "lower_bound_hours": round(req_hours),
               "fte_40h": round(paid_hours / 40, 1)}
    return summary, pd.DataFrame(shifts)
