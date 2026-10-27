"""Discrete-event simulation (SimPy) of one contact-center day.

Erlang C assumes exponential handle times, infinite patience and steady
arrivals. The simulation relaxes those assumptions to check the staffing plan:
  - Poisson arrivals that change rate every 30 minutes
  - lognormal handle times (mean = AHT, more realistic long tail)
  - callers hang up if not answered within their patience (exponential)
  - staffing changes by interval (agents finish their current call first)
"""
import numpy as np, pandas as pd, simpy
from staffing import AHT, INTERVAL, ANSWER_SEC

PATIENCE = 180     # mean seconds before a caller abandons


def simulate_day(calls, agents, seed=0, aht=AHT, cv=0.8):
    """calls, agents: arrays per 30-min interval. Returns per-interval KPIs."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    max_n = int(max(agents)) + 1
    pool = simpy.PriorityResource(env, capacity=max_n)
    sig = np.sqrt(np.log(1 + cv ** 2)); mu = np.log(aht) - sig ** 2 / 2
    log = []
    blockers = []

    def caller(i):
        t0 = env.now
        with pool.request(priority=1) as req:
            res = yield req | env.timeout(rng.exponential(PATIENCE))
            if req in res:
                wait = env.now - t0
                log.append((i, wait, False))
                yield env.timeout(rng.lognormal(mu, sig))
            else:
                log.append((i, env.now - t0, True))

    def blocker(stop_evt):
        with pool.request(priority=0) as req:      # takes the next free agent off the phones
            yield req
            yield stop_evt

    def staffing():
        for i, n in enumerate(agents):
            need = max_n - int(n)
            while len(blockers) < need:
                ev = env.event(); blockers.append(ev); env.process(blocker(ev))
            while len(blockers) > need:
                blockers.pop().succeed()
            yield env.timeout(INTERVAL)

    def arrivals():
        for i, c in enumerate(calls):
            end = (i + 1) * INTERVAL
            rate = c / INTERVAL
            while rate > 0:
                gap = rng.exponential(1 / rate)
                if env.now + gap >= end:
                    break
                yield env.timeout(gap)
                env.process(caller(i))
            if env.now < end:
                yield env.timeout(end - env.now)

    env.process(staffing()); env.process(arrivals())
    env.run(until=len(calls) * INTERVAL + 3600)
    d = pd.DataFrame(log, columns=["interval", "wait", "abandoned"])
    g = d.groupby("interval")
    return pd.DataFrame({"offered": g.size(),
                         "service_level": g.apply(lambda x: ((~x.abandoned) & (x.wait <= ANSWER_SEC)).sum() / len(x)),
                         "abandon_rate": g.abandoned.mean(),
                         "asa_sec": g.apply(lambda x: x.loc[~x.abandoned, "wait"].mean())})


def replicate(calls, agents, reps=20):
    runs = [simulate_day(calls, agents, seed=s) for s in range(reps)]
    tot = pd.concat(runs)
    daily = [{"service_level": (r.service_level * r.offered).sum() / r.offered.sum(),
              "abandon_rate": (r.abandon_rate * r.offered).sum() / r.offered.sum()} for r in runs]
    per_iv = tot.groupby(level=0).mean()
    return pd.DataFrame(daily).mean(), per_iv
