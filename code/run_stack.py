"""Experiments for the planning/safety stack: verification, planner, tracking, fault injection."""
import itertools, json, os, time
import numpy as np
from sim import A_MIN, A_MAX
from stack import (Ctx, build_tree, tick, run_bt, run_planner, FAULTS)
from run_experiments import test_sets
from driving import evaluate
from nn import MLP
from driving import OBS_DIM

from paths import RES
OUT = str(RES)


def verify_tree():
    """Exhaustive check of the behaviour tree over its abstract state space: hazard level
    (0 none, 1 conflict, 2 must brake, 3 emergency) x occluder x sensor-bad x braking capability (8/4),
    executing the real action code."""
    tree, rows = build_tree(), {}
    for lv, o, bad, ab in itertools.product(range(4), [0, 1], [0, 1], [8.0, 4.0]):
        c = Ctx()
        c.conflict, c.brake_need, c.emerg = (np.array([lv >= k]) for k in (1, 2, 3))
        c.occl, c.bad = np.array([bool(o)]), np.array([bool(bad)])
        c.abrk, c.v = np.array([ab]), np.array([10.0])
        c.d = np.array([{0: 200.0, 1: 200.0, 2: 40.0, 3: 4.0}[lv]])
        c.a_req = c.v ** 2 / (2 * c.d)
        rows[(lv, o, bad, ab)] = (float(tick(tree, c, 1)[0]), int(c.chosen[0]))
    names = ["P1 total, within actuator limits", "P2 emergency => full brake", "P3 must-brake => brake in [-abrk,-0.3]",
             "P4 conflict => never accelerate", "P5 no conflict => comfortable accel", "P6 monotone in hazard level"]
    P = dict.fromkeys(names, 0)
    for (lv, o, bad, ab), (a, ch) in rows.items():
        P[names[0]] += ch >= 0 and np.isfinite(a) and A_MIN <= a <= A_MAX
        P[names[1]] += lv < 3 or a == -ab
        P[names[2]] += lv != 2 or (-ab <= a <= -0.3)
        P[names[3]] += lv == 0 or a <= 0.0
        P[names[4]] += lv > 0 or (-3.0 <= a <= 2.5)
        P[names[5]] += lv == 0 or a <= rows[(lv - 1, o, bad, ab)][0]
    return len(rows), P


def bmc_grid():
    g = itertools.product([6., 8., 10., 12., 14.], [35., 45., 55., 70.], [0.8, 1.4, 2.2, 3.5],
                          np.arange(-6, 6.01, 0.5), [0., .5, 1.], [0.25, 0.6, 1.0])
    P = np.array([[v, d, s, t, o, r, 0.0, 0.05] for v, d, s, t, o, r in g])
    sim, _ = run_bt(P, np.random.default_rng(0), mode="oracle")
    bad = np.where(sim.collided)[0]
    return len(P), len(bad), P[bad][:8].tolist()


def stat(sim):
    return dict(coll=float(sim.collided.mean()), passed=float(sim.passed.mean()),
                gap=float(np.minimum(sim.min_gap, 20).mean()))


if __name__ == "__main__":
    t0, R = time.time(), {}
    n_abs, props = verify_tree()
    R["tree"] = dict(n_states=n_abs, props=props)
    print("tree", n_abs, props, flush=True)
    n, nbad, ex = bmc_grid()
    R["bmc"] = dict(n=n, violations=nbad, examples=ex)
    print("bmc", n, nbad, ex[:3], flush=True)

    Tn, Th, hard = test_sets()
    A = MLP([OBS_DIM, 64, 64, 1], np.random.default_rng(0), out_scale=0.1); A.load(f"{OUT}/policy_A_seed0.npz")
    N = 100
    R["compare"] = {
        "Learned policy A": {k: float(v.mean()) for k, v in evaluate(Th[:N], A, seed=1).items() if k in ("collided", "passed")},
        "BT (lidar perception)": stat(run_bt(Th[:N], np.random.default_rng(1))[0]),
        "BT (oracle perception)": stat(run_bt(Th[:N], np.random.default_rng(1), mode="oracle")[0]),
    }
    for nm, kw in [("Lattice+opt+PD", {}), ("Lattice+opt+PD+BT supervisor", dict(supervisor=True))]:
        sim, st = run_planner(Th[:N], np.random.default_rng(1), **kw)
        R["compare"][nm] = {**stat(sim), **st}
        print(nm, R["compare"][nm], flush=True)
    sim, _ = run_bt(Th, np.random.default_rng(1)); R["bt_full_hazard"] = stat(sim)
    sim, _ = run_bt(Tn, np.random.default_rng(1)); R["bt_full_normal"] = stat(sim)
    print("BT full", R["bt_full_hazard"], R["bt_full_normal"], flush=True)

    R["tracking"] = {}
    for nm, kw in [("feed-forward only", dict(kp=0.0, kd=0.0)), ("PD + feed-forward", {})]:
        for mm in (1.0, 0.8):
            sim, st = run_planner(Th[:60], np.random.default_rng(2), mismatch=mm, supervisor=False, **kw)
            R["tracking"][f"{nm} | brake gain {mm}"] = {**stat(sim), **st}
            print(nm, mm, R["tracking"][f"{nm} | brake gain {mm}"], flush=True)
    sim, st = run_planner(Th[:60], np.random.default_rng(2), use_smooth=False)
    R["no_smooth"] = {**stat(sim), **st}

    R["faults"] = {}
    for f in [None] + FAULTS:
        R["faults"][str(f)] = {}
        for mon in (False, True):
            sim, _ = run_bt(Th[:400], np.random.default_rng(3), fault=f, monitor=mon)
            R["faults"][str(f)]["monitor" if mon else "plain"] = stat(sim)
        print(f, R["faults"][str(f)], flush=True)
    R["seconds"] = time.time() - t0
    json.dump(R, open(f"{OUT}/stack.json", "w"), indent=1)
    print("done", round(R["seconds"]))
