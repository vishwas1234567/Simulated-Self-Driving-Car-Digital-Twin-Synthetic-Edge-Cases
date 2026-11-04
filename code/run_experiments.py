"""Experiment 2: does diffusion-generated edge-case data make the driving policy safer?

Arms (equal total number of training scenarios = 2700):
  A  normal only                       (baseline)
  B  1500 normal + 1200 guided-diffusion edge cases   (failure-seeking generator)
  C  1500 normal + 1200 unguided diffusion samples    (generator control)
  D  1500 normal + 1200 uniform random scenarios      (naive domain randomisation)
Repeated for several seeds. Test sets are fixed and independent of every generator.
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool

from sim import sample_normal, sample_uniform, open_loop_gap
from driving import train_policy, evaluate
from scenario_gen import ScenarioDiffusion

from paths import RES
OUT = str(RES)
N_NORMAL_BASE, N_GEN = 1500, 1200
GUIDANCE = 8.0
SEEDS = [0, 1, 2, 3, 4]


def test_sets():
    rng = np.random.default_rng(1000)
    normal = sample_normal(rng, 800)
    U = sample_uniform(np.random.default_rng(1001), 12000)
    haz = U[open_loop_gap(U) < 0.3][:800]
    hard = (haz[:, 6] > 0.4) | (haz[:, 5] > 0.6)        # dark clothing / odd costume
    return normal, haz, hard


def summarise(r, mask=None):
    m = np.ones(len(r["collided"]), bool) if mask is None else mask
    return dict(coll=float(r["collided"][m].mean()), passed=float(r["passed"][m].mean()),
                gap=float(r["min_gap"][m].mean()), n=int(m.sum()))


def run_seed(seed):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    Tn, Th, hard = test_sets()
    P_norm = sample_normal(rng, N_NORMAL_BASE + N_GEN)

    # ---- baseline policy (arm A)
    A = train_policy(P_norm, seed)
    # ---- failure surrogate: closed-loop min gap of A on uniform scenarios
    Pu = sample_uniform(rng, 3000)
    ru = evaluate(Pu, A, seed=seed)
    y = np.clip(ru["min_gap"], 0, 5) / 5.0
    # ---- diffusion prior + surrogate
    gen = ScenarioDiffusion(seed)
    gen.fit_prior(sample_normal(rng, 20000), iters=6000)
    gen.fit_surrogate(Pu, y, iters=5000)
    G_guided = gen.sample(N_GEN, GUIDANCE, seed=seed + 100)
    G_free = gen.sample(N_GEN, 0.0, seed=seed + 200)
    G_unif = sample_uniform(rng, N_GEN)

    sets = {"A": P_norm,
            "B": np.concatenate([P_norm[:N_NORMAL_BASE], G_guided]),
            "C": np.concatenate([P_norm[:N_NORMAL_BASE], G_free]),
            "D": np.concatenate([P_norm[:N_NORMAL_BASE], G_unif])}
    res = {}
    for arm, P in sets.items():
        net = A if arm == "A" else train_policy(P, seed + 7)
        rn, rh = evaluate(Tn, net, seed=seed), evaluate(Th, net, seed=seed)
        res[arm] = {"normal": summarise(rn), "hazard": summarise(rh),
                    "hard": summarise(rh, hard), "hazard_easy": summarise(rh, ~hard)}
        print(f"[seed {seed}] arm {arm}: normal coll {res[arm]['normal']['coll']:.3f} | "
              f"hazard coll {res[arm]['hazard']['coll']:.3f} | hard {res[arm]['hard']['coll']:.3f}", flush=True)

    # ---- diagnostics for how 'hard' each generated set is (for the report)
    diag = {}
    for name, S in [("normal", P_norm[:N_GEN]), ("unguided", G_free), ("guided", G_guided), ("uniform", G_unif)]:
        ra, re = evaluate(S, A, seed=seed), evaluate(S, "expert", seed=seed)
        diag[name] = dict(A_coll=float(ra["collided"].mean()), expert_coll=float(re["collided"].mean()),
                          ol_hazard=float((open_loop_gap(S) < 0.3).mean()),
                          A_gap=float(ra["min_gap"].mean()))

    extra = {}
    if seed == SEEDS[0]:        # guidance-strength sweep + arrays for figures
        sweep = []
        for lam in [0, 2, 4, 6, 8, 10, 12]:
            S = gen.sample(800, float(lam), seed=500 + lam)
            sweep.append(dict(lam=lam, A_coll=float(evaluate(S, A, seed=1)["collided"].mean()),
                              expert_coll=float(evaluate(S, "expert", seed=1)["collided"].mean()),
                              ol_hazard=float((open_loop_gap(S) < 0.3).mean())))
        extra["sweep"] = sweep
        coll_g = evaluate(G_guided, A, seed=seed)["collided"]
        coll_n = evaluate(P_norm[:N_GEN], A, seed=seed)["collided"]
        np.savez(f"{OUT}/fig_data_seed0.npz", normal=P_norm[:N_GEN], free=G_free, guided=G_guided,
                 uniform=G_unif, coll_guided=coll_g, coll_normal=coll_n,
                 prior_loss=np.array(gen.loss_curve), surr_loss=np.array(gen.surr_curve),
                 dagger_loss=np.array(A.curve))
        A.save(f"{OUT}/policy_A_seed0.npz")
    return dict(seed=seed, res=res, diag=diag, extra=extra, time=time.time() - t0)


if __name__ == "__main__":

    Tn, Th, hard = test_sets()
    ex = {"normal": summarise(evaluate(Tn, "expert")), "hazard": summarise(evaluate(Th, "expert")),
          "hard": summarise(evaluate(Th, "expert"), hard)}
    ol = float((open_loop_gap(Th) < 0.3).mean())
    print("expert reference:", ex, flush=True)
    with Pool(2) as pool:
        runs = pool.map(run_seed, SEEDS)
    json.dump(dict(expert=ex, runs=runs, n_hard=int(hard.sum()), seeds=SEEDS,
                   guidance=GUIDANCE, n_normal=N_NORMAL_BASE, n_gen=N_GEN),
              open(f"{OUT}/experiments.json", "w"), indent=1)
    print("done", [round(r["time"]) for r in runs])
