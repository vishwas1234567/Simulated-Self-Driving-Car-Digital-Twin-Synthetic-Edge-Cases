"""Experiment 1: NeRF-style digital twin of a logged drive + what-if replay.

1. Log 10 expert drives through one scene (parked car + one crossing pedestrian).
2. Fit the spacetime neural field to the lidar logs by differentiable volume rendering.
3. Novel-viewpoint check on two held-out drives (different lateral offsets / speeds).
4. What-if: re-render the scene with the pedestrian shifted by `delta` seconds and drive
   the learned policy *inside the twin*; compare with the same what-if in the real
   simulator (ground truth).
"""
import json, os, sys, time
import numpy as np
from nn import MLP
from sim import BatchSim, ped_traj, PED_Y0, N_RAYS, ANG, R_MAX, SENSOR_DX, N_STEPS
from driving import policy_action, OBS_DIM
from nerf2d import NeuralField, run_logged_drives, to_rays, TwinSim

from paths import RES
OUT = str(RES)
T0, T_FIELD = 1.2, 12.0
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
BASE = dict(dist=60.0, s=1.6, tau=0.0, occl=0.8, pr=0.35, drop=0.0)


def scene(v0, noise, pr=None):
    v0 = np.atleast_1d(np.asarray(v0, float))
    n = len(v0)
    cols = [v0, np.full(n, BASE["dist"]), np.full(n, BASE["s"]), np.full(n, BASE["tau"]),
            np.full(n, BASE["occl"]), np.full(n, BASE["pr"] if pr is None else pr),
            np.full(n, BASE["drop"]), np.full(n, noise)]
    return np.stack(cols, 1)


def load_policy():
    net = MLP([OBS_DIM, 64, 64, 1], np.random.default_rng(0), out_scale=0.1)
    net.load(f"{OUT}/policy_A_seed0.npz")
    return net


if __name__ == "__main__":

    t0 = time.time()
    # ---- 1. logged drives
    K = 10
    v_log, ey_log = np.linspace(8.0, 13.0, K), np.linspace(-0.8, 0.8, K)
    rec = run_logged_drives(scene(v_log, 0.03), np.full(K, T0), ey_log, v_log, seed=1)
    rays = to_rays(rec, T_FIELD)
    print(f"logged {K} drives -> {len(rays[0])} lidar rays", flush=True)

    # ---- 2. fit the field
    field = NeuralField(BASE["dist"], T_FIELD, seed=0)
    t1 = time.time()
    field.fit(rays, iters=ITERS)
    fit_s = time.time() - t1
    print(f"fit time {fit_s:.0f}s", flush=True)

    # ---- 3. novel viewpoints (held-out drives, noise-free ground truth)
    v_te, ey_te = np.array([9.5, 11.5]), np.array([1.3, -1.3])
    rec_te = run_logged_drives(scene(v_te, 0.0), np.full(2, T0), ey_te, v_te, seed=2)
    ox, oy, ang, tt, gt = to_rays(rec_te, T_FIELD)
    pred = field.render(ox, oy, ang, tt)
    pred_static = field.render(ox, oy, ang, tt, use_dyn=False)
    # rays whose nearest hit is the pedestrian: same poses, pedestrian removed
    keep = rec_te["t"] <= T_FIELD
    s0 = BatchSim(scene(np.full(keep.sum(), 10.0), 0.0, pr=1e-3), np.random.default_rng(0))
    s0.x, s0.ey = rec_te["ox"][keep] - SENSOR_DX, rec_te["oy"][keep].copy()
    gt_np = s0.lidar().ravel()
    ped_rays = np.abs(gt - gt_np) > 0.3
    nv = dict(mae=float(np.abs(pred - gt).mean()), rmse=float(np.sqrt(((pred - gt) ** 2).mean())),
              within_half_m=float((np.abs(pred - gt) < 0.5).mean()),
              ped_rays=int(ped_rays.sum()), n_rays=int(len(gt)),
              mae_ped=float(np.abs(pred - gt)[ped_rays].mean()) if ped_rays.any() else None,
              within_half_m_ped=float((np.abs(pred - gt)[ped_rays] < 0.5).mean()) if ped_rays.any() else None,
              mae_static_only_on_ped=float(np.abs(pred_static - gt)[ped_rays].mean()) if ped_rays.any() else None,
              mae_static_only=float(np.abs(pred_static - gt).mean()))
    print("novel view:", nv, flush=True)
    # range images for the first held-out drive (rays x time)
    sel = (rec_te["drive"] == 0) & (rec_te["t"] <= T_FIELD)
    ns = sel.sum()
    gt_img = rec_te["rng"][sel]                                   # (steps, rays)
    ox1, oy1, tt1 = np.repeat(rec_te["ox"][sel], N_RAYS), np.repeat(rec_te["oy"][sel], N_RAYS), np.repeat(rec_te["t"][sel], N_RAYS)
    pr_img = field.render(ox1, oy1, np.tile(ANG, ns), tt1).reshape(ns, N_RAYS)
    st_img = field.render(ox1, oy1, np.tile(ANG, ns), tt1, use_dyn=False).reshape(ns, N_RAYS)
    t_img = rec_te["t"][sel]

    # ---- 4. what-if replay inside the twin vs the real simulator
    policy = load_policy()
    deltas = np.linspace(-3.0, 3.0, 13)
    speeds = np.array([8.0, 10.0, 12.0, 14.0])
    D, V = np.meshgrid(deltas, speeds)
    D, V = D.ravel(), V.ravel()
    P = scene(V, 0.03)
    real = BatchSim(P, np.random.default_rng(5), t_start=T0 - D)
    twin = TwinSim(field, P, np.random.default_rng(5), T0 - D, D)
    nopd = TwinSim(field, P, np.random.default_rng(5), T0 - D, D)
    nopd.use_dyn = False
    traj = {k: [] for k in ("real", "twin")}
    for k in range(N_STEPS):
        for name, sim in (("real", real), ("twin", twin), ("nopd", nopd)):
            if sim.done.all():
                continue
            obs = sim.observe()
            sim.step(policy_action(policy, obs))
            if name in traj:
                traj[name].append(sim.x.copy())
        if real.done.all() and twin.done.all() and nopd.done.all():
            break
    def pad(L, n=N_STEPS):
        a = np.array(L)
        return np.vstack([a, np.repeat(a[-1:], n - len(a), 0)]) if len(a) < n else a
    wi = dict(
        n_cfg=int(len(D)),
        outcome_agreement=float((real.collided == twin.collided).mean()),
        collision_real=float(real.collided.mean()), collision_twin=float(twin.collided.mean()),
        gap_mae=float(np.abs(np.minimum(real.min_gap, 20) - np.minimum(twin.min_gap, 20)).mean()),
        gap_corr=float(np.corrcoef(np.minimum(real.min_gap, 20), np.minimum(twin.min_gap, 20))[0, 1]),
        xfinal_mae=float(np.abs(real.x - twin.x).mean()),
        pass_agreement=float((real.passed == twin.passed).mean()),
        collision_no_ped_twin=float(nopd.collided.mean()),
        mean_gap_no_ped_twin=float(np.minimum(nopd.min_gap, 20).mean()),
    )
    print("what-if:", wi, flush=True)

    # ---- BEV density maps for the figures
    xs, ys = np.arange(40, 80.01, 0.25), np.arange(-8, 8.01, 0.25)
    bev_t = [2.0, 3.5, 5.0]
    bev = {f"bev_{t}": field.bev(xs, ys, t) for t in bev_t}
    bev["bev_static"] = field.bev(xs, ys, 0.0, "static")
    np.savez(f"{OUT}/twin.npz", gt_img=gt_img, pred_img=pr_img, static_img=st_img, t_img=t_img,
             loss=np.array(field.loss_curve), D=D, V=V,
             gap_real=np.minimum(real.min_gap, 20), gap_twin=np.minimum(twin.min_gap, 20),
             gap_nopd=np.minimum(nopd.min_gap, 20),
             col_real=real.collided, col_twin=twin.collided, col_nopd=nopd.collided,
             traj_real=pad(traj["real"]), traj_twin=pad(traj["twin"]), xs=xs, ys=ys, **bev)
    json.dump(dict(novel_view=nv, what_if=wi, iters=ITERS, n_logged_rays=int(len(rays[0])),
                   n_drives=K, fit_seconds=fit_s), open(f"{OUT}/twin.json", "w"), indent=1)
    print(f"total {time.time() - t0:.0f}s")
