"""Builds every figure and the LaTeX number macros from results/*.json|npz."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from sim import (BatchSim, ped_traj, ANG, R_MAX, SENSOR_DX, EGO_L, EGO_W, PC_Y0, PC_Y1, WALL_Y,
                 PED_Y0, PED_Y1, N_RAYS, PNAMES)
from driving import expert_action

from paths import RES, FIG
R, F = str(RES), str(FIG)

# validated categorical slots (light mode): blue, orange, aqua, yellow
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ARMC = {"A": BLUE, "B": ORANGE, "C": AQUA, "D": YELLOW}
ARML = {"A": "A  normal only", "B": "B  + guided diffusion", "C": "C  + unguided diffusion",
        "D": "D  + uniform random"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "axes.titlesize": 10,
    "axes.titleweight": "normal", "axes.titlecolor": INK})


def save(fig, name):
    fig.savefig(f"{F}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{F}/{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- simulator overview
def fig_sim_overview():
    P = np.array([[10.0, 60.0, 1.6, 0.0, 0.8, 0.35, 0.0, 0.03]])
    rng = np.random.default_rng(3)
    sim = BatchSim(P, rng, t_start=np.array([1.2]))
    snaps = {}
    for k in range(80):
        if k in (0, 22, 36):
            snaps[k] = (sim.x[0], sim.v[0], sim.t, sim.lidar()[0].copy(), sim.ped_state()[0][0])
        sim.step(expert_action(sim))
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.4))
    for ax, (k, (x, v, t, lid, py)) in zip(axes, snaps.items()):
        ax.set_xlim(x - 8, 66); ax.set_ylim(-8, 8); ax.set_aspect("equal"); ax.grid(False)
        ax.axhline(WALL_Y, color=INK2, lw=2); ax.axhline(-WALL_Y, color=INK2, lw=2)
        ax.add_patch(Rectangle((60 - 3 - 4.8, PC_Y0), 4.8, PC_Y1 - PC_Y0, color="#9a9890"))
        ax.plot([60, 60], [PED_Y0, PED_Y1], ls=":", color=INK2, lw=1)
        ox, oy = x + SENSOR_DX, 0.0
        for a, r in zip(ANG[::2], lid[::2]):
            ax.plot([ox, ox + r * np.cos(a)], [oy, oy + r * np.sin(a)], color=BLUE, lw=0.4, alpha=0.55)
        ax.add_patch(Rectangle((x - EGO_L / 2, -EGO_W / 2), EGO_L, EGO_W, color=BLUE, zorder=3))
        ax.add_patch(Circle((60, py), 0.35, color=ORANGE, zorder=4))
        ax.set_title(f"t = {t:.1f} s   speed = {v:.1f} m/s", loc="left", color=INK2, fontsize=9)
        ax.set_ylabel("y [m]")
    for ax in axes:
        ax.set_xlabel("x [m]")
    fig.tight_layout()
    save(fig, "sim_overview")


# ----------------------------------------------------------------------- experiment 2
def fig_experiments():
    J = json.load(open(f"{R}/experiments.json"))
    runs = J["runs"]
    arms = ["A", "B", "C", "D"]
    sets = [("normal", "Normal test set"), ("hazard", "Hazard test set"), ("hard", "Hard-perception subset")]
    stats = {a: {s: np.array([r["res"][a][s]["coll"] for r in runs]) * 100 for s, _ in sets} for a in arms}
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    w, gap = 0.19, 0.012
    for i, a in enumerate(arms):
        xs = np.arange(len(sets)) + (i - 1.5) * (w + gap)
        m = [stats[a][s].mean() for s, _ in sets]
        e = [stats[a][s].std(ddof=1) for s, _ in sets]
        ax.bar(xs, m, w, color=ARMC[a], label=ARML[a], yerr=e, capsize=2.5,
               error_kw=dict(ecolor=INK2, lw=1))
        for x_, m_, e_ in zip(xs, m, e):
            ax.text(x_, m_ + e_ + 0.6, f"{m_:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([n for _, n in sets])
    ax.set_ylabel("collision rate [%]"); ax.set_ylim(0, 30); ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper left", fontsize=8)
    ax.set_title(f"Collision rate (mean +/- std over {len(runs)} seeds; equal training-set size)", loc="left")
    save(fig, "exp_collision_rates")

    # guidance sweep
    sw = runs[0]["extra"]["sweep"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot([s["lam"] for s in sw], [100 * s["A_coll"] for s in sw], "-o", color=ORANGE, lw=2, ms=5,
            label="baseline policy A")
    ax.plot([s["lam"] for s in sw], [100 * s["expert_coll"] for s in sw], "-s", color=BLUE, lw=2, ms=5,
            label="privileged expert")
    ax.set_xlabel("guidance strength $\\lambda$"); ax.set_ylabel("collision rate on generated set [%]")
    ax.legend(loc="upper left"); ax.set_ylim(0, 60)
    ax.set_title("Guided sampling finds failures that are avoidable", loc="left")
    save(fig, "exp_guidance_sweep")

    # parameter marginals
    d = np.load(f"{R}/fig_data_seed0.npz")
    keys = [("tau", "arrival offset $\\tau$ [s]"), ("ped_speed", "pedestrian speed [m/s]"),
            ("ped_r", "pedestrian radius [m]"), ("v0", "ego speed $v_0$ [m/s]")]
    fig, axs = plt.subplots(1, 4, figsize=(8.2, 2.5))
    for ax, (k, lab) in zip(axs, keys):
        j = PNAMES.index(k)
        lo, hi = np.percentile(np.concatenate([d["normal"][:, j], d["guided"][:, j]]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 26)
        for arr, c, ls, name in [(d["normal"], BLUE, "-", "everyday data"),
                                 (d["free"], AQUA, "--", "unguided"), (d["guided"], ORANGE, "-", "guided")]:
            ax.hist(arr[:, j], bins=bins, density=True, histtype="step", color=c, lw=1.8, ls=ls, label=name)
        ax.set_xlabel(lab, fontsize=8); ax.set_yticks([])
    axs[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Generated scenario parameters vs. everyday data (seed 0)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "exp_marginals")

    # scatter of tau vs ped speed
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True, sharex=True)
    for ax, (S, coll, ttl) in zip(axs, [(d["normal"], d["coll_normal"], "everyday scenarios"),
                                       (d["guided"], d["coll_guided"], "guided-diffusion scenarios")]):
        ok = ~coll.astype(bool)
        ax.scatter(S[ok, 3], S[ok, 2], s=5, color=BLUE, alpha=0.45, lw=0, label="A avoids collision")
        ax.scatter(S[~ok, 3], S[~ok, 2], s=7, color=ORANGE, alpha=0.9, lw=0, label="A collides")
        ax.set_title(f"{ttl}  ({100 * coll.mean():.0f}% failures)", loc="left", fontsize=9)
        ax.set_xlabel("arrival offset $\\tau$ [s]")
    axs[0].set_ylabel("pedestrian speed [m/s]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, markerscale=2.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "exp_scatter")

    # training curves
    fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.4))
    for ax, (k, ttl) in zip(axs, [("dagger_loss", "DAgger policy (MSE)"), ("prior_loss", "diffusion prior ($\\epsilon$-MSE)"),
                                  ("surr_loss", "failure surrogate (MSE)")]):
        y = d[k]
        ax.plot(np.arange(len(y)) * 100, y, color=BLUE, lw=1.6)
        ax.set_yscale("log"); ax.set_title(ttl, loc="left", fontsize=9); ax.set_xlabel("iteration")
    fig.tight_layout()
    save(fig, "exp_training_curves")

    # numbers for the report
    def ms(a, s, key="coll", scale=100):
        v = np.array([r["res"][a][s][key] for r in runs]) * scale
        return v.mean(), v.std(ddof=1)
    rows = []
    for a in arms:
        n, h, hd = ms(a, "normal"), ms(a, "hazard"), ms(a, "hard")
        g, pas = ms(a, "hazard", "gap", 1), ms(a, "normal", "passed")
        rows.append((a, n, h, hd, g, pas))
    diag = {k: np.mean([r["diag"][k][m] for r in runs]) for k in runs[0]["diag"] for m in ["A_coll"]}
    diag = {k: {m: float(np.mean([r["diag"][k][m] for r in runs])) for m in runs[0]["diag"][k]}
            for k in runs[0]["diag"]}
    return J, rows, diag


# ---------------------------------------------------------------------- experiment 1
def fig_twin():
    T = json.load(open(f"{R}/twin.json"))
    d = np.load(f"{R}/twin.npz")
    t, ang = d["t_img"], np.rad2deg(ANG)
    # range images
    fig, axs = plt.subplots(1, 4, figsize=(8.4, 2.9), sharey=True)
    ext = [t[0], t[-1], ang[0], ang[-1]]
    for ax, (img, ttl, cm, vmax) in zip(axs, [(d["gt_img"], "ground-truth lidar", "Blues_r", R_MAX),
                                              (d["pred_img"], "NeRF render", "Blues_r", R_MAX),
                                              (d["static_img"], "static field only", "Blues_r", R_MAX),
                                              (np.abs(d["pred_img"] - d["gt_img"]), "|error| [m]", "Oranges", 5)]):
        im = ax.imshow(img.T, aspect="auto", origin="lower", extent=ext, cmap=cm, vmin=0, vmax=vmax)
        ax.set_title(ttl, loc="left", fontsize=9); ax.set_xlabel("time [s]"); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    axs[0].set_ylabel("ray angle [deg]")
    fig.tight_layout()
    save(fig, "twin_range_images")

    # BEV density maps with ground-truth pedestrian
    xs, ys = d["xs"], d["ys"]
    panels = [("bev_static", "static field", None)] + [(f"bev_{tt}", f"static + dynamic, t = {tt} s", tt) for tt in (2.0, 3.5, 5.0)]
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 4.6), sharex=True, sharey=True)
    for ax, (key, ttl, tt) in zip(axs.ravel(), panels):
        ax.imshow(np.minimum(d[key], 25), origin="lower", cmap="Blues", vmin=0, vmax=25, aspect="equal",
                  extent=[xs[0] - 0.125, xs[-1] + 0.125, ys[0] - 0.125, ys[-1] + 0.125])
        ax.grid(False); ax.set_title(ttl, loc="left", fontsize=9)
        if tt is not None:
            py = ped_traj(np.array(1.6), np.array(1.2), tt)[0]
            ax.add_patch(Circle((60, float(py)), 0.9, fill=False, ec=ORANGE, lw=1.4))
    for ax in axs[1]:
        ax.set_xlabel("x [m]")
    for ax in axs[:, 0]:
        ax.set_ylabel("y [m]")
    fig.tight_layout()
    save(fig, "twin_bev_density")

    # loss
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.plot(np.arange(len(d["loss"])) * 50, d["loss"], color=BLUE, lw=1.6)
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("Huber loss on range [m]")
    ax.set_title("Field training", loc="left")
    save(fig, "twin_loss")

    # what-if comparison
    D, V = d["D"], d["V"]
    deltas, speeds = np.unique(D), np.unique(V)
    shape = (len(speeds), len(deltas))
    gr, gt_ = d["gap_real"].reshape(shape), d["gap_twin"].reshape(shape)
    cr, ct = d["col_real"].reshape(shape), d["col_twin"].reshape(shape)
    gr, gt_ = np.maximum(gr, 0), np.maximum(gt_, 0)
    fig, axs = plt.subplots(1, 5, figsize=(9.6, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 0.05, 0.32, 0.9], wspace=0.3))
    axs[3].axis("off")
    ext = [deltas[0] - 0.25, deltas[-1] + 0.25, 0, len(speeds)]
    for ax, (g, c, ttl) in zip(axs[:2], [(gr, cr, "real simulator"), (gt_, ct, "NeRF digital twin")]):
        im = ax.imshow(g, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=5, extent=ext)
        ax.grid(False)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if c[i, j]:
                    ax.text(deltas[j], i + 0.5, "\u00d7", ha="center", va="center", color=ORANGE, fontsize=13, fontweight="bold")
        ax.set_yticks(np.arange(len(speeds)) + 0.5); ax.set_yticklabels([f"{s:.0f}" for s in speeds])
        ax.set_xlabel("pedestrian shift $\\Delta$ [s]"); ax.set_title(ttl, loc="left", fontsize=9)
    axs[0].set_ylabel("ego speed $v_0$ [m/s]")
    fig.colorbar(im, cax=axs[2], label="min gap [m]  (\u00d7 = collision)")
    ax = axs[4]
    ax.scatter(np.maximum(d["gap_real"], 0), np.maximum(d["gap_twin"], 0), s=14, color=BLUE, alpha=0.8, lw=0)
    ax.plot([0, 5], [0, 5], color=INK2, lw=1, ls="--")
    ax.set_xlim(-0.2, 5); ax.set_ylim(-0.2, 5)
    ax.set_xlabel("min gap, real [m]"); ax.set_ylabel("min gap, twin [m]")
    ax.set_title(f"r = {T['what_if']['gap_corr']:.2f}", loc="left", fontsize=9)
    save(fig, "twin_whatif")

    # trajectories
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    n = d["traj_real"].shape[0]
    tt = (np.arange(n) + 1) * 0.1
    for dlt, c in [(-2.0, YELLOW), (0.0, ORANGE), (2.0, BLUE)]:
        i = int(np.where((D == dlt) & (V == 12.0))[0][0])
        ax.plot(tt, d["traj_real"][:, i], color=c, lw=2.2, label=f"real, $\\Delta$={dlt:+.0f} s")
        ax.plot(tt, d["traj_twin"][:, i], color=INK, lw=1.0, ls="--")
    ax.plot([], [], color=INK, lw=1, ls="--", label="digital twin")
    ax.axhline(60 - 0.35 - 2.25, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("time [s]"); ax.set_ylabel("ego position x [m]"); ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Same policy, real vs. twin ($v_0$ = 12 m/s)", loc="left")
    save(fig, "twin_trajectories")
    return T


def macros(J, rows, diag, T):
    L = []
    def mac(n, v):
        L.append(f"\\newcommand{{\\{n}}}{{{v}}}")
    names = {"A": "A", "B": "B", "C": "C", "D": "D"}
    for a, n, h, hd, g, pas in rows:
        mac(f"coll{a}normal", f"{n[0]:.1f} \\pm {n[1]:.1f}")
        mac(f"coll{a}hazard", f"{h[0]:.1f} \\pm {h[1]:.1f}")
        mac(f"coll{a}hard", f"{hd[0]:.1f} \\pm {hd[1]:.1f}")
        mac(f"gap{a}hazard", f"{g[0]:.2f} \\pm {g[1]:.2f}")
        mac(f"pass{a}normal", f"{pas[0]:.1f} \\pm {pas[1]:.1f}")
    ex = J["expert"]
    mac("expertnormal", f"{100 * ex['normal']['coll']:.1f}"); mac("experthazard", f"{100 * ex['hazard']['coll']:.1f}")
    mac("experthard", f"{100 * ex['hard']['coll']:.1f}"); mac("expertgap", f"{ex['hazard']['gap']:.2f}")
    for k, v in diag.items():
        mac(f"diag{k}Acoll", f"{100 * v['A_coll']:.1f}"); mac(f"diag{k}Excoll", f"{100 * v['expert_coll']:.1f}")
        mac(f"diag{k}OL", f"{100 * v['ol_hazard']:.1f}")
    nv, wi = T["novel_view"], T["what_if"]
    mac("nvMAE", f"{nv['mae']:.2f}"); mac("nvRMSE", f"{nv['rmse']:.2f}"); mac("nvWithin", f"{100 * nv['within_half_m']:.1f}")
    mac("nvMAEstatic", f"{nv['mae_static_only']:.2f}")
    mac("nvPedRays", f"{nv['ped_rays']}"); mac("nvAllRays", f"{nv['n_rays']}")
    if nv["mae_ped"] is not None:
        mac("nvMAEped", f"{nv['mae_ped']:.2f}"); mac("nvWithinPed", f"{100 * nv['within_half_m_ped']:.1f}")
        mac("nvMAEstaticped", f"{nv['mae_static_only_on_ped']:.2f}")
    mac("wiAgree", f"{100 * wi['outcome_agreement']:.1f}"); mac("wiCollReal", f"{100 * wi['collision_real']:.1f}")
    mac("wiCollTwin", f"{100 * wi['collision_twin']:.1f}"); mac("wiGapMAE", f"{wi['gap_mae']:.2f}")
    mac("wiGapCorr", f"{wi['gap_corr']:.2f}"); mac("wiXMAE", f"{wi['xfinal_mae']:.2f}")
    mac("wiCollNoPed", f"{100 * wi['collision_no_ped_twin']:.1f}"); mac("wiNcfg", f"{wi['n_cfg']}")
    mac("nerfRays", f"{T['n_logged_rays']}"); mac("nerfIters", f"{T['iters']}"); mac("nerfFit", f"{T['fit_seconds']:.0f}")
    mac("nSeeds", f"{len(J['seeds'])}"); mac("nNormal", f"{J['n_normal']}"); mac("nGen", f"{J['n_gen']}")
    mac("guidanceLam", f"{J['guidance']:.0f}"); mac("nHard", f"{J['n_hard']}")
    open(f"{R}/macros.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    fig_sim_overview()
    J, rows, diag = fig_experiments()
    print("experiment figures done")
    if os.path.exists(f"{R}/twin.json"):
        T = fig_twin()
        macros(J, rows, diag, T)
        print("twin figures + macros done")
