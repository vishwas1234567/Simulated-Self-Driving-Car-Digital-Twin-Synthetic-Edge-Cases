"""Figures + LaTeX macros for the planning/safety stack (reads results/stack.json, embedded.json)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_figures as mf                                   # shared style + colours
from stack import lattice_plan, rollout, smooth
from sim import DT

from paths import RES, FIG
R, F = str(RES), str(FIG)
S, E = json.load(open(f"{R}/stack.json")), json.load(open(f"{R}/embedded.json"))
BLUE, ORANGE, AQUA, YELLOW, INK2 = mf.BLUE, mf.ORANGE, mf.AQUA, mf.YELLOW, mf.INK2

# ---- 1. controller comparison
names = list(S["compare"]); cols = [BLUE, ORANGE, AQUA, YELLOW, "#6250d6"]
vals = [100 * S["compare"][n].get("coll", S["compare"][n].get("collided")) for n in names]
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.bar(range(len(names)), vals, color=cols[:len(names)], width=0.62)
for i, v in enumerate(vals): ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(" (", "\n(").replace("+PD+", "+PD\n+") for n in names], fontsize=7.5)
ax.set_ylabel("collision rate [%]"); ax.grid(axis="x", visible=False)
ax.set_title("Controllers on the same 100 hazard scenarios", loc="left")
mf.save(fig, "stack_compare")

# ---- 2. fault injection
fs = list(S["faults"]); lab = {"None": "nominal", "dropout": "lidar\ndropout", "stuck": "frozen\nframe", "bias": "range\nbias", "latency": "latency\n0.3 s", "brake": "brake\nfade"}
x = np.arange(len(fs)); fig, ax = plt.subplots(figsize=(7.4, 3.0))
for j, (k, c, n) in enumerate([("plain", BLUE, "BT, no monitor"), ("monitor", ORANGE, "BT + health monitor")]):
    v = [100 * S["faults"][f][k]["coll"] for f in fs]
    ax.bar(x + (j - 0.5) * 0.36, v, 0.34, color=c, label=n)
    for xi, vi in zip(x + (j - 0.5) * 0.36, v): ax.text(xi, vi + 0.4, f"{vi:.1f}", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([lab[f] for f in fs], fontsize=8); ax.set_ylabel("collision rate [%]")
ax.set_ylim(0, 11); ax.legend(loc="upper center", fontsize=8, ncol=2); ax.grid(axis="x", visible=False)
ax.set_title("Fault injection (400 hazard scenarios)", loc="left")
mf.save(fig, "stack_faults")

# ---- 3. lattice vs optimised trajectory
ped = (True, 3.0, -1.6, 60.0, 0.6); al = lattice_plan(20.0, 10.0, ped, 8.0)
sl, vl = rollout(20.0, 10.0, al); ss = smooth(sl); vs = np.gradient(ss, DT); as_ = np.gradient(vs, DT)
t = np.arange(len(sl)) * DT
fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.6))
axs[0].plot(t, sl, color=BLUE, lw=2, label="lattice"); axs[0].plot(t, ss, color=ORANGE, lw=1.4, ls="--", label="optimised")
axs[0].set_ylabel("position s [m]"); axs[0].legend(fontsize=7.5)
axs[1].plot(t[:-1], al if len(al) == len(t) - 1 else al, color=BLUE, lw=2); axs[1].plot(t, as_, color=ORANGE, lw=1.4)
axs[1].set_ylabel("acceleration [m/s$^2$]")
axs[2].plot(t[:-2], np.abs(np.diff(np.append(al, al[-1])))[:len(t) - 2] / DT, color=BLUE, lw=2)
axs[2].plot(t[:-1], np.abs(np.diff(as_)) / DT, color=ORANGE, lw=1.4); axs[2].set_ylabel("|jerk| [m/s$^3$]")
for a in axs: a.set_xlabel("time [s]")
fig.tight_layout(); mf.save(fig, "stack_trajectory")

# ---- macros
M = []
def mac(n, v): M.append(f"\\newcommand{{\\{n}}}{{{v}}}")
P = lambda x: f"{100 * x:.1f}"
mac("treeStates", S["tree"]["n_states"]); mac("treeProps", len(S["tree"]["props"])); mac("bmcN", S["bmc"]["n"]); mac("bmcViol", S["bmc"]["violations"])
for k, key in [("stackA", "Learned policy A"), ("stackBT", "BT (lidar perception)"), ("stackBTo", "BT (oracle perception)"),
               ("stackPl", "Lattice+opt+PD"), ("stackPs", "Lattice+opt+PD+BT supervisor")]:
    d = S["compare"][key]; mac(k + "Coll", P(d.get("coll", d.get("collided")))); mac(k + "Pass", P(d["passed"]))
    if "gap" in d: mac(k + "Gap", f"{d['gap']:.2f}")
d = S["compare"]["Lattice+opt+PD+BT supervisor"]
for k in ("rms_track", "rms_acc", "jerk95", "jerk_lat", "jerk_sm", "t_pass"): mac("pl" + k.replace("_", "").replace("95", "NF"), f"{d[k]:.2f}")
mac("btFullHz", P(S["bt_full_hazard"]["coll"])); mac("btFullNorm", P(S["bt_full_normal"]["coll"]))
for k, key in [("trFF", "feed-forward only | brake gain 1.0"), ("trFFm", "feed-forward only | brake gain 0.8"), ("trPD", "PD + feed-forward | brake gain 1.0"), ("trPDm", "PD + feed-forward | brake gain 0.8")]:
    mac(k + "Rms", f"{S['tracking'][key]['rms_track']:.3f}"); mac(k + "Coll", P(S["tracking"][key]["coll"])); mac(k + "Jerk", f"{S['tracking'][key]['jerk95']:.1f}")
mac("nsJerk", f"{S['no_smooth']['jerk95']:.1f}"); mac("nsColl", P(S["no_smooth"]["coll"]))
for f in S["faults"]:
    for k in ("plain", "monitor"): mac(f"f{f}{k}", P(S["faults"][f][k]["coll"]))
NM = {"float32": "Fl", "int8": "Iq"}
for t in ("float32", "int8"):
    n_ = NM[t]
    mac(f"emb{n_}ns", f"{E[t + '-O2']['ns_per_inference'] / 1000:.1f}"); mac(f"emb{n_}flash", f"{E[t + '-Os']['text_bytes'] / 1024:.1f}")
    mac(f"emb{n_}err", f"{E[t + '_max_abs_err_vs_numpy']:.1e}"); mac(f"emb{n_}n", P(E['closed_loop_' + t]['normal'])); mac(f"emb{n_}h", P(E['closed_loop_' + t]['hazard']))
mac("embParams", E["params"]); mac("embMacs", E["macs"])
open(f"{R}/macros_stack.tex", "w").write("\n".join(M) + "\n"); print("ok")
