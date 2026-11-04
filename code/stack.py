"""Behaviour tree + perception + monitor + space-time lattice planner + smoothing + tracking + faults.
Pure NumPy, batched over scenarios (same simulator as the rest of the project)."""
import numpy as np
from sim import (BatchSim, DT, N_STEPS, EGO_L, EGO_W, A_MIN, A_MAX, COS, SIN, R_MAX, SENSOR_DX,
                 WALL_Y, PED_Y0, PED_Y1, PC_Y0, PC_Y1, rect_circle_gap)

# ------------------------------------------------------------------ vectorised behaviour tree
FAIL, SUCC, RUN = 0, 1, 2


class Cond:
    def __init__(s, name, fn): s.name, s.fn = name, fn
    def tick(s, c, m): return np.where(s.fn(c), SUCC, FAIL)


class Act:
    def __init__(s, name, fn, ident): s.name, s.fn, s.id = name, fn, ident
    def tick(s, c, m):
        c.a = np.where(m, s.fn(c), c.a); c.chosen = np.where(m, s.id, c.chosen)
        return np.full(len(m), RUN)


class Sel:
    def __init__(s, *k): s.k = k
    def tick(s, c, m):
        res, rem = np.full(len(m), FAIL), m.copy()
        for ch in s.k:
            st = ch.tick(c, rem); d = rem & (st != FAIL)
            res = np.where(d, st, res); rem &= ~d
        return res


class Seq:
    def __init__(s, *k): s.k = k
    def tick(s, c, m):
        res, rem = np.full(len(m), SUCC), m.copy()
        for ch in s.k:
            st = ch.tick(c, rem); f = rem & (st != SUCC)
            res = np.where(f, st, res); rem &= ~f
        return res


V_CRUISE, V_OCCL, V_BAD = 12.0, 8.0, 4.0
T_IN_MARGIN, T_OUT_MARGIN = 0.3, 0.8     # timing margins on the crossing prediction [s]
V_PED_WC = 3.5                           # worst-case start speed of a pedestrian waiting at the curb [m/s]


def a_emerg(c): return -c.abrk
def a_brake(c):
    b = -np.clip(1.3 * c.a_req + 0.3, 0.0, c.abrk)
    return np.minimum(np.where(c.d <= 0, -c.abrk, b), a_cruise(c))
def a_cruise(c):
    vd = np.where(c.bad, V_BAD, np.where(c.occl, V_OCCL, V_CRUISE))
    vd = np.where(c.abrk < 8.0, np.minimum(vd, 6.0), vd)
    return np.clip(vd - c.v, -3.0, 2.5)


def a_yield(c): return np.minimum(a_cruise(c), 0.0)


def build_tree():
    """Priority: 1 emergency brake > 2 brake to stop line > 3 yield (never accelerate in conflict) > 4 cruise (speed-capped near occluder / faults)."""
    return Sel(Seq(Cond("emerg", lambda c: c.emerg), Act("EMERGENCY_BRAKE", a_emerg, 0)),
               Seq(Cond("brake", lambda c: c.brake_need), Act("BRAKE_TO_LINE", a_brake, 1)),
               Seq(Cond("conflict", lambda c: c.conflict), Act("YIELD_NO_ACCEL", a_yield, 3)),
               Act("CRUISE", a_cruise, 2))


class Ctx: pass


def derive(c, sim, est, bad, abrk):
    """Fill the blackboard from ego state + pedestrian estimate (valid, y, vy, xp, pr)."""
    valid, y, vy, xp, pr = est
    x, v = sim.x, sim.v
    xf, rear = x + EGO_L / 2, x - EGO_L / 2
    band = EGO_W / 2 + pr + 0.4
    xs, xc = xp - pr - 0.5, xp + pr + 0.5
    in_band = np.abs(y) < band
    vy = np.where((vy >= -0.05) & (~in_band) & (y > 0), -V_PED_WC, vy)   # curb pedestrian may start at any time
    moving = vy < -0.05
    with np.errstate(divide="ignore", invalid="ignore"):
        t_in = np.where(moving, np.maximum((y - band) / (-vy) - T_IN_MARGIN, 0.0), 0.0)
        t_out = np.where(moving, np.maximum((y + band) / (-vy), 0.0), np.inf)
        gone = np.where(np.isinf(t_out), False, (xf + v * (t_out + T_OUT_MARGIN)) < xs)
    safe = ((~moving) & (~in_band)) | (moving & (y < -band)) | ((rear + v * t_in) > xc) | gone
    c.conflict = valid & ~safe
    c.d = xs - xf
    c.a_req = v ** 2 / (2.0 * np.maximum(c.d, 0.2))
    c.emerg = c.conflict & ((c.a_req > 0.75 * abrk) | (c.d <= 0))
    c.brake_need = c.conflict & ((c.a_req > 0.8) | (c.d <= 0))
    c.occl = (sim.pc_len > 0) & (sim.pc_x0 - xf < 35.0) & (xf < sim.pc_x1)
    c.bad, c.abrk, c.v = bad, abrk, v


def tick(tree, c, B):
    c.a, c.chosen = np.full(B, np.nan), np.full(B, -1)
    tree.tick(c, np.ones(B, bool))
    return c.a


def oracle_est(sim):
    y, vy = sim.ped_state()
    return np.ones(sim.B, bool), y, vy, sim.xp, sim.pr


# ------------------------------------------------------------------ perception from lidar
class Perception:
    K = 10

    def __init__(s, B):
        s.hy, s.ht = np.full((B, s.K), np.nan), np.zeros((B, s.K))
        s.last, s.x, s.y = np.full(B, -9.0), np.zeros(B), np.zeros(B)

    def update(s, sim, lid, t):
        ox, oy = (sim.x + SENSOR_DX)[:, None], sim.ey[:, None]
        X, Y = ox + lid * COS, oy + lid * SIN
        tw = np.where(SIN > 0, (WALL_Y - oy) / SIN, (-WALL_Y - oy) / SIN)
        inpc = ((X > sim.pc_x0[:, None] - 0.5) & (X < sim.pc_x1[:, None] + 0.5) & (Y > PC_Y0 - 0.5)
                & (Y < PC_Y1 + 0.5) & (sim.pc_len[:, None] > 0))           # static map filter
        m = (lid < tw - 1.0) & (lid < R_MAX - 0.5) & ~inpc
        n = m.sum(1)
        cx, cy = (X * m).sum(1) / np.maximum(n, 1), (Y * m).sum(1) / np.maximum(n, 1)
        det = (n >= 1) & (np.abs(cy) <= 6.5)
        s.hy = np.roll(s.hy, -1, 1); s.hy[:, -1] = np.where(det, cy, np.nan)
        s.ht = np.roll(s.ht, -1, 1); s.ht[:, -1] = t
        s.x, s.y, s.last = np.where(det, cx, s.x), np.where(det, cy, s.y), np.where(det, t, s.last)
        ok = ~np.isnan(s.hy); cnt = ok.sum(1)
        hy, ht = np.nan_to_num(s.hy), s.ht
        tm = (ok * ht).sum(1) / np.maximum(cnt, 1); ym = (ok * hy).sum(1) / np.maximum(cnt, 1)
        num = (ok * (ht - tm[:, None]) * (hy - ym[:, None])).sum(1)
        den = (ok * (ht - tm[:, None]) ** 2).sum(1)
        vy = np.where((cnt >= 4) & (den > 1e-6), num / np.maximum(den, 1e-9), np.nan)
        vy = np.where(np.isnan(vy), np.where(s.y > 0, -1.5, 0.0), vy)       # pessimistic default
        valid = (t - s.last) <= 1.0
        y = np.clip(s.y + vy * (t - s.last), PED_Y1, PED_Y0)
        return valid, y, vy, s.x.copy(), np.full(len(vy), 0.6)


class Monitor:
    """Sensor-health (range sanity vs. known walls, frozen frame) and brake-health (achieved vs commanded)."""

    def __init__(s, B):
        s.until, s.prev, s.pv, s.brk = np.zeros(B), None, None, np.zeros(B, bool)

    def update(s, sim, lid, t, a_prev):
        oy = sim.ey[:, None]
        tw = np.where(SIN > 0, (WALL_Y - oy) / SIN, (-WALL_Y - oy) / SIN)
        w = tw < 30
        frac = ((lid > tw + 1.5) & w).sum(1) / w.sum(1)
        frozen = np.zeros(sim.B, bool) if s.prev is None else np.all(lid == s.prev, 1)
        s.until = np.where((frac > 0.3) | frozen, t + 2.0, s.until)
        if s.pv is not None:
            ach = (s.pv - sim.v) / DT
            s.brk |= (a_prev < -3.5) & (ach < 0.7 * (-a_prev)) & (sim.v > 0.5)
        s.prev, s.pv = lid.copy(), sim.v.copy()
        return t < s.until, np.where(s.brk, 4.0, 8.0)


# ------------------------------------------------------------------ fault injection
FAULTS = ["dropout", "stuck", "bias", "latency", "brake"]


class FaultySim(BatchSim):
    """Fault window starts `pre` s before the pedestrian starts to cross (worst timing)."""

    def __init__(s, P, rng, fault=None, pre=1.0, dur=2.0, **kw):
        super().__init__(P, rng, **kw)
        s.fault, s.f0 = fault, s.t_start - pre
        s.f1, s.buf, s.fz = s.f0 + dur, [], None

    def lidar(s):
        r, f = super().lidar(), s.fault
        if f is None: return r
        act = ((s.t >= s.f0) & (s.t < s.f1))[:, None]
        if f == "dropout": return np.where(act, R_MAX, r)
        if f == "stuck":
            s.fz = r.copy() if s.fz is None else np.where(act, s.fz, r)
            return np.where(act, s.fz, r)
        if f == "bias": return np.where((s.t >= s.f0)[:, None], np.minimum(r * 1.25, R_MAX), r)
        if f == "latency":
            s.buf = (s.buf + [r])[-5:]
            return np.where((s.t >= s.f0)[:, None], s.buf[max(0, len(s.buf) - 4)], r)
        return r

    def step(s, a):
        if s.fault == "brake": a = np.where(s.t >= s.f0, np.maximum(a, -4.0), a)
        return super().step(a)


def run_bt(P, rng, fault=None, mode="perc", monitor=False, hist=False):
    B = len(P)
    sim = FaultySim(P, rng, fault)
    per, mon, tree, a_prev = Perception(B), Monitor(B), build_tree(), np.zeros(B)
    H = []
    for k in range(N_STEPS):
        lid, t = (None if mode == "oracle" else sim.lidar()), sim.t
        est = oracle_est(sim) if mode == "oracle" else per.update(sim, lid, t)
        bad, abrk = mon.update(sim, lid, t, a_prev) if monitor else (np.zeros(B, bool), np.full(B, 8.0))
        c = Ctx(); derive(c, sim, est, bad, abrk)
        a = tick(tree, c, B); a_prev = a
        if hist: H.append((sim.x.copy(), sim.v.copy(), c.chosen.copy()))
        sim.step(a)
        if sim.done.all(): break
    return sim, H


# ------------------------------------------------------------------ space-time lattice planner
ACTS, NSUB = np.array([-6.0, -3.0, -1.5, 0.0, 1.0, 2.0]), 5


def lattice_plan(x0, v0, ped, vd, H=6.0, dtp=0.5, margin=1.0):
    """Layered DP over (time, s, v) nodes; edges = constant-accel primitives checked against the
    predicted pedestrian occupancy in space-time. Returns accel profile at DT resolution or None."""
    tau = np.arange(1, NSUB + 1) * dtp / NSUB
    S, V, C = np.array([x0]), np.array([v0]), np.zeros(1)
    par, acc = [], []
    for k in range(int(H / dtp)):
        n = len(S)
        s0, vv, c0 = np.repeat(S, len(ACTS)), np.repeat(V, len(ACTS)), np.repeat(C, len(ACTS))
        a, pid = np.tile(ACTS, n), np.repeat(np.arange(n), len(ACTS))
        tstop = np.where(a < 0, vv / np.maximum(-a, 1e-9), np.inf)
        te = np.minimum(tau[None, :], tstop[:, None])
        pos = s0[:, None] + vv[:, None] * te + 0.5 * a[:, None] * te ** 2
        v1 = np.maximum(vv + a * dtp, 0.0)
        ok = np.ones(len(a), bool)
        if ped is not None:
            valid, y0, vy, xp, pr = ped
            py = np.clip(y0 + vy * (k * dtp + tau), PED_Y1, PED_Y0)[None, :]
            ok = (rect_circle_gap(pos, 0.0, xp, py, pr) > margin).all(1)
        cost = c0 + dtp * ((vd - 0.5 * (vv + v1)) ** 2 + 0.15 * a ** 2)
        if not ok.any(): return None
        idx = np.where(ok)[0]
        idx = idx[np.argsort(cost[idx])]
        key = np.round(pos[idx, -1] / 0.5).astype(int) * 100 + np.round(v1[idx] / 0.5).astype(int)
        idx = idx[np.unique(key, return_index=True)[1]]
        S, V, C = pos[idx, -1], v1[idx], cost[idx]
        par.append(pid[idx]); acc.append(a[idx])
    j, prof = int(np.argmin(C)), []
    for k in reversed(range(len(par))):
        prof.append(acc[k][j]); j = par[k][j]
    return np.repeat(prof[::-1], int(round(dtp / DT)))


def rollout(x0, v0, a):
    v, s = v0, x0
    S, V = [s], [v]
    for ak in a:
        v2 = max(v + ak * DT, 0.0)
        s += 0.5 * (v + v2) * DT; v = v2
        S.append(s); V.append(v)
    return np.array(S), np.array(V)


def smooth(s_lat, lam=2e-5, band=0.4):
    """Jerk-minimising refinement: min |s-s_lat|^2 + lam |s'''|^2, s0,s1 pinned, then clipped to a
    tube around the (collision-checked) lattice path."""
    N = len(s_lat)
    D = np.zeros((N - 3, N))
    for i in range(N - 3): D[i, i:i + 4] = np.array([-1, 3, -3, 1]) / DT ** 3
    W = np.eye(N); W[0, 0] = W[1, 1] = 1e4
    s = np.linalg.solve(W + lam * D.T @ D, W @ s_lat)
    return np.clip(s, s_lat - band, s_lat + band)


def run_planner(P, rng, use_smooth=True, kp=2.0, kd=2.5, mismatch=1.0, lag=0.25, supervisor=False):
    B = len(P)
    sim = BatchSim(P, rng)
    per, tree = Perception(B), build_tree()
    aact, plan, pk = np.zeros(B), [None] * B, np.full(B, -99)
    ex, A, terr, jerk_lat, jerk_sm, tpass = [], [], [], [], [], np.full(B, np.nan)
    for k in range(N_STEPS):
        lid, t = sim.lidar(), sim.t
        est = per.update(sim, lid, t)
        occl = (sim.pc_len > 0) & (sim.pc_x0 - (sim.x + EGO_L / 2) < 35.0) & ((sim.x + EGO_L / 2) < sim.pc_x1)
        acmd = np.full(B, -8.0)
        for i in range(B):
            if sim.done[i]: continue
            if k - pk[i] >= 5 or plan[i] is None:
                ped = tuple(float(e[i]) if np.ndim(e) else float(e) for e in est)
                al = lattice_plan(sim.x[i], sim.v[i], ped if ped[0] else None, 8.0 if occl[i] else 12.0)
                if al is None: plan[i] = None; pk[i] = k; continue
                sl, _ = rollout(sim.x[i], sim.v[i], al)
                ss = smooth(sl) if use_smooth else sl
                vr = np.gradient(ss, DT); ar = np.gradient(vr, DT)
                if use_smooth: jerk_sm.append(np.abs(np.diff(ar)).max() / DT); jerk_lat.append(np.abs(np.diff(al)).max() / DT)
                plan[i], pk[i] = (ss, vr, ar), k
            if plan[i] is None: continue
            j = min(k - pk[i], len(plan[i][0]) - 1)
            sr, vr, ar = plan[i][0][j], plan[i][1][j], plan[i][2][j]
            acmd[i] = ar + kp * (sr - sim.x[i]) + kd * (vr - sim.v[i])
            terr.append(sr - sim.x[i])
        if supervisor:                      # simplex: BT safety layer can only reduce acceleration
            c = Ctx(); derive(c, sim, est, np.zeros(B, bool), np.full(B, 8.0))
            acmd = np.minimum(acmd, tick(tree, c, B))
        acmd = np.clip(acmd, A_MIN, A_MAX)
        aact += (acmd - aact) * DT / lag
        aeff = np.where(aact < 0, aact * mismatch, aact)
        A.append(np.where(sim.done, 0.0, aeff))
        was = sim.passed.copy()
        sim.step(aeff)
        tpass = np.where(np.isnan(tpass) & sim.passed & ~was, sim.t, tpass)
        if sim.done.all(): break
    A = np.array(A)
    return sim, dict(rms_track=float(np.sqrt(np.mean(np.square(terr)))) if terr else 0.0,
                     rms_acc=float(np.sqrt((A ** 2)[A != 0].mean())),
                     jerk95=float(np.percentile(np.abs(np.diff(A, axis=0)) / DT, 95)),
                     jerk_lat=float(np.mean(jerk_lat)) if jerk_lat else 0.0,
                     jerk_sm=float(np.mean(jerk_sm)) if jerk_sm else 0.0,
                     t_pass=float(np.nanmean(tpass)))
