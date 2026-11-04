"""A small, fully vectorised 2-D driving simulator (pedestrian-crossing world).

World (bird's-eye view, ego drives along +x):

    y=+7 ===================== building line =====================
    y=+5.5   [ped waits here]            (hidden behind parked car)
    y=+4.0       [parked car]  y=+2.2
    y= 0     ego ------------------------------------------>
    y=-5.5   [ped ends here]
    y=-7 ===================== building line =====================

A pedestrian crosses the road at x = dist. The ego car (longitudinal control only)
has a 64-ray lidar. Everything is batched over scenarios, so thousands of
episodes run at once in NumPy.

Scenario vector  z = (v0, dist, ped_speed, tau, occl, ped_r, dropout, noise)
    v0        initial ego speed [m/s]
    dist      x-position of the crossing [m]
    ped_speed pedestrian walking speed [m/s]
    tau       the pedestrian reaches the lane centre tau seconds after the ego
              would arrive there at constant speed (tau ~ 0  ->  collision course)
    occl      parked-car occluder size (0 = none, 1 = 6 m long)
    ped_r     pedestrian footprint radius [m]   (0.3 normal, up to 1.0 = odd costume)
    dropout   probability that a lidar ray on the pedestrian is lost (dark clothes)
    noise     lidar range noise std [m]
"""
import numpy as np

DT = 0.1
T_MAX = 18.0
N_STEPS = int(T_MAX / DT)
EGO_L, EGO_W = 4.5, 1.9
A_MIN, A_MAX = -8.0, 3.0
N_RAYS = 64
FOV = np.deg2rad(120.0)
R_MAX = 40.0
SENSOR_DX = 2.0
WALL_Y = 7.0
PED_Y0, PED_Y1 = 5.5, -5.5
PC_Y0, PC_Y1, PC_GAP = 2.2, 4.0, 3.0

PNAMES = ["v0", "dist", "ped_speed", "tau", "occl", "ped_r", "dropout", "noise"]
LO = np.array([6.0, 35.0, 0.8, -6.0, 0.0, 0.25, 0.0, 0.02])
HI = np.array([14.0, 70.0, 3.5, 6.0, 1.0, 1.0, 0.8, 0.30])
ANG = np.linspace(-FOV / 2, FOV / 2, N_RAYS)
COS, SIN = np.cos(ANG), np.sin(ANG)


# ----------------------------------------------------------------- scenario priors
def sample_normal(rng, n):
    """'Everyday' driving distribution: hazards and odd pedestrians are rare."""
    v0 = rng.uniform(7, 13, n)
    dist = rng.uniform(35, 70, n)
    s = rng.uniform(0.8, 2.2, n)
    crit = rng.random(n) < 0.15
    tau = np.where(crit, rng.uniform(-3, 3, n), rng.choice([-1.0, 1.0], n) * rng.uniform(3, 6, n))
    occl = rng.uniform(0, 1, n)
    big = rng.random(n) < 0.05
    pr = np.where(big, rng.uniform(0.4, 1.0, n), rng.uniform(0.25, 0.4, n))
    drop = np.where(rng.random(n) < 0.10, rng.uniform(0, 0.3, n), 0.0)
    noise = rng.uniform(0.02, 0.10, n)
    return np.stack([v0, dist, s, tau, occl, pr, drop, noise], 1)


def sample_uniform(rng, n):
    return rng.uniform(LO, HI, (n, len(LO)))


# ------------------------------------------------------------------------ geometry
def ped_traj(s, t_start, t):
    y = np.clip(PED_Y0 - s * (t - t_start), PED_Y1, PED_Y0)
    moving = (t > t_start) & (y > PED_Y1)
    return y, np.where(moving, -s, 0.0)


def ped_t_start(P):
    v0, dist, s, tau = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    return dist / v0 + tau - PED_Y0 / s


def rect_circle_gap(ex, ey, px, py, pr):
    dx = np.maximum(np.abs(px - ex) - EGO_L / 2, 0.0)
    dy = np.maximum(np.abs(py - ey) - EGO_W / 2, 0.0)
    return np.hypot(dx, dy) - pr


def open_loop_gap(P):
    """Minimum ego-pedestrian gap if the ego simply keeps its speed (no reaction)."""
    t = np.arange(0, T_MAX, DT)[None, :]
    v0 = P[:, 0:1]
    ts = ped_t_start(P)[:, None]
    y, _ = ped_traj(P[:, 2:3], ts, t)
    g = rect_circle_gap(v0 * t, 0.0, P[:, 1:2], y, P[:, 5:6])
    return np.maximum(g.min(1), 0.0)


# --------------------------------------------------------------------- batch simulator
class BatchSim:
    def __init__(self, P, rng, t_start=None, ey=0.0, v0=None):
        self.P, self.rng = np.asarray(P, float), rng
        B = len(self.P)
        self.B = B
        self.xp, self.s, self.pr = self.P[:, 1].copy(), self.P[:, 2].copy(), self.P[:, 5].copy()
        self.drop, self.noise = self.P[:, 6].copy(), self.P[:, 7].copy()
        self.t_start = ped_t_start(self.P) if t_start is None else np.asarray(t_start, float)
        self.pc_len = np.where(self.P[:, 4] > 0.1, 6.0 * self.P[:, 4], 0.0)
        self.pc_x1 = self.xp - PC_GAP
        self.pc_x0 = self.pc_x1 - self.pc_len
        self.x = np.zeros(B)
        self.v = (self.P[:, 0] if v0 is None else np.asarray(v0, float)).copy()
        self.ey = np.broadcast_to(np.asarray(ey, float), (B,)).copy()
        self.t, self.k = 0.0, 0
        self.done = np.zeros(B, bool)
        self.collided = np.zeros(B, bool)
        self.passed = np.zeros(B, bool)
        self.min_gap = np.full(B, np.inf)
        self.prev = None
        self.lid = None

    def ped_state(self, t=None):
        return ped_traj(self.s, self.t_start, self.t if t is None else t)

    # --------------------------------------------------------------------- lidar
    def lidar(self):
        B = self.B
        ox, oy = (self.x + SENSOR_DX)[:, None], self.ey[:, None]
        dx, dy = COS[None, :], SIN[None, :]
        tw = np.where(dy > 0, (WALL_Y - oy) / dy, (-WALL_Y - oy) / dy)
        # parked car (axis-aligned box, slab method)
        t1, t2 = (self.pc_x0[:, None] - ox) / dx, (self.pc_x1[:, None] - ox) / dx
        u1, u2 = (PC_Y0 - oy) / dy, (PC_Y1 - oy) / dy
        tmin = np.maximum(np.minimum(t1, t2), np.minimum(u1, u2))
        tmax = np.minimum(np.maximum(t1, t2), np.maximum(u1, u2))
        hit = (tmax >= np.maximum(tmin, 0)) & (self.pc_len[:, None] > 0)
        tp = np.where(hit, np.maximum(tmin, 0.0), np.inf)
        # pedestrian (circle)
        py, _ = self.ped_state()
        cx, cy = self.xp[:, None] - ox, py[:, None] - oy
        b = dx * cx + dy * cy
        c0 = cx ** 2 + cy ** 2 - (self.pr[:, None] ** 2)
        disc = b * b - c0
        tc = np.where(disc >= 0, b - np.sqrt(np.maximum(disc, 0)), np.inf)
        tc = np.where(tc > 0, tc, np.where(c0 < 0, 0.0, np.inf))
        lost = self.rng.random((B, N_RAYS)) < self.drop[:, None]
        tc = np.where(lost, np.inf, tc)
        r = np.minimum(np.minimum(tw, tp), np.minimum(tc, R_MAX))
        r = r + self.rng.standard_normal((B, N_RAYS)) * self.noise[:, None]
        return np.clip(r, 0.0, R_MAX)

    def observe(self):
        self.lid = self.lidar()
        if self.prev is None:
            self.prev = self.lid.copy()
        obs = make_obs(self.lid, self.prev, self.v)
        self.prev = self.lid
        return obs

    # ---------------------------------------------------------------------- step
    def step(self, a):
        act = ~self.done
        a = np.clip(a, A_MIN, A_MAX)
        v_new = np.maximum(self.v + a * DT, 0.0)
        self.v = np.where(act, v_new, self.v)
        self.x = np.where(act, self.x + self.v * DT, self.x)
        self.t += DT
        self.k += 1
        y, _ = self.ped_state()
        gap = rect_circle_gap(self.x, self.ey, self.xp, y, self.pr)
        self.min_gap = np.where(act, np.minimum(self.min_gap, gap), self.min_gap)
        col = act & (gap <= 0)
        self.collided |= col
        psd = act & ((self.x - EGO_L / 2) > self.xp + self.pr + 1.0)
        self.passed |= psd
        self.done |= col | psd
        return act


def make_obs(lid, prev, v):
    """Policy input: min-pooled lidar (32), its temporal difference (32), speed."""
    a = lid.reshape(len(lid), N_RAYS // 2, 2).min(2) / R_MAX
    b = prev.reshape(len(prev), N_RAYS // 2, 2).min(2) / R_MAX
    return np.concatenate([a, (a - b) * 10.0, (v / 15.0)[:, None]], 1).astype(np.float32)
