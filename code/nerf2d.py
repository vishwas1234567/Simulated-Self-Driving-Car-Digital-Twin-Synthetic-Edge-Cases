"""NeRF-style 'digital twin' of a logged drive, in 2-D, written from scratch in NumPy.

A neural radiance field normally maps (x, y, z, view) -> (colour, density) and is trained
by differentiable volume rendering of camera pixels. Here the sensor is a lidar and the
world is 2-D, so the field predicts only *density*, and a ray is 'rendered' into an
expected hit range:

    D(ray) = sum_i  w_i r_i + T_N * R_max ,   w_i = T_{i-1} (1 - exp(-sigma_i delta_i))

The scene is split, as in D-NeRF / dynamic NeRFs, into
    sigma(x, y, t) = sigma_static(x, y)  +  sigma_dynamic(x, y, t)
Each part is a learnable feature grid (bilinear / trilinear interpolation, Instant-NGP
style) followed by a small MLP decoder. Because time is an explicit input of the dynamic
part, we can re-render the same drive with the pedestrian shifted in time (a 'what-if').
All gradients (volume rendering, interpolation, decoders) are derived by hand.
"""
import numpy as np
from nn import MLP, Adam
from sim import (R_MAX, ANG, COS, SIN, N_RAYS, BatchSim, SENSOR_DX, N_STEPS, make_obs)

X0, X1, Y0, Y1 = 0.0, 100.0, -8.0, 8.0
CELL, TCELL, C = 0.25, 0.2, 4
SCALE = 20.0
N_SAMP = 128


def softplus(z):
    return np.logaddexp(0.0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class NeuralField:
    def __init__(self, x_ped, t_max=12.0, seed=0):
        rng = np.random.default_rng(seed)
        self.rng, self.t_max = rng, t_max
        self.nx, self.ny = int((X1 - X0) / CELL) + 1, int((Y1 - Y0) / CELL) + 1
        self.wx0, self.wx1 = x_ped - 10.0, x_ped + 10.0                 # dynamic window
        self.nxd, self.nt = int((self.wx1 - self.wx0) / CELL) + 1, int(t_max / TCELL) + 1
        self.Fs = (rng.standard_normal((self.nx * self.ny, C)) * 0.1).astype(np.float32)
        self.Fd = (rng.standard_normal((self.nxd * self.ny * self.nt, C)) * 0.1).astype(np.float32)
        self.dec_s = MLP([C, 16, 1], rng, out_scale=0.5)
        self.dec_d = MLP([C, 16, 1], rng, out_scale=0.5)
        self.dec_s.b[-1][:] = -6.0
        self.dec_d.b[-1][:] = -8.0
        self.opt_s, self.opt_d = Adam(self.Fs), Adam(self.Fd)
        self.loss_curve = []

    # --------------------------------------------------------------- interpolation
    def _static_idx(self, x, y):
        fx = np.clip((x - X0) / CELL, 0, self.nx - 1 - 1e-6)
        fy = np.clip((y - Y0) / CELL, 0, self.ny - 1 - 1e-6)
        ix, iy = fx.astype(np.int64), fy.astype(np.int64)
        ax, ay = (fx - ix).astype(np.float32), (fy - iy).astype(np.float32)
        idx, w = [], []
        for dx in (0, 1):
            for dy in (0, 1):
                idx.append((ix + dx) * self.ny + (iy + dy))
                w.append((ax if dx else 1 - ax) * (ay if dy else 1 - ay))
        mask = ((x >= X0) & (x <= X1) & (y >= Y0) & (y <= Y1)).astype(np.float32)
        return np.stack(idx, 1), np.stack(w, 1), mask

    def _dyn_idx(self, x, y, t):
        fx = np.clip((x - self.wx0) / CELL, 0, self.nxd - 1 - 1e-6)
        fy = np.clip((y - Y0) / CELL, 0, self.ny - 1 - 1e-6)
        ft = np.clip(t / TCELL, 0, self.nt - 1 - 1e-6)
        ix, iy, it = fx.astype(np.int64), fy.astype(np.int64), ft.astype(np.int64)
        ax, ay, at = (fx - ix).astype(np.float32), (fy - iy).astype(np.float32), (ft - it).astype(np.float32)
        idx, w = [], []
        for dx in (0, 1):
            for dy in (0, 1):
                for dt in (0, 1):
                    idx.append(((ix + dx) * self.ny + (iy + dy)) * self.nt + (it + dt))
                    w.append((ax if dx else 1 - ax) * (ay if dy else 1 - ay) * (at if dt else 1 - at))
        mask = ((x >= self.wx0) & (x <= self.wx1) & (y >= Y0) & (y <= Y1)).astype(np.float32)
        return np.stack(idx, 1), np.stack(w, 1), mask

    # ------------------------------------------------------------------ density
    def density(self, x, y, t, keep=False, use_dyn=True):
        """sigma(x,y,t) for flat arrays of points. keep=True stores caches for backward."""
        out = {}
        si, sw, sm = self._static_idx(x, y)
        fs = (self.Fs[si] * sw[..., None]).sum(1)
        os_ = self.dec_s.forward(fs)[:, 0]
        sig = SCALE * softplus(os_) * sm
        out["s"] = (si, sw, sm, os_)
        if use_dyn:
            di, dw, dm = self._dyn_idx(x, y, t)
            fd = (self.Fd[di] * dw[..., None]).sum(1)
            od = self.dec_d.forward(fd)[:, 0]
            sig_d = SCALE * softplus(od) * dm
            sig = sig + sig_d
            out["d"] = (di, dw, dm, od)
            out["sig_d"] = sig_d
        if keep:
            self.cache = out
        return sig

    def density_backward(self, dsig, dsig_d_extra=None):
        si, sw, sm, os_ = self.cache["s"]
        g = (dsig * SCALE * sigmoid(os_) * sm)[:, None].astype(np.float32)
        dfs = self.dec_s.backward(g)
        gFs = np.zeros_like(self.Fs)
        for c in range(C):
            gFs[:, c] = np.bincount(si.ravel(), weights=(sw * dfs[:, c:c + 1]).ravel(), minlength=len(self.Fs))
        di, dw, dm, od = self.cache["d"]
        dd = dsig if dsig_d_extra is None else dsig + dsig_d_extra
        g = (dd * SCALE * sigmoid(od) * dm)[:, None].astype(np.float32)
        dfd = self.dec_d.backward(g)
        gFd = np.zeros_like(self.Fd)
        for c in range(C):
            gFd[:, c] = np.bincount(di.ravel(), weights=(dw * dfd[:, c:c + 1]).ravel(), minlength=len(self.Fd))
        return gFs, gFd

    # ---------------------------------------------------------------- rendering
    def _ray_points(self, ox, oy, ang, jitter):
        n = N_SAMP
        edges = np.linspace(0.0, R_MAX, n + 1)
        r0 = edges[:-1][None, :] + (self.rng.random((len(ox), n)) if jitter else 0.5) * (R_MAX / n)
        r = np.broadcast_to(r0, (len(ox), n))
        px = ox[:, None] + r * np.cos(ang)[:, None]
        py = oy[:, None] + r * np.sin(ang)[:, None]
        delta = np.concatenate([r[:, 1:] - r[:, :-1], (R_MAX - r[:, -1:])], 1)
        return r, px, py, delta

    def render(self, ox, oy, ang, t, jitter=False, use_dyn=True, chunk=40000):
        R = len(ox)
        per = max(1, chunk // N_SAMP)
        out = np.empty(R)
        for s in range(0, R, per):
            sl = slice(s, s + per)
            r, px, py, delta = self._ray_points(ox[sl], oy[sl], ang[sl], jitter)
            tt = np.broadcast_to(t[sl][:, None], px.shape)
            sig = self.density(px.ravel(), py.ravel(), tt.ravel(), use_dyn=use_dyn).reshape(px.shape)
            E = np.exp(-np.cumsum(sig * delta, 1))
            g = np.concatenate([r[:, 1:] - r[:, :-1], R_MAX - r[:, -1:]], 1)
            out[sl] = r[:, 0] + (E * g).sum(1)
        return out

    # ----------------------------------------------------------------- training
    def fit(self, rays, iters=4000, bs=192, lr_grid=2e-2, lr_dec=2e-3, l1_dyn=0.5, verbose=True):
        ox, oy, ang, t, gt = rays
        N = len(ox)
        for it in range(iters):
            decay = 0.5 * (1 + np.cos(np.pi * it / iters)) * 0.95 + 0.05
            idx = self.rng.integers(0, N, bs)
            r, px, py, delta = self._ray_points(ox[idx], oy[idx], ang[idx], True)
            tt = np.broadcast_to(t[idx][:, None], px.shape)
            sig = self.density(px.ravel(), py.ravel(), tt.ravel(), keep=True).reshape(px.shape)
            E = np.exp(-np.cumsum(sig * delta, 1))
            g = np.concatenate([r[:, 1:] - r[:, :-1], R_MAX - r[:, -1:]], 1)
            D = r[:, 0] + (E * g).sum(1)
            err = D - gt[idx]
            loss = float(np.where(np.abs(err) < 1, 0.5 * err ** 2, np.abs(err) - 0.5).mean())
            dD = np.clip(err, -1, 1) / bs
            tail = np.cumsum((E * g)[:, ::-1], 1)[:, ::-1]               # sum_{i>=k} E_i g_i
            dsig = (-delta * tail * dD[:, None]).ravel()
            reg = np.full(dsig.shape, l1_dyn / dsig.size)
            gFs, gFd = self.density_backward(dsig, reg)
            self.opt_s.step(gFs, lr_grid * decay)
            self.opt_d.step(gFd, lr_grid * decay)
            self.dec_s.step(lr_dec * decay, clip=5.0)
            self.dec_d.step(lr_dec * decay, clip=5.0)
            if it % 50 == 0:
                self.loss_curve.append(loss)
            if verbose and it % 500 == 0:
                print(f"    nerf iter {it:5d}  huber loss {loss:.4f}", flush=True)

    # ------------------------------------------------------------------ BEV map
    def bev(self, xs, ys, t, which="both"):
        X, Y = np.meshgrid(xs, ys)
        tt = np.full(X.size, t)
        sig = self.density(X.ravel(), Y.ravel(), tt, use_dyn=(which != "static"))
        if which == "dyn":
            sig = self.density(X.ravel(), Y.ravel(), tt)
            sig = sig - self.density(X.ravel(), Y.ravel(), tt, use_dyn=False)
        return sig.reshape(X.shape)


# =========================================================================== logging
def run_logged_drives(P, t_start, ey, v0, seed):
    """Drive the privileged expert through a fixed scene and record all lidar returns."""
    from driving import expert_action
    rng = np.random.default_rng(seed)
    sim = BatchSim(P, rng, t_start=t_start, ey=ey, v0=v0)
    rec = dict(ox=[], oy=[], t=[], rng=[], drive=[], step=[])
    for k in range(N_STEPS):
        lid = sim.lidar()
        act = ~sim.done
        rec["ox"].append((sim.x + SENSOR_DX)[act]); rec["oy"].append(sim.ey[act])
        rec["t"].append(np.full(act.sum(), sim.t)); rec["rng"].append(lid[act])
        rec["drive"].append(np.where(act)[0]); rec["step"].append(np.full(act.sum(), k))
        sim.step(expert_action(sim))
        if sim.done.all():
            break
    return {k: np.concatenate(v) for k, v in rec.items()}


def to_rays(rec, t_max):
    keep = rec["t"] <= t_max
    n = keep.sum()
    ox = np.repeat(rec["ox"][keep], N_RAYS)
    oy = np.repeat(rec["oy"][keep], N_RAYS)
    t = np.repeat(rec["t"][keep], N_RAYS)
    ang = np.tile(ANG, n)
    gt = rec["rng"][keep].ravel()
    return ox, oy, ang, t, gt


# ======================================================================= twin sim
class TwinSim(BatchSim):
    """Closed-loop simulation whose lidar comes from the NeRF instead of the simulator.
    The ped is shifted in time by +delta (field queried at t+delta); the true shifted
    pedestrian is still used for scoring (collision / gap)."""

    def __init__(self, field, P, rng, t_start, deltas, **kw):
        super().__init__(P, rng, t_start=t_start, **kw)
        self.field, self.deltas = field, np.asarray(deltas, float)

    def lidar(self):
        B = self.B
        ox = np.repeat(self.x + SENSOR_DX, N_RAYS)
        oy = np.repeat(self.ey, N_RAYS)
        t = np.repeat(np.clip(self.t + self.deltas, 0, self.field.t_max), N_RAYS)
        ang = np.tile(ANG, B)
        r = self.field.render(ox, oy, ang, t, use_dyn=getattr(self, "use_dyn", True)).reshape(B, N_RAYS)
        r = r + self.rng.standard_normal(r.shape) * self.noise[:, None]
        return np.clip(r, 0, R_MAX)
