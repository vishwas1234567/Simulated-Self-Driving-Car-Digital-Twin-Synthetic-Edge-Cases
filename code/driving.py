"""Expert controller, DAgger imitation learning and evaluation (all NumPy)."""
import numpy as np
from nn import MLP
from sim import (BatchSim, N_STEPS, EGO_L, EGO_W, A_MIN, A_MAX, make_obs)

OBS_DIM = 65
ACT_SCALE = 8.0


# ------------------------------------------------------------------ privileged expert
def expert_action(sim):
    """Rule-based expert that sees the true pedestrian state (privileged information)."""
    x, v = sim.x, sim.v
    xf, rear = x + EGO_L / 2, x - EGO_L / 2
    y, vy = sim.ped_state()
    band = EGO_W / 2 + sim.pr + 0.4
    xs = sim.xp - sim.pr - 0.5          # stop line for the ego front
    xc = sim.xp + sim.pr + 0.5          # line the ego rear must have passed
    moving = vy < 0
    in_band = np.abs(y) < band
    with np.errstate(divide="ignore", invalid="ignore"):
        t_enter = np.where(moving, np.maximum((y - band) / (-vy), 0.0), 0.0)
        t_leave = np.where(moving, np.maximum((y + band) / (-vy), 0.0), np.inf)
    static_out = (~moving) & (~in_band)
    past = moving & (y < -band)
    clears_before = (rear + v * t_enter) > xc
    with np.errstate(invalid="ignore"):
        gone_first = np.where(np.isinf(t_leave), False, (xf + v * (t_leave + 0.5)) < xs)
    conflict = ~(static_out | past | clears_before | gone_first)
    d = xs - xf
    a_req = v ** 2 / (2.0 * np.maximum(d, 0.2))
    brake = -np.clip(1.3 * a_req + 0.3, 0.0, 8.0)
    brake = np.where(d <= 0, -8.0, brake)
    v_des = np.full_like(v, 12.0)
    ahead = (sim.pc_len > 0) & (sim.pc_x0 - xf < 35.0) & (xf < sim.pc_x1)
    v_des = np.where(ahead, 8.0, v_des)          # defensive driving near occluders
    a_c = np.clip(1.0 * (v_des - v), -3.0, 2.5)
    return np.where(conflict & ((a_req > 0.8) | (d <= 0)), np.minimum(a_c, brake), a_c)


# ------------------------------------------------------------------------ policy
def policy_action(net, obs):
    return np.clip(net.forward(obs)[:, 0] * ACT_SCALE, A_MIN, A_MAX)


def collect(P, net, beta, rng, act_noise=0.4):
    """Roll out (a mixture of expert and learner) and label every state with the expert."""
    sim = BatchSim(P, rng)
    O, Y = [], []
    for _ in range(N_STEPS):
        obs = sim.observe()
        a_exp = expert_action(sim)
        if net is None:
            a = a_exp
        else:
            use_exp = rng.random(sim.B) < beta
            a = np.where(use_exp, a_exp, policy_action(net, obs))
        act = ~sim.done
        O.append(obs[act])
        Y.append(a_exp[act])
        sim.step(a + rng.standard_normal(sim.B) * act_noise)
        if sim.done.all():
            break
    return np.concatenate(O), np.concatenate(Y).astype(np.float32)


def fit(net, O, Y, iters, rng, lr=2e-3, bs=512):
    losses = []
    for i in range(iters):
        idx = rng.integers(0, len(O), bs)
        pred = net.forward(O[idx])[:, 0]
        err = pred - Y[idx] / ACT_SCALE
        net.backward((2.0 * err / bs)[:, None])
        net.step(lr * (0.5 * (1 + np.cos(np.pi * i / iters)) * 0.9 + 0.1), clip=5.0)
        if i % 100 == 0:
            losses.append(float((err ** 2).mean()))
    return losses


def train_policy(P, seed, rounds=3, iters=2500, hidden=64, verbose=False):
    """DAgger: round 0 = expert only, later rounds mix in the learner (beta 0.5, 0)."""
    rng = np.random.default_rng(seed)
    net = MLP([OBS_DIM, hidden, hidden, 1], rng, out_scale=0.1)
    betas = [1.0, 0.5, 0.0, 0.0][:rounds]
    O, Y, curve = None, None, []
    for r, beta in enumerate(betas):
        o, y = collect(P, None if r == 0 else net, beta, rng)
        O = o if O is None else np.concatenate([O, o])
        Y = y if Y is None else np.concatenate([Y, y])
        curve += fit(net, O, Y, iters, rng)
        if verbose:
            print(f"  dagger round {r}: {len(O)} samples, loss {curve[-1]:.4f}")
    net.curve = curve
    return net


# ---------------------------------------------------------------------- evaluation
def evaluate(P, controller, seed=0, t_start=None):
    """controller: 'expert' or an MLP policy. Returns per-scenario metrics."""
    rng = np.random.default_rng(seed)
    sim = BatchSim(P, rng, t_start=t_start)
    for _ in range(N_STEPS):
        obs = sim.observe()
        a = expert_action(sim) if isinstance(controller, str) else policy_action(controller, obs)
        sim.step(a)
        if sim.done.all():
            break
    return dict(collided=sim.collided.copy(), passed=sim.passed.copy(),
                min_gap=np.minimum(sim.min_gap, 20.0), x_final=sim.x.copy())
