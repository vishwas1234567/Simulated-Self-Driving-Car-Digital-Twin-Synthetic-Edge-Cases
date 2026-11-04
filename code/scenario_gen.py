"""Generative AI for edge cases: a small DDPM over scenario vectors + failure guidance.

1.  Train a denoising-diffusion model on 'logged' everyday scenarios (it learns the prior).
2.  Train a noise-aware surrogate  g(x_t, t)  that predicts how close the *current driving
    policy* comes to the pedestrian (minimum gap) in that scenario.
3.  Sample with classifier guidance:  eps' = eps + lam * sqrt(1 - abar_t) * grad g
    which tilts the prior towards scenarios where the policy has small gaps / collides.
"""
import numpy as np
from nn import MLP
from sim import LO, HI, PNAMES

T_DIFF = 200
BETAS = np.linspace(1e-4, 0.05, T_DIFF)
ALPHAS = 1.0 - BETAS
ABAR = np.cumprod(ALPHAS)
D = len(PNAMES)
TFREQ = np.array([1, 2, 4, 8, 16, 32, 64, 128], float)


def normalize(P):
    return (P - LO) / (HI - LO) * 2.0 - 1.0


def denormalize(X):
    return (np.clip(X, -1, 1) + 1.0) / 2.0 * (HI - LO) + LO


def t_embed(t):
    u = (np.asarray(t, float) / T_DIFF)[:, None] * TFREQ[None, :] * np.pi
    return np.concatenate([np.sin(u), np.cos(u)], 1).astype(np.float32)


def noisy(x0, t, rng):
    eps = rng.standard_normal(x0.shape).astype(np.float32)
    ab = ABAR[t][:, None]
    return (np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps).astype(np.float32), eps


class ScenarioDiffusion:
    def __init__(self, seed=0, hidden=128):
        self.rng = np.random.default_rng(seed)
        self.eps_net = MLP([D + 16, hidden, hidden, hidden, D], self.rng, act="silu", out_scale=0.5)
        self.surr = None
        self.loss_curve, self.surr_curve = [], []

    # ------------------------------------------------------------- prior training
    def fit_prior(self, P, iters=6000, bs=512, lr=1e-3):
        X = normalize(P).astype(np.float32)
        for i in range(iters):
            x0 = X[self.rng.integers(0, len(X), bs)]
            t = self.rng.integers(0, T_DIFF, bs)
            xt, eps = noisy(x0, t, self.rng)
            pred = self.eps_net.forward(np.concatenate([xt, t_embed(t)], 1))
            err = pred - eps
            self.eps_net.backward(2.0 * err / (bs * D))
            self.eps_net.step(lr * (0.5 * (1 + np.cos(np.pi * i / iters)) * 0.95 + 0.05), clip=5.0)
            if i % 100 == 0:
                self.loss_curve.append(float((err ** 2).mean()))

    # --------------------------------------------------------- surrogate training
    def fit_surrogate(self, P, y, iters=6000, bs=512, lr=1e-3, hidden=128):
        """y in [0,1]: normalised minimum gap of the current policy (0 = collision)."""
        self.surr = MLP([D + 16, hidden, hidden, 1], self.rng, act="silu", out_scale=0.3)
        X = normalize(P).astype(np.float32)
        y = y.astype(np.float32)
        for i in range(iters):
            idx = self.rng.integers(0, len(X), bs)
            t = self.rng.integers(0, T_DIFF, bs)
            xt, _ = noisy(X[idx], t, self.rng)
            pred = self.surr.forward(np.concatenate([xt, t_embed(t)], 1))[:, 0]
            err = pred - y[idx]
            self.surr.backward((2.0 * err / bs)[:, None])
            self.surr.step(lr * (0.5 * (1 + np.cos(np.pi * i / iters)) * 0.95 + 0.05), clip=5.0)
            if i % 100 == 0:
                self.surr_curve.append(float((err ** 2).mean()))

    # ------------------------------------------------------------------ sampling
    def sample(self, n, guidance=0.0, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, D)).astype(np.float32)
        for t in reversed(range(T_DIFF)):
            tt = np.full(n, t)
            inp = np.concatenate([x, t_embed(tt)], 1)
            eps = self.eps_net.forward(inp)
            if guidance > 0 and self.surr is not None:
                self.surr.forward(inp)
                dinp = self.surr.backward(np.ones((n, 1), np.float32))[:, :D]
                eps = eps + guidance * np.sqrt(1 - ABAR[t]) * dinp
            x0 = np.clip((x - np.sqrt(1 - ABAR[t]) * eps) / np.sqrt(ABAR[t]), -1, 1)
            ab_prev = ABAR[t - 1] if t > 0 else 1.0
            mean = (np.sqrt(ab_prev) * BETAS[t] / (1 - ABAR[t])) * x0 \
                + (np.sqrt(ALPHAS[t]) * (1 - ab_prev) / (1 - ABAR[t])) * x
            if t > 0:
                var = BETAS[t] * (1 - ab_prev) / (1 - ABAR[t])
                x = (mean + np.sqrt(var) * rng.standard_normal((n, D))).astype(np.float32)
            else:
                x = mean.astype(np.float32)
        return denormalize(x)
