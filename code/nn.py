"""Tiny neural-network toolkit in pure NumPy (manual back-propagation + Adam).

Karpathy-style: no deep-learning framework, every gradient is written out, so the
whole project runs anywhere NumPy runs (laptop, Colab, Jupyter).
"""
import numpy as np


def _act(z, kind):
    if kind == "relu":
        return np.maximum(z, 0.0)
    if kind == "silu":
        return z / (1.0 + np.exp(-z))
    raise ValueError(kind)


def _dact(z, kind):
    if kind == "relu":
        return (z > 0).astype(z.dtype)
    s = 1.0 / (1.0 + np.exp(-z))
    return s * (1.0 + z * (1.0 - s))


class MLP:
    """Fully-connected network: sizes=[in, h1, ..., out]. Linear output layer."""

    def __init__(self, sizes, rng, act="relu", out_scale=1.0):
        self.sizes, self.act = list(sizes), act
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            W = rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i])
            self.W.append(W.astype(np.float32))
            self.b.append(np.zeros(sizes[i + 1], np.float32))
        self.W[-1] *= out_scale
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        h = np.asarray(x, np.float32)
        self.xs, self.zs = [], []
        L = len(self.W)
        for i in range(L):
            self.xs.append(h)
            z = h @ self.W[i] + self.b[i]
            if i < L - 1:
                self.zs.append(z)
                h = _act(z, self.act)
            else:
                h = z
        return h

    # --------------------------------------------------------------- backward
    def backward(self, dy):
        """Back-propagate dL/d(output); stores parameter grads, returns dL/d(input)."""
        L = len(self.W)
        self.gW, self.gb = [None] * L, [None] * L
        g = np.asarray(dy, np.float32)
        for i in reversed(range(L)):
            self.gW[i] = self.xs[i].T @ g
            self.gb[i] = g.sum(0)
            g = g @ self.W[i].T
            if i > 0:
                g = g * _dact(self.zs[i - 1], self.act)
        return g

    # ------------------------------------------------------------------- Adam
    def step(self, lr, b1=0.9, b2=0.999, eps=1e-8, clip=None):
        self.t += 1
        params, grads = self.W + self.b, self.gW + self.gb
        if clip is not None:
            n = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            if n > clip:
                grads = [g * (clip / n) for g in grads]
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    # ------------------------------------------------------------------- misc
    def save(self, path):
        np.savez(path, **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    def load(self, path):
        d = np.load(path)
        for i in range(len(self.W)):
            self.W[i][...] = d[f"W{i}"]
            self.b[i][...] = d[f"b{i}"]


class Adam:
    """Adam for a single dense array (used for the NeRF feature grids)."""

    def __init__(self, p, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.b1, self.b2, self.eps = p, b1, b2, eps
        self.m, self.v, self.t = np.zeros_like(p), np.zeros_like(p), 0

    def step(self, g, lr):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        self.p -= lr * mh / (np.sqrt(vh) + self.eps)
