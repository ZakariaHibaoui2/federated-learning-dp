"""Multinomial logistic regression with per-example gradients (needed for DP-SGD clipping)."""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LogReg:
    """Parameters are one flat vector so they can be averaged, clipped, masked and noised uniformly."""

    def __init__(self, n_features: int, n_classes: int):
        self.d, self.k = n_features, n_classes
        self.theta = np.zeros((n_features + 1) * n_classes)

    @property
    def size(self) -> int:
        return self.theta.size

    def _split(self, theta=None):
        t = self.theta if theta is None else theta
        w = t[: self.d * self.k].reshape(self.d, self.k)
        return w, t[self.d * self.k:]

    def proba(self, x: np.ndarray, theta=None) -> np.ndarray:
        w, b = self._split(theta)
        return softmax(x @ w + b)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.proba(x).argmax(axis=1)

    def accuracy(self, x, y) -> float:
        return float((self.predict(x) == y).mean())

    def per_example_loss(self, x, y, theta=None) -> np.ndarray:
        p = self.proba(x, theta)
        return -np.log(p[np.arange(len(y)), y] + 1e-12)

    def per_example_grads(self, x: np.ndarray, y: np.ndarray, l2: float = 0.0) -> np.ndarray:
        """(n, size) matrix: the gradient of each example's loss, as DP-SGD needs."""
        p = self.proba(x)
        p[np.arange(len(y)), y] -= 1.0                              # dL/dlogits
        gw = np.einsum("nd,nk->ndk", x, p).reshape(len(x), -1)
        g = np.concatenate([gw, p], axis=1)
        if l2:
            g[:, : self.d * self.k] += l2 * self.theta[: self.d * self.k]
        return g

    def copy(self) -> LogReg:
        m = LogReg(self.d, self.k)
        m.theta = self.theta.copy()
        return m
