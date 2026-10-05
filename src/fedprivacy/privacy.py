"""Differential privacy: per-example clipping + Gaussian noise, and a Renyi-DP accountant.

Accountant (the same analysis used by Opacus / TensorFlow Privacy):
* Each DP-SGD step is a **Poisson-subsampled Gaussian mechanism**: every record joins the batch independently
  with probability q, gradients are clipped to norm C and N(0, (sigma*C)^2) noise is added to their sum.
* For integer orders alpha >= 2 its RDP is bounded by (Mironov, Talwar & Zhang, 2019):
      RDP(alpha) = 1/(alpha-1) * log( sum_k  C(alpha,k) (1-q)^(alpha-k) q^k exp((k^2 - k) / (2 sigma^2)) )
* RDP composes additively over T steps and converts to (epsilon, delta)-DP via
      epsilon = min_alpha [ T * RDP(alpha) + log(1/delta) / (alpha - 1) ]          (Mironov, 2017)
With q = 1 (full batch) the formula reduces to the plain Gaussian mechanism, alpha / (2 sigma^2).
"""
from __future__ import annotations

import math

import numpy as np

ORDERS = np.arange(2, 257)


def clip_rows(g: np.ndarray, c: float) -> np.ndarray:
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    return g * np.minimum(1.0, c / np.maximum(norms, 1e-12))


def private_sum_gradient(per_example: np.ndarray, clip: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Clip every example to norm `clip`, sum, add N(0, (sigma*clip)^2) noise."""
    total = clip_rows(per_example, clip).sum(axis=0) if len(per_example) else np.zeros(per_example.shape[1])
    return total + rng.normal(0.0, sigma * clip, per_example.shape[1])


def _log_add(a: float, b: float) -> float:
    hi, lo = max(a, b), min(a, b)
    return hi if lo == -math.inf else hi + math.log1p(math.exp(lo - hi))


def rdp_subsampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    if q == 0:
        return 0.0
    if q >= 1.0:
        return alpha / (2 * sigma ** 2)
    log_a = -math.inf
    for k in range(alpha + 1):
        term = (math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
                + k * math.log(q) + (alpha - k) * math.log1p(-q) + (k * k - k) / (2 * sigma ** 2))
        log_a = _log_add(log_a, term)
    return log_a / (alpha - 1)


def epsilon(sigma: float, steps: int, q: float = 1.0, delta: float = 1e-5) -> float:
    """(epsilon, delta) after `steps` Poisson-subsampled Gaussian steps with sampling rate q."""
    if sigma <= 0:
        return float("inf")
    best = float("inf")
    for a in ORDERS:
        rdp = steps * rdp_subsampled_gaussian(q, sigma, int(a))
        best = min(best, rdp + math.log(1 / delta) / (a - 1))
    return best
