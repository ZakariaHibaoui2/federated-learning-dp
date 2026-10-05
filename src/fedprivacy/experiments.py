"""Experiments: 'hospitals' training a digit classifier (stand-in for sensitive records)."""
from __future__ import annotations

import math

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .attacks import membership_advantage
from .federated import FLConfig, centralized, fedavg


def load(seed: int = 0):
    d = load_digits()
    x = d.data / 16.0
    return train_test_split(x, d.target, test_size=0.4, stratify=d.target, random_state=seed)


def split_iid(x, y, n: int, seed: int = 0):
    idx = np.random.default_rng(seed).permutation(len(y))
    return [(x[p], y[p]) for p in np.array_split(idx, n)]


def split_non_iid(x, y, n: int, classes_per_client: int = 2, seed: int = 0):
    """Each client holds mostly a few classes (like hospitals with different specialities)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(y + rng.uniform(0, 0.1, len(y)))      # sort by label
    shards = np.array_split(order, n * classes_per_client)
    perm = rng.permutation(len(shards))
    return [(x[np.concatenate([shards[s] for s in perm[i::n]])], y[np.concatenate([shards[s] for s in perm[i::n]])])
            for i in range(n)]


def run(seed: int = 0, clients: int = 5) -> dict:
    x_tr, x_te, y_tr, y_te = load(seed)
    d, k = x_tr.shape[1], 10
    out: dict = {}

    cen = centralized(x_tr, y_tr, d, k, epochs=60, seed=seed)
    out["centralized"] = cen.model.accuracy(x_te, y_te)
    for name, parts in (("fedavg_iid", split_iid(x_tr, y_tr, clients, seed)),
                        ("fedavg_non_iid", split_non_iid(x_tr, y_tr, clients, seed=seed))):
        r = fedavg(parts, d, k, FLConfig(rounds=30, seed=seed), x_te, y_te)
        out[name] = r.history[-1]
        out[name + "_curve"] = r.history

    dp = []
    for sigma, rounds, lr in ((1.0, 30, 0.5), (1.5, 20, 1.0), (2.0, 20, 1.0), (3.0, 15, 1.5)):
        r = fedavg(split_iid(x_tr, y_tr, clients, seed), d, k,
                   FLConfig(rounds=rounds, local_epochs=1, batch=16, sigma=sigma, clip=1.0, lr=lr, seed=seed),
                   x_te, y_te)
        dp.append({"sigma": sigma, "rounds": rounds, "accuracy": r.history[-1], "epsilon": r.epsilon,
                   "steps": r.steps_per_client})
    out["dp"] = dp

    out["mia"] = mia_curve(x_tr, y_tr, x_te, y_te)
    return out


def mia_curve(x_tr, y_tr, x_te, y_te, seeds: int = 3, n: int = 200) -> list[dict]:
    """Membership-inference advantage vs privacy budget (small training sets make leakage measurable)."""
    rows = []
    for sigma, epochs in ((0.0, 300), (1.0, 30), (1.5, 30), (3.0, 20), (6.0, 15)):
        adv, acc, eps = [], [], float("inf")
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            i, o = rng.permutation(len(y_tr))[:n], rng.permutation(len(y_te))[:n]
            r = centralized(x_tr[i], y_tr[i], x_tr.shape[1], 10, epochs=epochs, lr=1.0, batch=16 if sigma else 32,
                            sigma=sigma, clip=1.0, seed=seed)
            adv.append(membership_advantage(r.model, x_tr[i], y_tr[i], x_te[o], y_te[o])["advantage"])
            acc.append(r.model.accuracy(x_te, y_te))
            eps = r.epsilon
        bound = 1.0 if math.isinf(eps) else (math.exp(eps) - 1) / (math.exp(eps) + 1)
        rows.append({"sigma": sigma, "epsilon": float(eps), "test_accuracy": float(np.mean(acc)),
                     "mia_advantage": float(np.mean(adv)), "dp_bound": float(bound)})
    return rows
