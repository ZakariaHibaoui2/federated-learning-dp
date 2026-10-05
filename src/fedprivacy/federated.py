"""FedAvg with optional client-side DP-SGD and secure aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import LogReg
from .privacy import epsilon, private_sum_gradient
from .secagg import aggregate, mask_update


@dataclass
class FLConfig:
    rounds: int = 40
    local_epochs: int = 2
    batch: int = 32
    lr: float = 0.5
    l2: float = 0.0
    clip: float = 1.0
    sigma: float = 0.0              # DP noise multiplier (0 = no DP)
    secure_aggregation: bool = True
    seed: int = 0


@dataclass
class FLResult:
    model: LogReg
    history: list[float] = field(default_factory=list)
    epsilon: float = float("inf")
    steps_per_client: int = 0


def local_train(global_model: LogReg, x, y, cfg: FLConfig, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    m = global_model.copy()
    steps = 0
    n = len(x)
    steps_per_epoch = max(1, n // cfg.batch)
    for _ in range(cfg.local_epochs):
        if cfg.sigma > 0:
            # DP-SGD: Poisson sampling (each record independently with prob q) - required by the accountant
            q = cfg.batch / n
            for _ in range(steps_per_epoch):
                idx = np.nonzero(rng.random(n) < q)[0]
                g = m.per_example_grads(x[idx], y[idx], cfg.l2) if len(idx) else np.zeros((0, m.size))
                m.theta -= cfg.lr * private_sum_gradient(g, cfg.clip, cfg.sigma, rng) / cfg.batch
                steps += 1
        else:
            order = rng.permutation(n)
            for i in range(0, n, cfg.batch):
                idx = order[i:i + cfg.batch]
                m.theta -= cfg.lr * m.per_example_grads(x[idx], y[idx], cfg.l2).mean(axis=0)
                steps += 1
    return m.theta - global_model.theta, steps


def fedavg(clients: list[tuple[np.ndarray, np.ndarray]], n_features: int, n_classes: int, cfg: FLConfig,
           x_eval=None, y_eval=None) -> FLResult:
    rng = np.random.default_rng(cfg.seed)
    model = LogReg(n_features, n_classes)
    ids = list(range(len(clients)))
    sizes = np.array([len(c[1]) for c in clients], dtype=float)
    weights = sizes / sizes.sum()
    secret = rng.bytes(32)                         # stands in for pairwise key agreement
    result = FLResult(model)
    steps_per_client = 0
    for r in range(cfg.rounds):
        updates = []
        for cid, (x, y) in enumerate(clients):
            delta, steps = local_train(model, x, y, cfg, rng)
            weighted = delta * weights[cid]
            updates.append(mask_update(weighted, cid, ids, r, secret) if cfg.secure_aggregation else weighted)
            if cid == 0:
                steps_per_client += steps
        model.theta += aggregate(updates)
        if x_eval is not None:
            result.history.append(model.accuracy(x_eval, y_eval))
    result.steps_per_client = steps_per_client
    q = min(1.0, cfg.batch / min(sizes))           # worst case: the smallest client has the highest sampling rate
    result.epsilon = epsilon(cfg.sigma, steps_per_client, q) if cfg.sigma > 0 else float("inf")
    return result


def centralized(x, y, n_features, n_classes, epochs: int = 80, lr: float = 0.5, batch: int = 32, l2: float = 0.0,
                sigma: float = 0.0, clip: float = 1.0, seed: int = 0) -> FLResult:
    """Single-party training (optionally DP-SGD) for comparison."""
    cfg = FLConfig(rounds=1, local_epochs=epochs, batch=batch, lr=lr, l2=l2, sigma=sigma, clip=clip, seed=seed,
                   secure_aggregation=False)
    return fedavg([(x, y)], n_features, n_classes, cfg)
