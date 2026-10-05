"""Secure aggregation with pairwise additive masks (simplified Bonawitz et al., 2017).

Every pair of clients (i, j) derives a shared random mask m_ij from a common seed. Client i adds +m_ij,
client j adds -m_ij. Each masked update looks like noise to the server, but the masks cancel in the sum,
so the server learns only the aggregate. (A production protocol adds key agreement and dropout recovery.)
"""
from __future__ import annotations

import hashlib

import numpy as np


def _pair_seed(i: int, j: int, round_id: int, secret: bytes) -> int:
    h = hashlib.sha256(secret + f"{min(i, j)}-{max(i, j)}-{round_id}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def mask_update(update: np.ndarray, client: int, clients: list[int], round_id: int, secret: bytes,
                scale: float = 100.0) -> np.ndarray:
    masked = update.copy()
    for other in clients:
        if other == client:
            continue
        m = np.random.default_rng(_pair_seed(client, other, round_id, secret)).normal(0, scale, update.shape)
        masked += m if client < other else -m
    return masked


def aggregate(masked_updates: list[np.ndarray]) -> np.ndarray:
    return np.sum(masked_updates, axis=0)
