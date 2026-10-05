"""Membership inference: was this record in the training set? (loss-threshold attack, Yeom et al., 2018)

Models fit their training records better than unseen ones, so a low loss suggests "member". We report the
best balanced attack accuracy over all thresholds and the membership advantage = TPR - FPR at that threshold
(0 = no leakage, 1 = perfect inference). DP-SGD bounds this advantage by e^epsilon - 1 (and by 1).
"""
from __future__ import annotations

import numpy as np

from .model import LogReg


def membership_advantage(model: LogReg, x_in, y_in, x_out, y_out) -> dict:
    l_in, l_out = model.per_example_loss(x_in, y_in), model.per_example_loss(x_out, y_out)
    thresholds = np.unique(np.concatenate([l_in, l_out]))
    best_adv, best_t = 0.0, 0.0
    for t in thresholds:
        adv = float((l_in <= t).mean() - (l_out <= t).mean())
        if adv > best_adv:
            best_adv, best_t = adv, float(t)
    return {"advantage": best_adv, "attack_accuracy": 0.5 + best_adv / 2, "threshold": best_t,
            "mean_loss_members": float(l_in.mean()), "mean_loss_non_members": float(l_out.mean())}
