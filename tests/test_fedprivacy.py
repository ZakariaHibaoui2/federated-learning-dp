import math

import numpy as np
import pytest

from fedprivacy.attacks import membership_advantage
from fedprivacy.cli import main
from fedprivacy.experiments import load, split_iid, split_non_iid
from fedprivacy.federated import FLConfig, centralized, fedavg
from fedprivacy.model import LogReg
from fedprivacy.privacy import clip_rows, epsilon, rdp_subsampled_gaussian
from fedprivacy.secagg import aggregate, mask_update


@pytest.fixture(scope="module")
def data():
    return load(0)


def test_per_example_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    m = LogReg(5, 3)
    m.theta = rng.normal(0, 0.3, m.size)
    x, y = rng.uniform(0, 1, (4, 5)), np.array([0, 1, 2, 1])
    g = m.per_example_grads(x, y)
    h = 1e-6
    for j in (0, 7, m.size - 1):
        t = m.theta.copy()
        t[j] += h
        lp = m.per_example_loss(x, y, t)
        t[j] -= 2 * h
        lm = m.per_example_loss(x, y, t)
        assert np.allclose(g[:, j], (lp - lm) / (2 * h), atol=1e-5)


def test_clipping_bounds_norm():
    g = np.random.default_rng(1).normal(0, 10, (50, 20))
    assert np.linalg.norm(clip_rows(g, 1.5), axis=1).max() <= 1.5 + 1e-9


def test_accountant_properties():
    assert rdp_subsampled_gaussian(1.0, 2.0, 8) == pytest.approx(8 / (2 * 4))   # full batch = Gaussian mechanism
    assert rdp_subsampled_gaussian(0.01, 1.0, 8) < rdp_subsampled_gaussian(0.1, 1.0, 8)   # amplification
    assert epsilon(1.0, 100, 0.05) < epsilon(1.0, 1000, 0.05)                           # composition grows
    assert epsilon(2.0, 500, 0.05) < epsilon(1.0, 500, 0.05)                            # more noise, less eps
    assert epsilon(0, 10) == math.inf
    assert 3 < epsilon(1.1, 10000, 0.01) < 9                                            # same ballpark as Opacus


def test_secure_aggregation_masks_cancel():
    rng = np.random.default_rng(2)
    updates = [rng.normal(0, 1, 30) for _ in range(4)]
    masked = [mask_update(u, i, [0, 1, 2, 3], round_id=7, secret=b"k") for i, u in enumerate(updates)]
    assert np.allclose(aggregate(masked), np.sum(updates, axis=0))
    assert np.abs(masked[0] - updates[0]).mean() > 10      # an individual masked update reveals nothing useful


def test_non_iid_split_is_skewed(data):
    x_tr, _, y_tr, _ = data
    parts = split_non_iid(x_tr, y_tr, 5)
    assert sum(len(p[1]) for p in parts) == len(y_tr)
    assert max(len(np.unique(p[1])) for p in parts) <= 6
    assert min(len(np.unique(p[1])) for p in split_iid(x_tr, y_tr, 5)) == 10


def test_fedavg_close_to_centralized(data):
    x_tr, x_te, y_tr, y_te = data
    cen = centralized(x_tr, y_tr, 64, 10, epochs=30).model.accuracy(x_te, y_te)
    fed = fedavg(split_iid(x_tr, y_tr, 5), 64, 10, FLConfig(rounds=15), x_te, y_te).history[-1]
    assert cen > 0.95 and fed > cen - 0.05


def test_dp_fedavg_trains_and_reports_epsilon(data):
    x_tr, x_te, y_tr, y_te = data
    r = fedavg(split_iid(x_tr, y_tr, 5), 64, 10,
               FLConfig(rounds=10, local_epochs=1, batch=16, sigma=1.5, lr=1.0), x_te, y_te)
    assert 1 < r.epsilon < 20 and r.history[-1] > 0.6


def test_dp_reduces_membership_leakage(data):
    x_tr, x_te, y_tr, y_te = data
    i, o = slice(0, 200), slice(0, 200)
    leaky = centralized(x_tr[i], y_tr[i], 64, 10, epochs=300, lr=1.0).model
    priv = centralized(x_tr[i], y_tr[i], 64, 10, epochs=15, lr=1.0, batch=16, sigma=6.0).model
    a_leaky = membership_advantage(leaky, x_tr[i], y_tr[i], x_te[o], y_te[o])["advantage"]
    a_priv = membership_advantage(priv, x_tr[i], y_tr[i], x_te[o], y_te[o])["advantage"]
    assert a_leaky > 0.1 and a_priv < a_leaky


def test_cli(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "FedAvg, non-IID" in out and "Membership inference" in out
