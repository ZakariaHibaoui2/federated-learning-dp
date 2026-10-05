# FedPrivacy — Federated Learning with Differential Privacy & Secure Aggregation

> Several "hospitals" train one model together without sharing their records. This lab implements the whole privacy stack in NumPy and **measures** what each layer buys:
> - **FedAvg**, so raw data stays local;
> - **secure aggregation**, so the server only sees the sum of updates;
> - **DP-SGD** with a proper **Rényi-DP accountant**, so updates provably leak little about any single record;
> - a **membership-inference attack**, to check that claim empirically.

[![CI](https://github.com/ZakariaHibaoui2/federated-learning-dp/actions/workflows/ci.yml/badge.svg)](https://github.com/ZakariaHibaoui2/federated-learning-dp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-013243?logo=numpy&logoColor=white)
![Privacy](https://img.shields.io/badge/privacy-DP--SGD%20%7C%20RDP-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Results (`fedprivacy`, seed 0)

Task: 10-class digit recognition (scikit-learn digits, 1,078 training records split across 5 clients), multinomial logistic regression.

**Utility**

| Setting | Test accuracy |
|---|---|
| Centralized (all data pooled, the privacy-free upper bound) | 97.4% |
| FedAvg, IID clients | 95.4% |
| FedAvg, non-IID clients (each holds ~2 classes) | 92.9% |

**Privacy / utility trade-off**: record-level DP at every client, δ = 1e-5

| Noise σ | Rounds | ε | Accuracy |
|---|---|---|---|
| 1.0 | 30 | 12.3 | 91.1% |
| 1.5 | 20 | 5.2 | 87.6% |
| 2.0 | 20 | 3.5 | 84.8% |
| 3.0 | 15 | 1.9 | 74.4% |

**Does DP actually stop leakage?** Loss-threshold membership inference on 200 training records, averaged over 3 seeds:

| ε | Test accuracy | Attack advantage (TPR−FPR) | Theoretical DP bound |
|---|---|---|---|
| ∞ (no DP) | 93.6% | **0.203** | 1.000 |
| 12.8 | 81.8% | 0.153 | 1.000 |
| 6.65 | 71.1% | 0.148 | 0.997 |
| 2.23 | 39.9% | 0.130 | 0.806 |
| 0.90 | 19.2% | **0.050** | 0.424 |

**Reading the numbers honestly:**

- Leakage falls monotonically as ε shrinks and always stays under the DP bound (e^ε−1)/(e^ε+1).
- At the moderate budgets people often deploy (ε ≈ 5–13), the *formal* guarantee against membership inference is nearly vacuous, though empirical leakage still drops by about 25%.
- With about 200 records per party, strong privacy (ε < 2) costs a great deal of accuracy. DP needs data volume: the noise is fixed per step while the signal grows with the number of records.

## How it works

| Layer | Implementation |
|---|---|
| Model | Multinomial logistic regression with **per-example gradients** (checked against finite differences) |
| FedAvg | Clients run local epochs; the server averages updates weighted by client size; IID and label-skewed non-IID partitions |
| DP-SGD | **Poisson sampling** (each record joins a batch with probability q), per-example clipping to norm C, Gaussian noise N(0, σ²C²) on the sum |
| Accountant | Subsampled-Gaussian RDP for integer orders (Mironov, Talwar & Zhang 2019), composed over steps, converted to (ε, δ). Reproduces the same ballpark as Opacus / TF-Privacy (q=0.01, σ=1.1, 10k steps → ε≈6.3) |
| Secure aggregation | Pairwise additive masks from shared seeds (simplified Bonawitz et al. 2017): each masked update looks like noise, and the masks cancel exactly in the sum |
| Attack | Loss-threshold membership inference (Yeom et al. 2018), best threshold, advantage = TPR − FPR |

> The secure-aggregation module illustrates the masking idea. A production protocol adds authenticated key agreement, dropout recovery and malicious-client defences.

## Usage

```bash
pip install -e ".[dev]"
fedprivacy                      # full suite (~10 s)
fedprivacy --clients 10 --json results.json
pytest -q                       # 9 tests
```

```python
from fedprivacy.experiments import load, split_non_iid
from fedprivacy.federated import FLConfig, fedavg

x_tr, x_te, y_tr, y_te = load()
result = fedavg(split_non_iid(x_tr, y_tr, 5), 64, 10,
                FLConfig(rounds=20, sigma=1.5, batch=16, lr=1.0, local_epochs=1), x_te, y_te)
print(result.history[-1], result.epsilon)
```

## Project structure

```
src/fedprivacy/
├── model.py        # logistic regression, per-example gradients
├── privacy.py      # clipping, noisy sums, subsampled-Gaussian RDP accountant
├── secagg.py       # pairwise-mask secure aggregation
├── federated.py    # FedAvg + DP-SGD (Poisson sampling) + secure aggregation
├── attacks.py      # membership inference
├── experiments.py  # utility, DP trade-off, leakage-vs-epsilon
└── cli.py
tests/              # 9 tests: gradient check, clipping, accountant properties, mask cancellation, non-IID split, FedAvg, DP, MIA
```

## License

[MIT](LICENSE)
