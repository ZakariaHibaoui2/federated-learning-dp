"""fedprivacy command line: run the experiment suite."""
from __future__ import annotations

import argparse
import json
import sys

from .experiments import run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fedprivacy", description="Federated learning + differential privacy lab")
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="save raw results")
    a = ap.parse_args(argv)
    r = run(a.seed, a.clients)

    print(f"Utility (test accuracy, {a.clients} clients, secure aggregation on)")
    print(f"  centralized (data pooled)   {r['centralized']:.1%}")
    print(f"  FedAvg, IID clients         {r['fedavg_iid']:.1%}")
    print(f"  FedAvg, non-IID clients     {r['fedavg_non_iid']:.1%}")
    print("\nDP-FedAvg (record-level DP at each client, Poisson-sampled DP-SGD, delta=1e-5)")
    print(f"  {'sigma':>6} {'rounds':>7} {'epsilon':>8} {'accuracy':>9}")
    for d in r["dp"]:
        print(f"  {d['sigma']:>6} {d['rounds']:>7} {d['epsilon']:>8.2f} {d['accuracy']:>9.1%}")
    print("\nMembership inference (loss-threshold attack, 200 training records)")
    print(f"  {'epsilon':>8} {'test acc':>9} {'advantage':>10} {'DP bound':>9}")
    for m in r["mia"]:
        eps = "inf" if m["epsilon"] == float("inf") else f"{m['epsilon']:.2f}"
        print(f"  {eps:>8} {m['test_accuracy']:>9.1%} {m['mia_advantage']:>10.3f} {m['dp_bound']:>9.3f}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, default=float)
    return 0


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
