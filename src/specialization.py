"""
Is the fly's wiring a WORSE hash, or a SPECIALISED one?

Experiment 1 fed the circuit inputs whose feature-to-glomerulus assignment was
arbitrary, and the real matrix lost to a uniform random one. But a circuit
tuned to the statistics of its own sensory world would be expected to lose on
arbitrary input and win on input drawn from that world.

So: generate input distributions matched to each matrix's own co-occurrence
structure, then test every matrix on every distribution. If this is
specialisation there will be an interaction - each matrix should do relatively
best on the world built from its own wiring. If the real matrix simply hashes
badly, it will lose everywhere.

Input covariance is taken as the PN x PN co-occurrence Gram matrix A @ A.T,
so projection neurons that share Kenyon cells are correlated in the input -
which is exactly what it means for a circuit to be matched to its world.
"""

import json
from pathlib import Path

import numpy as np
import torch

from benchmark import KS, N_POINTS, SPARSITY, DEV, fly_hash, precision_at_k, true_neighbours
import extract_connectome as ec

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
SEEDS = [0, 1, 2]


def world_from(M, n_points, rng):
    """Sample inputs whose covariance matches this matrix's co-occurrence."""
    A = (M > 0).astype(np.float64)
    C = A @ A.T                                  # PN x PN, Gram => PSD
    C = C / np.abs(C).max()
    C += 1e-3 * np.eye(len(C))                   # numerical floor
    L = np.linalg.cholesky(C)
    x = rng.normal(size=(n_points, len(C))) @ L.T
    x = x - x.mean(axis=1, keepdims=True)
    return x.astype(np.float32)


def main():
    mats = np.load(OUT / "matrices.npz", allow_pickle=True)
    W = mats["real"]

    acc = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        ec.RNG = np.random.default_rng(2000 + seed)

        matrices = {
            "real": (W > 0).astype(np.float32),   # binary, to isolate wiring from weights
            "degree_preserving": ec.null_degree_preserving(W),
            "random_flyhash": ec.null_random_flyhash(W),
        }
        # the worlds each wiring implies
        worlds = {name: world_from(M, N_POINTS, rng) for name, M in matrices.items()}

        for wname, x in worlds.items():
            truth = true_neighbours(x, max(KS))
            for mname, M in matrices.items():
                p = precision_at_k(fly_hash(x, M), truth, KS, 4242 + seed)
                acc.setdefault(wname, {}).setdefault(mname, []).append(p[16])
            torch.cuda.empty_cache()
        print(f"  seed {seed} done")

    summary = {
        w: {m: {"mean": float(np.mean(v)), "sem": float(np.std(v) / np.sqrt(len(v)))}
            for m, v in ms.items()}
        for w, ms in acc.items()
    }
    (OUT / "specialization.json").write_text(json.dumps(summary, indent=2))

    names = ["real", "degree_preserving", "random_flyhash"]
    print("\n" + "=" * 74)
    print("precision@16 - rows = input world, cols = matrix doing the hashing")
    print("=" * 74)
    print(f"{'world':<22}" + "".join(f"{m:>17}" for m in names))
    for w in names:
        row = f"{'from ' + w:<22}"
        for m in names:
            s = summary[w][m]
            star = " *" if max(summary[w], key=lambda z: summary[w][z]["mean"]) == m else "  "
            row += f"{s['mean']:>13.4f}{star}"
        print(row)
    print("=" * 74)
    print("* = best matrix for that world")
    for w in names:
        r = summary[w]["real"]["mean"]
        b = summary[w]["random_flyhash"]["mean"]
        print(f"  in the {w:<18} world: real vs random = {100 * (r - b) / b:+.1f}%")


if __name__ == "__main__":
    main()
