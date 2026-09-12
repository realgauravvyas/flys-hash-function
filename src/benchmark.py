"""
Does the fly's REAL wiring hash better than the random wiring everyone assumes?

Protocol follows Dasgupta, Stevens & Navlakha (Science 2017):
  1. centre each input vector        (antennal lobe divisive normalisation)
  2. project up through a sparse matrix   PN (130) -> KC (1745)
  3. winner-take-all, keep the top 5%     (APL feedback inhibition)
  4. the surviving KC identities ARE the hash code

The only thing that differs between conditions is the 130 x 1745 matrix.
Everything upstream and downstream is held identical, so any difference in
retrieval quality is attributable to the wiring alone.

Conditions
  real                 measured hemibrain synapse counts
  real_binary          measured wiring, weights flattened to 1
  random_flyhash       Dasgupta 2017's assumption: uniform random sampling
  degree_preserving    same claw counts AND same PN out-degrees, rewired
  glomerulus_matched   same PN sampling bias, independent per KC
  weight_shuffled      same wiring, synapse counts permuted
  simhash              classical dense LSH, matched bit budget
"""

import gzip
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

from extract_connectome import (
    null_degree_preserving,
    null_glomerulus_matched,
    null_random_flyhash,
    null_weight_shuffled,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N_POINTS = 10_000
SPARSITY = 0.05          # fraction of Kenyon cells that survive inhibition
KS = [2, 4, 8, 16, 32, 64, 100]
SEEDS = [0, 1, 2, 3, 4]

URLS = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "fashion_mnist": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",
}


def load_idx_images(name):
    path = DATA / f"{name}.idx.gz"
    if not path.exists():
        print(f"  downloading {name} ...")
        urllib.request.urlretrieve(URLS[name], path)
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        buf = f.read(n * rows * cols)
    x = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
    return x.astype(np.float32) / 255.0


def make_odour_like(n_points, n_dim, rank, rng):
    """Synthetic naturalistic sensory data: low-rank correlated structure,
    the statistical regime real odour environments live in."""
    loadings = rng.normal(size=(rank, n_dim))
    scores = rng.normal(size=(n_points, rank))
    x = scores @ loadings + 0.35 * rng.normal(size=(n_points, n_dim))
    return np.maximum(x, 0).astype(np.float32)


def datasets(rng):
    out = {}
    for name in ("mnist", "fashion_mnist"):
        x = load_idx_images(name)
        idx = rng.choice(len(x), N_POINTS, replace=False)
        out[name] = x[idx]
    out["odour_correlated"] = make_odour_like(N_POINTS, 256, rank=20, rng=rng)
    return out


def encode(x, n_pn, rng):
    """Shared, fixed random encoder into PN space. Identical for every
    condition within a seed, so it can never favour one matrix over another."""
    x = x - x.mean(axis=0, keepdims=True)
    e = rng.normal(scale=1.0 / np.sqrt(x.shape[1]), size=(x.shape[1], n_pn))
    z = x @ e
    z = z - z.mean(axis=1, keepdims=True)       # divisive normalisation
    return z.astype(np.float32)


def fly_hash(z, M, sparsity=SPARSITY):
    """Project up, then winner-take-all. Returns a binary code."""
    zt = torch.as_tensor(z, device=DEV)
    Mt = torch.as_tensor(M, device=DEV)
    y = zt @ Mt
    n_keep = max(1, int(round(sparsity * y.shape[1])))
    idx = y.topk(n_keep, dim=1).indices
    b = torch.zeros_like(y, dtype=torch.float16)
    b.scatter_(1, idx, 1.0)
    return b


def simhash(z, n_bits, rng):
    r = rng.normal(size=(z.shape[1], n_bits))
    return torch.as_tensor((z @ r > 0).astype(np.float16), device=DEV)


def true_neighbours(z, k_max):
    zt = torch.as_tensor(z, device=DEV)
    zt = zt / (zt.norm(dim=1, keepdim=True) + 1e-9)
    nn = []
    for i in range(0, len(zt), 1024):
        s = zt[i : i + 1024] @ zt.T
        s[torch.arange(s.shape[0]), torch.arange(i, min(i + 1024, len(zt)))] = -1e9
        nn.append(s.topk(k_max, dim=1).indices)
    return torch.cat(nn)


def precision_at_k(codes, truth, ks, tie_seed):
    """Fraction of the true top-k that the hash actually retrieves.

    Hash similarities are integer overlap counts, so ties are rampant and
    would otherwise be broken by index order. Jitter is drawn from a
    generator seeded identically for every condition, so tie-breaking is
    random but fair across the comparison.
    """
    k_max = max(ks)
    res = {k: 0.0 for k in ks}
    n = codes.shape[0]
    cf = codes.float()
    g = torch.Generator(device=DEV).manual_seed(tie_seed)
    for i in range(0, n, 1024):
        blk = cf[i : i + 1024]
        s = blk @ cf.T
        s += torch.rand(s.shape, generator=g, device=DEV) * 1e-2
        s[torch.arange(blk.shape[0]), torch.arange(i, min(i + 1024, n))] = -1e9
        ret = s.topk(k_max, dim=1).indices
        for k in ks:
            hit = (ret[:, :k].unsqueeze(2) == truth[i : i + 1024, :k].unsqueeze(1)).any(2)
            res[k] += hit.float().sum(1).div(k).sum().item()
        del s
    return {k: v / n for k, v in res.items()}


def main():
    mats = np.load(OUT / "matrices.npz", allow_pickle=True)
    W = mats["real"]
    n_pn, n_kc = W.shape
    n_bits = int(round(SPARSITY * n_kc))
    print(f"matrix {n_pn} PN x {n_kc} KC | code = top {n_bits} KCs\n")

    results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        nrng = np.random.default_rng(1000 + seed)
        conditions = {
            "real": W,
            "real_binary": (W > 0).astype(np.float32),
            "random_flyhash": null_random_flyhash(W),
            "degree_preserving": null_degree_preserving(W),
            "glomerulus_matched": null_glomerulus_matched(W),
            "weight_shuffled": null_weight_shuffled(W),
        }
        # re-randomise the stochastic nulls per seed for honest error bars
        globals()["RNG"] = nrng
        import extract_connectome as ec
        ec.RNG = nrng
        conditions["random_flyhash"] = ec.null_random_flyhash(W)
        conditions["degree_preserving"] = ec.null_degree_preserving(W)
        conditions["glomerulus_matched"] = ec.null_glomerulus_matched(W)
        conditions["weight_shuffled"] = ec.null_weight_shuffled(W)

        for dname, x in datasets(rng).items():
            z = encode(x, n_pn, rng)
            truth = true_neighbours(z, max(KS))
            tie_seed = 7777 + seed
            for cname, M in conditions.items():
                p = precision_at_k(fly_hash(z, M), truth, KS, tie_seed)
                results.setdefault(dname, {}).setdefault(cname, []).append(p)
            p = precision_at_k(simhash(z, n_bits, rng), truth, KS, tie_seed)
            results.setdefault(dname, {}).setdefault("simhash", []).append(p)
            torch.cuda.empty_cache()
        print(f"  seed {seed} done")

    summary = {}
    for dname, conds in results.items():
        summary[dname] = {}
        for cname, runs in conds.items():
            summary[dname][cname] = {
                str(k): {
                    "mean": float(np.mean([r[k] for r in runs])),
                    "sem": float(np.std([r[k] for r in runs]) / np.sqrt(len(runs))),
                }
                for k in KS
            }

    (OUT / "benchmark.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78)
    print("precision@16  (fraction of true 16 nearest neighbours retrieved)")
    print("=" * 78)
    order = ["real", "real_binary", "degree_preserving", "glomerulus_matched",
             "weight_shuffled", "random_flyhash", "simhash"]
    header = f"{'condition':<22}" + "".join(f"{d:>18}" for d in summary)
    print(header)
    for cname in order:
        row = f"{cname:<22}"
        for dname in summary:
            s = summary[dname][cname]["16"]
            row += f"{s['mean']:>12.4f}±{s['sem']:.3f}"
        print(row)
    print("=" * 78)
    for dname in summary:
        r = summary[dname]["real"]["16"]["mean"]
        b = summary[dname]["random_flyhash"]["16"]["mean"]
        print(f"{dname:>18}: real vs random FlyHash = {100 * (r - b) / b:+.1f}%")


if __name__ == "__main__":
    main()
