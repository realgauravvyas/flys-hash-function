"""
Extract the real PN -> Kenyon cell connectivity matrix from the Drosophila
hemibrain connectome, and build matched null models.

Source: Janelia hemibrain v1.2 traced adjacencies (public, CC-BY).
        https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz

The mushroom body is the fly's associative-memory centre. Uniglomerular
olfactory projection neurons (PNs) carry odour channels from the antennal
lobe; each Kenyon cell (KC) samples a handful of them through dendritic
"claws". Dasgupta, Stevens & Navlakha (Science 2017) showed this layer acts
as a locality-sensitive hash. They assumed the sampling was random, which is
what the biology looked like in 2017. Zheng et al. (Curr Biol 2022) later
showed it is not.

This script produces the real matrix plus four nulls so the two can be
compared head to head.
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

RNG = np.random.default_rng(0xF1E)

# Uniglomerular olfactory PN types look like  <GLOMERULUS>_<tract>PN
# e.g. DA1_lPN, VM5d_adPN, DL2d_adPN. The tract is one of ad/l/il/iv/lv/v/vl.
# Excluded: M_* (multiglomerular), WEDPN* (wedge, not olfactory), LPN, and
# the VP* thermo/hygrosensory channels which are not odour channels.
UNI_PN = re.compile(r"^([A-Z]{1,2}[0-9]?[a-z]?)_(ad|l|il|iv|lv|v|vl)PN[0-9]*$")


def load():
    neurons = pd.read_csv(DATA / "traced-neurons.csv")
    neurons["type"] = neurons["type"].fillna("")
    conns = pd.read_csv(DATA / "traced-total-connections.csv")
    return neurons, conns


def select_populations(neurons):
    t = neurons["type"]
    is_pn = (
        t.str.match(UNI_PN)
        & ~t.str.startswith("M_")
        & ~t.str.contains("WED")
        & ~t.str.startswith("VP")
    )
    pns = neurons[is_pn].copy()
    pns["glom"] = pns["type"].str.split("_").str[0]
    # order PNs by glomerulus so the visualisation groups odour channels
    pns = pns.sort_values(["glom", "bodyId"]).reset_index(drop=True)

    kcs = neurons[t.str.startswith("KC")].copy()
    # drop the handful of incomplete / ambiguous reconstructions
    kcs = kcs[~kcs["type"].str.contains(r"incomplete|half", case=False, regex=True)]
    kcs = kcs.sort_values(["type", "bodyId"]).reset_index(drop=True)
    return pns, kcs


def build_real_matrix(pns, kcs, conns, min_syn=3):
    """Weighted PN x KC matrix of synapse counts."""
    pn_idx = {b: i for i, b in enumerate(pns.bodyId)}
    kc_idx = {b: i for i, b in enumerate(kcs.bodyId)}

    e = conns[conns.bodyId_pre.isin(pn_idx) & conns.bodyId_post.isin(kc_idx)]
    e = e[e.weight >= min_syn]

    W = np.zeros((len(pns), len(kcs)), dtype=np.float32)
    W[e.bodyId_pre.map(pn_idx).to_numpy(), e.bodyId_post.map(kc_idx).to_numpy()] = (
        e.weight.to_numpy()
    )

    # keep only KCs that actually receive olfactory drive
    keep = W.sum(axis=0) > 0
    W = W[:, keep]
    kcs = kcs[keep].reset_index(drop=True)
    return W, kcs


# ---------------------------------------------------------------- null models

def null_random_flyhash(W):
    """Dasgupta 2017: each KC samples `claws` PNs uniformly at random, binary.

    This is the matrix the entire FlyHash / BioHash literature actually uses.
    """
    n_pn, n_kc = W.shape
    claws = (W > 0).sum(axis=0)
    M = np.zeros_like(W)
    for j, c in enumerate(claws):
        M[RNG.choice(n_pn, size=int(c), replace=False), j] = 1.0
    return M


def null_degree_preserving(W, n_swaps_per_edge=20):
    """Maslov-Sneppen double-edge swap.

    Preserves BOTH each KC's claw count and each PN's out-degree exactly,
    destroying only the higher-order correlation structure. This is the
    strict null: any advantage the real matrix shows over this one cannot be
    explained by degree sequence alone.
    """
    A = (W > 0).astype(np.int8)
    pre, post = np.nonzero(A)
    edges = list(zip(pre.tolist(), post.tolist()))
    eset = set(edges)
    n_target = n_swaps_per_edge * len(edges)
    done = 0
    for _ in range(n_target * 12):
        if done >= n_target:
            break
        i, j = RNG.integers(0, len(edges), size=2)
        if i == j:
            continue
        p1, k1 = edges[i]
        p2, k2 = edges[j]
        if p1 == p2 or k1 == k2:
            continue
        if (p1, k2) in eset or (p2, k1) in eset:
            continue
        eset.discard((p1, k1))
        eset.discard((p2, k2))
        eset.add((p1, k2))
        eset.add((p2, k1))
        edges[i] = (p1, k2)
        edges[j] = (p2, k1)
        done += 1

    M = np.zeros_like(W)
    for p, k in edges:
        M[p, k] = 1.0
    return M


def null_glomerulus_matched(W):
    """Preserves how often each PN is sampled overall (the marginal bias
    Zheng et al. measured) but samples each KC independently, destroying
    which PNs co-occur on the same cell."""
    n_pn, n_kc = W.shape
    claws = (W > 0).sum(axis=0)
    p = (W > 0).sum(axis=1).astype(np.float64)
    p = p / p.sum()
    M = np.zeros_like(W)
    for j, c in enumerate(claws):
        M[RNG.choice(n_pn, size=int(c), replace=False, p=p), j] = 1.0
    return M


def null_weight_shuffled(W):
    """Identical topology, synapse counts permuted among existing edges.
    Isolates whether the *weights* carry information beyond the wiring."""
    M = W.copy()
    nz = np.nonzero(M)
    vals = M[nz].copy()
    RNG.shuffle(vals)
    M[nz] = vals
    return M


def main():
    neurons, conns = load()
    pns, kcs = select_populations(neurons)
    print(f"uniglomerular olfactory PNs : {len(pns)} across {pns.glom.nunique()} glomeruli")
    print(f"Kenyon cells (all)          : {len(kcs)}")

    W, kcs = build_real_matrix(pns, kcs, conns)
    claws = (W > 0).sum(axis=0)
    print(f"Kenyon cells (olfactory)    : {W.shape[1]}")
    print(f"synapses                    : {int(W.sum()):,}")
    print(f"claws/KC  mean {claws.mean():.2f}  median {np.median(claws):.0f}  max {claws.max()}")
    print(f"density                     : {(W > 0).mean():.4f}")

    nulls = {
        "random_flyhash": null_random_flyhash(W),
        "degree_preserving": null_degree_preserving(W),
        "glomerulus_matched": null_glomerulus_matched(W),
        "weight_shuffled": null_weight_shuffled(W),
    }
    for k, M in nulls.items():
        assert (M > 0).sum() == (W > 0).sum(), f"{k} edge count mismatch"
        print(f"  null {k:20s} edges {(M > 0).sum():6d}  ok")

    np.savez_compressed(
        OUT / "matrices.npz",
        real=W,
        **nulls,
        pn_glom=pns.glom.to_numpy().astype(str),
        pn_type=pns["type"].to_numpy().astype(str),
        kc_type=kcs["type"].to_numpy().astype(str),
    )

    meta = {
        "source": "Janelia hemibrain v1.2 (CC-BY)",
        "n_pn": int(W.shape[0]),
        "n_kc": int(W.shape[1]),
        "n_glomeruli": int(pns.glom.nunique()),
        "n_synapses": int(W.sum()),
        "mean_claws": float(claws.mean()),
        "median_claws": float(np.median(claws)),
        "density": float((W > 0).mean()),
        "glomeruli": sorted(pns.glom.unique().tolist()),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT / 'matrices.npz'}")


if __name__ == "__main__":
    main()
