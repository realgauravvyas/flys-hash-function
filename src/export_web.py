"""
Pack the connectome + results into a compact payload the browser can run live.

Kenyon cells are laid out by t-SNE over their PN input vectors, so two dots
that sit near each other are cells tuned to similar odour channels. That makes
the layout itself carry the finding: the real wiring clusters, the random
wiring is a featureless blob.
"""

import base64
import json
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
WEB = ROOT / "web"
WEB.mkdir(exist_ok=True)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def layout(M, seed=0):
    """2D embedding of Kenyon cells by which PNs they listen to."""
    A = (M > 0).T.astype(np.float32)                  # KC x PN
    xy = TSNE(
        n_components=2, perplexity=30, init="pca",
        random_state=seed, max_iter=750,
    ).fit_transform(A)
    xy = xy - xy.mean(0)
    xy = xy / np.abs(xy).max()
    return xy.astype(np.float32)


def pack(M, name, seed):
    pn, kc = np.nonzero(M)
    w = M[pn, kc].astype(np.uint16)      # raw synapse counts, no quantisation
    xy = layout(M, seed)
    print(f"  {name:<16} edges {len(pn):5d}  laid out {len(xy)} KCs")
    return {
        "pn": b64(pn.astype(np.uint8)),
        "kc": b64(kc.astype(np.uint16)),
        "w": b64(w),
        "xy": b64((xy * 32767).astype(np.int16)),
    }


def world_factor(M):
    """Cholesky factor of the PN co-occurrence Gram matrix.

    Shipping this lets the browser sample inputs from the world a given
    wiring implies, so the double dissociation can be flipped live: x = L @ z.
    """
    A = (M > 0).astype(np.float64)
    C = A @ A.T
    C = C / np.abs(C).max()
    C += 1e-3 * np.eye(len(C))
    return np.linalg.cholesky(C).astype(np.float32)


def main():
    mats = np.load(OUT / "matrices.npz", allow_pickle=True)
    real = mats["real"]
    rand = mats["random_flyhash"]

    gloms = [str(g) for g in mats["pn_glom"]]
    kc_types = [str(t) for t in mats["kc_type"]]

    # group KC types into the three anatomical classes for colouring
    def kc_class(t):
        if t.startswith("KCg"):
            return 0
        if t.startswith("KCa'b'"):
            return 1
        return 2

    payload = {
        "meta": json.loads((OUT / "meta.json").read_text()),
        "n_pn": int(real.shape[0]),
        "n_kc": int(real.shape[1]),
        "gloms": gloms,
        "uniq_gloms": sorted(set(gloms)),
        "kc_class": b64(np.array([kc_class(t) for t in kc_types], dtype=np.uint8)),
        "real": pack(real, "real", 0),
        "random": pack(rand, "random", 1),
        "world_real": b64(world_factor(real)),
        "world_random": b64(world_factor(rand)),
        "benchmark": json.loads((OUT / "benchmark.json").read_text()),
        "specialization": json.loads((OUT / "specialization.json").read_text()),
    }

    path = WEB / "connectome.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"\nwrote {path}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
