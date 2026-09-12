"""
Where does the mushroom body lose the sudoku deduction?

Three measurements against the same task:
  1. what the task allows at all      - an MLP on the raw 27 facts
  2. graded Kenyon cell activity      - firing rates above the inhibition
                                        threshold, which is what real Kenyon
                                        cells do
  3. binarised Kenyon cell activity   - the sparse code used for hashing

If 2 is high and 3 is low, the loss is caused by the winner-take-all step
rather than by the readout or the wiring - the same feedback inhibition that
makes the circuit a good hash would be what stops it reasoning.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

from sudoku_fly import (
    UNKNOWN, build_encoder, encode, make_dataset, train_logistic, OUT
)

RNG = np.random.default_rng(11)


def kc_graded(x, M, sparsity):
    """Keep the top fraction, but preserve their firing rates."""
    y = x @ M
    k = max(1, int(round(sparsity * y.shape[1])))
    thr = np.partition(y, -k, axis=1)[:, -k][:, None]
    g = np.maximum(y - thr, 0.0)
    s = g.max(axis=1, keepdims=True)
    return g / np.maximum(s, 1e-6)


def kc_binary(x, M, sparsity):
    y = x @ M
    k = max(1, int(round(sparsity * y.shape[1])))
    thr = np.partition(y, -k, axis=1)[:, -k][:, None]
    return (y >= thr).astype(np.float32)


def main():
    mats = np.load(OUT / "matrices.npz", allow_pickle=True)
    real = mats["real"]
    gloms = [str(g) for g in mats["pn_glom"]]
    uniq = sorted(set(gloms))
    n_pn = real.shape[0]
    Mn = real / real.max()

    Xb, y = make_dataset(2500, RNG, balance=True)
    Xtb, yt = make_dataset(400, RNG)
    sing = yt != UNKNOWN
    print(f"train {len(Xb):,} | test {len(Xtb):,} | naked singles {sing.mean():.1%}\n")

    def acc(p):
        return (p[sing] == yt[sing]).mean()

    # 1. is the task learnable at all
    mlp = MLPClassifier((256, 128), max_iter=60, random_state=0)
    mlp.fit(Xb, y)
    print(f"  MLP on the raw 27 facts          {acc(mlp.predict(Xtb)):.4f}   (task ceiling)")

    pf = build_encoder(gloms, uniq, RNG, M=real)
    x, xt = encode(Xb, pf, n_pn), encode(Xtb, pf, n_pn)

    for sp in (0.05, 0.20, 0.50):
        Wg, bg = train_logistic(kc_graded(x, Mn, sp), y)
        ag = acc(np.argmax(kc_graded(xt, Mn, sp) @ Wg + bg, axis=1))
        Wb_, bb = train_logistic(kc_binary(x, Mn, sp), y)
        ab = acc(np.argmax(kc_binary(xt, Mn, sp) @ Wb_ + bb, axis=1))
        print(f"  Kenyon cells, top {sp:>4.0%}  graded {ag:.4f}   binarised {ab:.4f}")

    # graded, with a readout that can itself be nonlinear
    Kg = kc_graded(x, Mn, 0.20)
    Kgt = kc_graded(xt, Mn, 0.20)
    m2 = MLPClassifier((128,), max_iter=60, random_state=0)
    m2.fit(Kg, y)
    print(f"\n  graded Kenyon cells + nonlinear readout  {acc(m2.predict(Kgt)):.4f}")


if __name__ == "__main__":
    main()
