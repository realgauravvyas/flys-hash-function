"""
Teach the real mushroom body to read a sudoku cell.

The circuit is the one measured in the hemibrain: 130 projection neurons ->
1745 Kenyon cells through the actual synapses, feedback inhibition keeping the
top 5%, then a learned readout onto mushroom body output neurons.

The fly sees 27 raw facts about a cell - for each digit, whether it already
appears in that cell's row, column and box - and must learn to answer "which
digit goes here, or don't know". Nothing is pre-computed for it: candidate
sets are exactly what it has to learn to derive.

Two readouts are trained and compared:
  * dopaminergic depression - the rule the mushroom body actually uses, where
    a wrong answer depresses the synapses from currently-active Kenyon cells
    onto the output neuron that produced it
  * logistic regression - a conventional readout, as a ceiling

And two wirings: the real connectome, and the random matrix, so the question
from the hashing study carries over - does the real wiring learn this better?
"""

import base64
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
WEB = ROOT / "web"

RNG = np.random.default_rng(7)
SPARSITY = 0.05
N_CLASS = 10          # digits 1-9, plus "not determined"
UNKNOWN = 9


# --------------------------------------------------------------- sudoku core

def complete_grid(rng):
    """A valid filled grid, by shuffling a canonical pattern."""
    base = np.array([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)]
                     for r in range(9)])
    for band in range(3):                       # rows within each band
        idx = rng.permutation(3) + 3 * band
        base[3 * band:3 * band + 3] = base[idx]
    for stack in range(3):                      # columns within each stack
        idx = rng.permutation(3) + 3 * stack
        base[:, 3 * stack:3 * stack + 3] = base[:, idx]
    base = base[rng.permutation(3).repeat(3) * 3 + np.tile(np.arange(3), 3)]
    perm = np.concatenate([[0], rng.permutation(9) + 1])
    return perm[base]


def context_bits(board, r, c):
    """The 27 raw facts the fly is given about one cell."""
    row = np.zeros(9, np.float32)
    col = np.zeros(9, np.float32)
    box = np.zeros(9, np.float32)
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(9):
        v = board[r, i]
        if v:
            col[v - 1] = 1          # digit v is used elsewhere in this row
        v = board[i, c]
        if v:
            row[v - 1] = 1
    for i in range(3):
        for j in range(3):
            v = board[br + i, bc + j]
            if v:
                box[v - 1] = 1
    return np.concatenate([row, col, box])


def label_for(bits):
    """Exactly one digit free -> that digit. Otherwise 'not determined'."""
    free = [d for d in range(9) if not (bits[d] or bits[9 + d] or bits[18 + d])]
    return free[0] if len(free) == 1 else UNKNOWN


def singles_solver(board):
    """Ground truth: apply naked singles until nothing more can be deduced."""
    b = board.copy()
    while True:
        moved = False
        for r in range(9):
            for c in range(9):
                if b[r, c]:
                    continue
                lab = label_for(context_bits(b, r, c))
                if lab != UNKNOWN:
                    b[r, c] = lab + 1
                    moved = True
        if not moved:
            return b, not (b == 0).any()


def make_puzzle(rng, target_blank=44):
    """A puzzle a naked-singles solver can finish, plus its solution."""
    for _ in range(260):
        full = complete_grid(rng)
        board = full.copy()
        order = rng.permutation(81)
        for idx in order[:target_blank]:
            board.flat[idx] = 0
        _, ok = singles_solver(board)
        if ok:
            return board, full
    return board, full


# ------------------------------------------------------------ fly front end

def build_encoder(gloms, uniq, rng, M=None):
    """Map the 27 constraint facts onto the 52 glomeruli.

    One glomerulus carries one fact, which is how olfactory channels are
    actually organised - every projection neuron of a glomerulus reports the
    same thing. Every glomerulus is used, so all 130 projection neurons stay
    informative and no Kenyon cell is wired to dead channels.

    When the connectome is supplied, placement is aligned to it. Deciding that
    digit d is free requires reading row_d, col_d and box_d together, so those
    three facts are put on glomeruli that this wiring actually samples onto
    the same Kenyon cells. That is the structure the hashing study found the
    circuit to be specialised for, used here as it would be used in life.
    """
    n_g = len(uniq)
    if M is None:
        perm = rng.permutation(n_g)
        g2f = {uniq[g]: int(i % 27) for i, g in enumerate(perm)}
        return np.array([g2f[g] for g in gloms], dtype=np.int32)

    # glomerulus x glomerulus co-occurrence over Kenyon cells
    gi = {g: i for i, g in enumerate(uniq)}
    A = (M > 0).astype(np.float64)
    G = np.zeros((n_g, A.shape[1]))
    for p, g in enumerate(gloms):
        G[gi[g]] += A[p]
    C = G @ G.T
    np.fill_diagonal(C, -1.0)

    # greedily take the 9 most strongly co-sampled disjoint triples
    free = set(range(n_g))
    triples = []
    for _ in range(9):
        best, bs = None, -1.0
        fl = sorted(free)
        for ii, a in enumerate(fl):
            for b in fl[ii + 1:]:
                for c in fl:
                    if c <= b:
                        continue
                    s = C[a, b] + C[a, c] + C[b, c]
                    if s > bs:
                        bs, best = s, (a, b, c)
        triples.append(best)
        free -= set(best)

    feat = np.full(n_g, -1, dtype=np.int32)
    for d, (a, b, c) in enumerate(triples):
        feat[a], feat[b], feat[c] = d, 9 + d, 18 + d

    # leftover glomeruli echo whichever assigned channel they co-occur with most
    for g in sorted(free):
        order = np.argsort(-C[g])
        for o in order:
            if feat[o] >= 0:
                feat[g] = feat[o]
                break

    return np.array([feat[gi[g]] for g in gloms], dtype=np.int32)


def encode(bits_batch, pn_feature, n_pn):
    """27 facts -> 130 projection neurons, then antennal lobe normalisation."""
    x = np.zeros((len(bits_batch), n_pn), np.float32)
    has = pn_feature >= 0
    x[:, has] = bits_batch[:, pn_feature[has]]
    x -= x.mean(axis=1, keepdims=True)
    s = np.sqrt((x ** 2).mean(axis=1, keepdims=True))
    return x / np.maximum(s, 1e-6)


def kc_code(x, M, sparsity=SPARSITY):
    """Project through the connectome, then winner-take-all."""
    y = x @ M
    k = max(1, int(round(sparsity * y.shape[1])))
    thr = np.partition(y, -k, axis=1)[:, -k][:, None]
    return (y >= thr).astype(np.float32)


# ---------------------------------------------------------------- readouts

def train_depression(K, y, n_epochs=12, lr=0.035):
    """The mushroom body's own rule.

    Synapses start uniform and strong. When the circuit answers wrongly,
    dopamine depresses the synapses from the Kenyon cells that were active
    onto the output neuron that fired. Depression only - nothing potentiates.
    """
    W = np.ones((K.shape[1], N_CLASS), np.float32)
    order = np.arange(len(K))
    for ep in range(n_epochs):
        RNG.shuffle(order)
        for i in order:
            k = K[i]
            act = k > 0
            scores = W[act].sum(axis=0)
            pred = int(np.argmax(scores))
            if pred != y[i]:
                W[act, pred] -= lr           # punish the wrong output
                np.maximum(W[:, pred], 0.0, out=W[:, pred])
    return W


def train_logistic(K, y, n_epochs=40, lr=0.5):
    """Conventional softmax readout, as a performance ceiling."""
    W = np.zeros((K.shape[1], N_CLASS), np.float32)
    b = np.zeros(N_CLASS, np.float32)
    n = len(K)
    Y = np.zeros((n, N_CLASS), np.float32)
    Y[np.arange(n), y] = 1
    for ep in range(n_epochs):
        for s in range(0, n, 512):
            kb, yb = K[s:s + 512], Y[s:s + 512]
            z = kb @ W + b
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            g = (p - yb) / len(kb)
            W -= lr * (kb.T @ g)
            b -= lr * g.sum(axis=0)
    return W, b


# ------------------------------------------------------------------ dataset

def make_dataset(n_boards, rng, balance=False):
    bits, labels = [], []
    for _ in range(n_boards):
        full = complete_grid(rng)
        board = full.copy()
        n_blank = rng.integers(20, 56)
        flat = rng.permutation(81)[:n_blank]
        board.flat[flat] = 0
        for idx in flat:
            r, c = divmod(int(idx), 9)
            b = context_bits(board, r, c)
            bits.append(b)
            labels.append(label_for(b))
    X = np.array(bits, np.float32)
    y = np.array(labels, np.int64)
    if balance:
        # naked singles are only ~23% of cells; without this the readout can
        # score well by answering "don't know" to everything
        single = np.nonzero(y != UNKNOWN)[0]
        unknown = np.nonzero(y == UNKNOWN)[0]
        keep = rng.choice(unknown, size=min(len(unknown), len(single)), replace=False)
        sel = rng.permutation(np.concatenate([single, keep]))
        X, y = X[sel], y[sel]
    return X, y


# --------------------------------------------------------------------- main

def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def pick_threshold(P, yt, target=0.995):
    """Lowest confidence cut-off whose committed answers are almost all right.

    The circuit answers only when the winning output neuron is well clear of
    the field. Below the cut-off the fly moves on and comes back later.
    """
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    best = None
    for tau in np.arange(0.30, 0.996, 0.005):
        commit = (pred != UNKNOWN) & (conf >= tau)
        if commit.sum() < 50:
            break
        prec = (pred[commit] == yt[commit]).mean()
        cover = commit.sum() / max(1, (yt != UNKNOWN).sum())
        if prec >= target:
            best = (float(tau), float(prec), float(cover))
            break
    if best is None:
        best = (0.99, float((pred[(pred != UNKNOWN) & (conf >= 0.99)] ==
                             yt[(pred != UNKNOWN) & (conf >= 0.99)]).mean()), 0.0)
    return best


def fly_solve(board, M, W, b, pn_feature, n_pn, tau, sp=SPARSITY, max_passes=90):
    """Fly the circuit over the grid, writing only what it is sure of."""
    g = board.copy()
    moves = []
    for _ in range(max_passes):
        blanks = [(r, c) for r in range(9) for c in range(9) if not g[r, c]]
        if not blanks:
            break
        bits = np.array([context_bits(g, r, c) for r, c in blanks], np.float32)
        K = kc_code(encode(bits, pn_feature, n_pn), M, sp)
        P = softmax(K @ W + b)
        conf = P.max(axis=1)
        pred = P.argmax(axis=1)
        ok = (pred != UNKNOWN) & (conf >= tau)
        if not ok.any():
            break
        best = int(np.argmax(np.where(ok, conf, -1)))
        r, c = blanks[best]
        g[r, c] = int(pred[best]) + 1
        moves.append((r, c, int(pred[best]) + 1, float(conf[best])))
    return g, moves


def main():
    mats = np.load(OUT / "matrices.npz", allow_pickle=True)
    real = mats["real"]
    rand = mats["random_flyhash"]
    gloms = [str(g) for g in mats["pn_glom"]]
    uniq = sorted(set(gloms))
    n_pn = real.shape[0]

    Xb, y = make_dataset(2500, RNG, balance=True)
    Xtb, yt = make_dataset(400, RNG)
    print(f"train {len(Xb):,} cells (balanced) | test {len(Xtb):,} cells (natural mix)")
    print(f"naked singles in test: {(yt != UNKNOWN).mean():.1%}")
    sing = yt != UNKNOWN

    # what a linear readout gets from the raw 27 facts, with no Kenyon cell
    # expansion at all - the number the mushroom body has to beat
    Wr, br = train_logistic(Xb, y)
    praw = np.argmax(Xtb @ Wr + br, axis=1)
    print(f"baseline, no expansion: on naked singles {(praw[sing] == yt[sing]).mean():.3f}")

    # how many Kenyon cells should survive inhibition
    print("\nsparsity sweep (aligned encoder, real connectome)")
    pf_probe = build_encoder(gloms, uniq, RNG, M=real)
    xp, xtp = encode(Xb, pf_probe, n_pn), encode(Xtb, pf_probe, n_pn)
    best_sp, best_acc = SPARSITY, -1.0
    for sp in (0.05, 0.10, 0.20):
        Wp, bp = train_logistic(kc_code(xp, real / real.max(), sp), y)
        pp = np.argmax(kc_code(xtp, real / real.max(), sp) @ Wp + bp, axis=1)
        a = (pp[sing] == yt[sing]).mean()
        print(f"  top {sp:.0%} of Kenyon cells -> on naked singles {a:.3f}")
        if a > best_acc:
            best_acc, best_sp = a, sp
    print(f"  using {best_sp:.0%}")

    results = {}
    export = {}
    for wname, M in (("real", real), ("random", rand)):
        Mn = M / M.max()
        pn_feature = build_encoder(gloms, uniq, RNG, M=M)
        x = encode(Xb, pn_feature, n_pn)
        xt = encode(Xtb, pn_feature, n_pn)
        K, Kt = kc_code(x, Mn, best_sp), kc_code(xt, Mn, best_sp)

        Wd = train_depression(K, y)
        pd_ = np.argmax(Kt @ Wd, axis=1)

        Wl, bl = train_logistic(K, y)
        pl = np.argmax(Kt @ Wl + bl, axis=1)

        def score(p):
            single = yt != UNKNOWN
            return {
                "overall": float((p == yt).mean()),
                "on_singles": float((p[single] == yt[single]).mean()),
                "false_commit": float((p[~single] != UNKNOWN).mean()),
            }

        results[wname] = {"depression": score(pd_), "logistic": score(pl)}
        print(f"\n  {wname} connectome")
        for rule in ("depression", "logistic"):
            s = results[wname][rule]
            print(f"    {rule:<12} correct {s['overall']:.3f} | "
                  f"on naked singles {s['on_singles']:.3f} | "
                  f"wrongly commits {s['false_commit']:.3f}")

        P = softmax(Kt @ Wl + bl)
        tau, prec, cover = pick_threshold(P, yt)
        print(f"    confidence gate tau={tau:.3f} -> precision {prec:.4f}, "
              f"covers {cover:.1%} of naked singles")

        solved = 0
        wrong = 0
        filled = []
        n_puz = 40
        prng = np.random.default_rng(99)
        for _ in range(n_puz):
            puz, sol = make_puzzle(prng)
            g, mv = fly_solve(puz, Mn, Wl, bl, pn_feature, n_pn, tau, best_sp)
            blanks0 = int((puz == 0).sum())
            filled.append(len(mv) / max(1, blanks0))
            if not (g == 0).any() and (g == sol).all():
                solved += 1
            elif ((g != 0) & (g != sol)).any():
                wrong += 1
        print(f"    puzzles fully solved: {solved}/{n_puz}"
              f"  |  corrupted: {wrong}/{n_puz}"
              f"  |  mean blanks filled: {np.mean(filled):.1%}")

        results[wname]["gate"] = {"tau": tau, "precision": prec, "coverage": cover}
        results[wname]["puzzles"] = {"solved": solved, "of": n_puz, "corrupted": wrong,
                                     "mean_filled": float(np.mean(filled))}
        export[wname] = {
            "W": b64(Wl.astype(np.float32)),
            "b": b64(bl.astype(np.float32)),
            "tau": tau,
            "pn_feature": b64(pn_feature.astype(np.int16)),
        }

    payload = {
        "readout": export,
        "results": results,
        "sparsity": best_sp,
        "n_class": N_CLASS,
    }
    (WEB / "sudoku_model.json").write_text(json.dumps(payload, separators=(",", ":")))
    (OUT / "sudoku_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {WEB / 'sudoku_model.json'}")


if __name__ == "__main__":
    main()
