"""
Train the two readouts the sudoku demo switches between, and export them.

  mbon    the mushroom body as it is - Kenyon cells onto output neurons
          through ONE learned layer. This is the fly.
  plus    the same Kenyon cell code with one extra learned layer added.
          No fly has this. It is here to show what the missing layer costs.

Everything upstream is identical and real: the measured PN -> Kenyon cell
connectome, feedback inhibition, graded Kenyon cell firing rates.
"""

import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sudoku_fly import (
    UNKNOWN, N_CLASS, build_encoder, encode, make_dataset, make_puzzle,
    context_bits, OUT, WEB,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(23)
SPARSITY = 0.20
HIDDEN = 64


def kc_graded(x, M, sparsity=SPARSITY):
    """Kenyon cell firing rates above the inhibition threshold."""
    y = x @ M
    k = max(1, int(round(sparsity * y.shape[1])))
    thr = np.partition(y, -k, axis=1)[:, -k][:, None]
    g = np.maximum(y - thr, 0.0)
    return g / np.maximum(g.max(axis=1, keepdims=True), 1e-6)


def fit(model, K, y, epochs=90, lr=3e-3, bs=1024):
    model = model.to(DEV)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEV)
    yt = torch.as_tensor(y, dtype=torch.long, device=DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    n = len(Kt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = lossf(model(Kt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def probs(model, K):
    out = model(torch.as_tensor(K, dtype=torch.float32, device=DEV))
    return torch.softmax(out, dim=1).cpu().numpy()


def pick_threshold(P, yt, target=0.999):
    conf, pred = P.max(axis=1), P.argmax(axis=1)
    for tau in np.arange(0.50, 0.9995, 0.002):
        commit = (pred != UNKNOWN) & (conf >= tau)
        if commit.sum() < 40:
            break
        if (pred[commit] == yt[commit]).mean() >= target:
            return float(tau), float((pred[commit] == yt[commit]).mean()), \
                   float(commit.sum() / max(1, (yt != UNKNOWN).sum()))
    return 0.999, 0.0, 0.0


def solve(board, model, Mn, pf, n_pn, tau, max_steps=90):
    g = board.copy()
    moves = 0
    for _ in range(max_steps):
        blanks = [(r, c) for r in range(9) for c in range(9) if not g[r, c]]
        if not blanks:
            break
        bits = np.array([context_bits(g, r, c) for r, c in blanks], np.float32)
        P = probs(model, kc_graded(encode(bits, pf, n_pn), Mn))
        conf, pred = P.max(axis=1), P.argmax(axis=1)
        ok = (pred != UNKNOWN) & (conf >= tau)
        if not ok.any():
            break
        i = int(np.argmax(np.where(ok, conf, -1)))
        r, c = blanks[i]
        g[r, c] = int(pred[i]) + 1
        moves += 1
    return g, moves


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def main():
    mats = np.load(OUT / "matrices.npz", allow_pickle=True)
    real = mats["real"]
    gloms = [str(g) for g in mats["pn_glom"]]
    uniq = sorted(set(gloms))
    n_pn, n_kc = real.shape
    Mn = real / real.max()
    pf = build_encoder(gloms, uniq, RNG, M=real)

    Xb, y = make_dataset(3000, RNG, balance=True)
    Xtb, yt = make_dataset(400, RNG)
    sing = yt != UNKNOWN
    K = kc_graded(encode(Xb, pf, n_pn), Mn)
    Kt = kc_graded(encode(Xtb, pf, n_pn), Mn)
    print(f"train {len(K):,} cells | test {len(Kt):,} | naked singles {sing.mean():.1%}")
    print(f"Kenyon cells active: top {SPARSITY:.0%} of {n_kc}\n")

    models = {
        "mbon": nn.Linear(n_kc, N_CLASS),
        "plus": nn.Sequential(nn.Linear(n_kc, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, N_CLASS)),
    }
    label = {"mbon": "mushroom body output neurons (the fly)",
             "plus": "one extra learned layer (no fly has this)"}

    results, export = {}, {}
    prng = np.random.default_rng(5)
    puzzles = [make_puzzle(prng) for _ in range(40)]

    for name, model in models.items():
        model = fit(model, K, y)
        model.eval()
        P = probs(model, Kt)
        a = (P.argmax(1)[sing] == yt[sing]).mean()
        tau, prec, cover = pick_threshold(P, yt)

        solved = corrupt = 0
        filled = []
        for puz, sol in puzzles:
            g, mv = solve(puz, model, Mn, pf, n_pn, tau)
            blanks0 = int((puz == 0).sum())
            filled.append(mv / max(1, blanks0))
            if not (g == 0).any() and (g == sol).all():
                solved += 1
            elif ((g != 0) & (g != sol)).any():
                corrupt += 1

        print(f"  {label[name]}")
        print(f"    per-cell on naked singles   {a:.4f}")
        print(f"    confidence gate tau={tau:.3f}  precision {prec:.4f}  covers {cover:.1%}")
        print(f"    puzzles solved {solved}/{len(puzzles)} | corrupted {corrupt} | "
              f"blanks filled {np.mean(filled):.1%}\n")

        results[name] = {
            "accuracy": float(a), "tau": tau, "precision": prec, "coverage": cover,
            "solved": solved, "of": len(puzzles), "corrupted": corrupt,
            "mean_filled": float(np.mean(filled)),
        }

        sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        if name == "mbon":
            export[name] = {"kind": "linear", "tau": tau,
                            "W": b64(sd["weight"].T.astype(np.float16)),
                            "b": b64(sd["bias"].astype(np.float32))}
        else:
            export[name] = {"kind": "mlp", "tau": tau, "hidden": HIDDEN,
                            "W1": b64(sd["0.weight"].T.astype(np.float16)),
                            "b1": b64(sd["0.bias"].astype(np.float32)),
                            "W2": b64(sd["2.weight"].T.astype(np.float16)),
                            "b2": b64(sd["2.bias"].astype(np.float32))}

    demo = [{"puzzle": p.astype(np.uint8).flatten().tolist(),
             "solution": s.astype(np.uint8).flatten().tolist()}
            for p, s in puzzles[:24]]

    payload = {
        "pn_feature": b64(pf.astype(np.int16)),
        "sparsity": SPARSITY, "n_class": N_CLASS, "hidden": HIDDEN,
        "readout": export, "results": results, "puzzles": demo,
    }
    (WEB / "sudoku_model.json").write_text(json.dumps(payload, separators=(",", ":")))
    (OUT / "sudoku_results.json").write_text(json.dumps(results, indent=2))
    sz = (WEB / "sudoku_model.json").stat().st_size / 1024
    print(f"wrote {WEB / 'sudoku_model.json'} ({sz:.0f} KB)")


if __name__ == "__main__":
    main()
