"""The third column of Claim 1: the same axes priced by a fitted MIXTURE instead of one weight.

`claim1` showed that choosing ONE scalar weight on the scored rows costs almost nothing. The 3-9x
overstatement in the project's records came from `groupprice`, which is a different estimator: a
non-negative least-squares blend over the whole candidate set, in delta space, cross-fitted by well.
Its degrees of freedom equal the number of candidates.

This runs that estimator on the same four axes and reads it on the same eight regions, so the three
numbers finally sit in one table:

    in-sample   one weight, chosen on the scored rows          (claim1)
    price       a fitted mixture over k candidates, xfit well  (here)
    honest      one weight, chosen outside the region          (claim1)

If the optimism tracks k rather than "in-sample", price is far more optimistic than in-sample even
though it is cross-fitted -- which is the paper's point.
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd, torch
from scipy.optimize import nnls

from stride.gsel import Harness, load, DEV
from stride.score465 import families, STRONG_EXTRA, LATERAL
from stride.rowmod import per_well_z, zscore
from stride.regionout import regions

LAT = (0.10, 0.15, 0.20, 0.25)
W0 = (0.75, 0.85, 0.95)
K = (0.0, -0.05, -0.10)
NFOLD = 5
MVP = ("MV", "MW", "MX", "MY", "UW", "UV", "SO", "SN")
SHAPE = ("CH_n", "CN0", "CN1", "CN2", "CW0", "CW1", "CW4", "CW8",
         "SF25", "SFK", "SFsC", "S2a", "SK", "SL")


def main():
    ev, wid, names, A = load("465")
    kill, lat = families(names)
    idx = {n: i for i, n in enumerate(names)}
    known = set(STRONG_EXTRA) | set(LATERAL)
    sh = [n for n in names if "__" not in n and n not in known and n.startswith(SHAPE)]
    mv = [n for n in names if "__" not in n and n[:2] in MVP and n[2:].strip("0123456789") == ""]
    mv4 = [n for n in ("MV1", "MV2", "MV3", "MV4") if n in idx]
    gg = lambda L: A[[idx[n] for n in L]].mean(0, keepdims=True)
    us = [f"{n}__usd" for n in kill if f"{n}__usd" in idx]
    H = Harness(ev, wid, np.concatenate([gg(kill), gg(lat), gg(us)]))
    gate = 0.5 * (per_well_z(H, torch.log(H.A[2].clamp_min(1e-3)))
                  + zscore(torch.log((H.A[0] - H.A[1]).abs().clamp_min(1e-3))))
    wells = pd.factorize(ev.well)[1]
    nW = H.nW

    # the baseline A, tuned once on everything -- the same object claim1 prices against
    def readA(m, w, k, mask=None):
        T = (1 - m) * H.A[0] + m * H.A[1]
        return float(H.rms(H.B + w * gate.mul(k).exp() * (T - H.B), mask))
    best = None
    for m in LAT:
        for w in W0:
            for k in K:
                v = readA(m, w, k)
                if best is None or v < best[0]:
                    best = (v, m, w, k)
    _, m0, w0, k0 = best
    T0 = (1 - m0) * H.A[0] + m0 * H.A[1]
    PA = (H.B + w0 * gate.mul(k0).exp() * (T0 - H.B))
    y = H.y
    print(f"{nW} wells | baseline A: lat {m0:.2f} w0 {w0:.2f} k {k0:+.2f} -> {float(H.rms(PA)):.4f}")

    HOLD = []
    for kindr in ("isopach", "xy"):
        wl, lab = regions(wells, kindr)
        pos = {w: i for i, w in enumerate(wl)}
        wlab = np.array([lab[pos[w]] for w in wells])
        for r in sorted(set(lab[~pd.isna(lab)])):
            h = np.isin(wid, np.flatnonzero(wlab == r))
            HOLD.append((f"{kindr}{int(r)}", torch.as_tensor(h, device=DEV)))

    fold = np.random.default_rng(0).integers(0, NFOLD, nW)[wid]      # cross-fit BY WELL
    fold_t = torch.as_tensor(fold, device=DEV)
    r = (y - PA).double()

    print(f"\n{'axis':<20} {'k':>4} {'price':>9} {'wins':>6}")
    out = []
    for axis, mem in (("shape channel", sh), ("MV family (4)", mv4), ("MV family (all)", mv)):
        D = torch.stack([torch.as_tensor(np.array(A[idx[n]]), device=DEV).double() - PA
                         for n in mem], 1)
        P = PA.clone().double()
        for f in range(NFOLD):
            tr = fold_t != f
            G = (D[tr].T @ D[tr]).cpu().numpy()
            b = (D[tr].T @ r[tr]).cpu().numpy()
            wts, _ = nnls(G, b)
            P[~tr] = PA[~tr].double() + D[~tr] @ torch.as_tensor(wts, device=DEV).double()
        rms = lambda v, mk: float(((v - y.double())[mk] ** 2).mean().sqrt())
        d = [rms(P, h) - rms(PA.double(), h) for _, h in HOLD]
        price = float(np.mean(d))
        win = sum(1 for x in d if x < 0)
        print(f"{axis:<20} {len(mem):>4} {price:>9.4f} {win:>4}/{len(HOLD)}")
        out.append((axis, len(mem), price, win, len(HOLD)))
        del D

    pd.DataFrame(out, columns=["axis", "k", "price", "wins", "regions"]).to_csv(
        "stride/figs/claim1_price.csv", index=False)
    print("\nwrote stride/figs/claim1_price.csv")


if __name__ == "__main__":
    main()
