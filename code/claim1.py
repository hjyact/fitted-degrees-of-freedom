"""Claim 1, all four axes, one driver, one log.

The table in PAPER_EVIDENCE.md was assembled from four sessions and three surviving logs. This
recomputes every cell under one protocol so the claim has a single provenance, and it computes BOTH
columns in the same run, which is the whole point: the two numbers differ only in where the weight
was chosen.

    in-sample   A and B tuned on ALL wells (the held-out region included), read on that region
    honest      A and B tuned OUTSIDE the region, read on it

Both columns are then averaged over the same 8 regions, so THE ONLY DIFFERENCE IS WHERE THE WEIGHT
WAS CHOSEN. The first version of this driver compared a pooled in-sample number against a mean of
per-region numbers, which mixes the aggregation into the contrast and made the bias vanish -- the
comparison has to hold everything else fixed to mean anything.

A is always the same baseline -- strong + lateral with a tuned mixture -- so the four axes are
comparable to each other. Block sizes are printed because cutting x and y independently once gave
195/35/35/195 and inverted a conclusion.

    python -m stride.claim1
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd, torch

from stride.gsel import Harness, load, DEV
from stride.score465 import families, STRONG_EXTRA, LATERAL, row_frac
from stride.rowmod import per_well_z, zscore
from stride.regionout import regions

LAT = (0.10, 0.15, 0.20, 0.25)
W0 = (0.75, 0.85, 0.95)
K = (0.0, -0.05, -0.10)
WT = (0.0, 0.05, 0.075, 0.10, 0.15)          # the added term's weight
ALPHA = (0.0, -0.35)                          # the tail correction is signed, not a weight
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
    H = Harness(ev, wid, np.concatenate(
        [gg(kill), gg(lat), gg(us), gg(sh), gg(mv), gg(mv4)]))
    gate = 0.5 * (per_well_z(H, torch.log(H.A[2].clamp_min(1e-3)))
                  + zscore(torch.log((H.A[0] - H.A[1]).abs().clamp_min(1e-3))))
    p = row_frac(H)
    wells = pd.factorize(ev.well)[1]
    nW = H.nW

    # the tail flag: p90 of the strong family's spread per well, standardised
    spread = np.load("stride/tail_zspread.npy") if os.path.exists("stride/tail_zspread.npy") else None
    if spread is None:
        st = np.stack([A[idx[n]] for n in kill])
        per = np.array([np.percentile(st[:, wid == i].std(0), 90) for i in range(nW)])
        lg = np.log(per + 1e-6)
        spread = (lg - lg.mean()) / lg.std()
    zs = torch.as_tensor(spread, device=DEV, dtype=torch.float32)

    print(f"{nW} wells, {len(ev):,} rows | strong {len(kill)}, lateral {len(lat)}, "
          f"shape {len(sh)}, MV {len(mv)} (MV1-4 {len(mv4)})\n")

    AXES = {"shape channel": ("term", 3),
            "MV family (4)": ("term", 5),
            "MV family (all)": ("term", 4),
            "tail constant correction": ("tail", None)}

    def read(kind, col, m, q, w, k, mask=None):
        if kind == "term":
            T = (1 - m - q) * H.A[0] + m * H.A[1] + (q * H.A[col] if q else 0.0)
            P = H.B + w * gate.mul(k).exp() * (T - H.B)
        else:
            T = (1 - m) * H.A[0] + m * H.A[1]
            P = H.B + w * gate.mul(k).exp() * (T - H.B)
            if q:                       # q is alpha here; the gate needs a threshold, not sigmoid(0)
                s = torch.sigmoid(2.0 * (zs - 1.5))
                mvmean = torch.zeros(nW, device=DEV).index_add_(
                    0, H.wid, (T - H.B)) / torch.bincount(H.wid, minlength=nW).clamp_min(1)
                P = P - q * (s * mvmean)[H.wid]
        return float(H.rms(P, mask))

    def tune(kind, col, weights, mask):
        best = None
        for m in LAT:
            for q in weights:
                for w in W0:
                    for k in K:
                        v = read(kind, col, m, q, w, k, mask)
                        if best is None or v < best[0]:
                            best = (v, m, q, w, k)
        return best[1:]

    HOLD = []
    for kindr in ("isopach", "xy"):
        wl, lab = regions(wells, kindr)
        pos = {w: i for i, w in enumerate(wl)}
        wlab = np.array([lab[pos[w]] for w in wells])
        rs = sorted(set(lab[~pd.isna(lab)]))
        sizes = [int((wlab == r).sum()) for r in rs]
        print(f"  {kindr} block sizes: {sizes}")
        for r in rs:
            h = np.isin(wid, np.flatnonzero(wlab == r))
            HOLD.append((f"{kindr}{int(r)}",
                         torch.as_tensor(h, device=DEV),
                         torch.as_tensor(~h, device=DEV)))
    print(f"  {len(HOLD)} held-out regions\n")

    print(f"{'axis':<26} {'in-sample':>10} {'honest':>9} {'wins':>6}  {'ratio':>6}")
    out = []
    for axis, (kind, col) in AXES.items():
        W = ALPHA if kind == "tail" else WT
        ia, ib, da, db = [], [], [], []
        # tuned on EVERYTHING once; the same parameters are then read on each region
        pa = tune(kind, col, (0.0,), None); pb = tune(kind, col, W, None)
        for nm, hold, rest in HOLD:
            ia.append(read(kind, col, *pa, hold)); ib.append(read(kind, col, *pb, hold))
            qa = tune(kind, col, (0.0,), rest); qb = tune(kind, col, W, rest)
            da.append(read(kind, col, *qa, hold)); db.append(read(kind, col, *qb, hold))
        ins = float(np.mean(ib) - np.mean(ia))
        hon = float(np.mean(db) - np.mean(da))
        win = sum(1 for x, y in zip(db, da) if x < y)
        ratio = ins / hon if hon < -1e-9 else float("nan")
        print(f"{axis:<26} {ins:>10.4f} {hon:>9.4f} {win:>4}/{len(HOLD)}  {ratio:>6.1f}x")
        out.append((axis, ins, hon, win, len(HOLD), ratio))

    pd.DataFrame(out, columns=["axis", "in_sample", "honest", "wins", "regions", "ratio"]).to_csv(
        "stride/figs/claim1.csv", index=False)
    print("\nwrote stride/figs/claim1.csv")
    print("sign test: 7/8 p=0.035, 6/8 p=0.145, 5/8 chance")


if __name__ == "__main__":
    main()
