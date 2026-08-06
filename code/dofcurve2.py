"""Figure 2, final version: both families, a dense k grid, 100 draws each.

`dofcurve` established the shape on one family with 12 draws per k and a coarse grid, off the 29 GB
matrix at 8 minutes a run. With `subframe`'s 0.77 GB cache in RAM the same sweep is seconds, so the
three things the outline lists as gaps close together:

  * the shape family (37) as well as MV (50) -- otherwise a reader can call the collapse an MV quirk
  * 100 draws per k instead of 12, so the error bars stop dominating the figure
  * k = 34, 36, 38 filled in, so the zero crossing is located rather than bracketed

Same protocol as before: NNLS cross-fitted by well, read on the same 8 held-out regions, everything
computed from sufficient statistics so a draw costs a 50x50 slice.

    n * mse(R) = sum(rr^2) + 2 w.(D^T rr)_R + w^T (D^T D)_R w,   rr = PA - y,  G w = -(D^T rr)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.optimize import nnls

from stride.subframe import load_sub
from stride.regionout import regions

LAT = (0.10, 0.15, 0.20, 0.25)
W0 = (0.75, 0.85, 0.95)
KG = (0.0, -0.05, -0.10)
NFOLD, REPEAT = 5, 100
KS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 34, 36, 38, 40, 45, 50)


def zscore(v, lo=-2.0, hi=3.0):
    return np.clip((v - v.mean()) / max(v.std(), 1e-9), lo, hi)


def per_well_z(v, wid, nW, lo=-2.0, hi=3.0):
    n = np.bincount(wid, minlength=nW).clip(1)
    m = np.bincount(wid, weights=v, minlength=nW) / n
    d = v - m[wid]
    s = np.sqrt(np.bincount(wid, weights=d * d, minlength=nW) / n).clip(1e-6)
    return np.clip(d / s[wid], lo, hi)


def sweep(fam_name, cols_idx, M, meta, ev, wid, nW, PA, y, reg, fold, rng):
    rr = (PA - y).astype(np.float64)
    D = M[cols_idx].astype(np.float64).T - PA[:, None]
    kmax = D.shape[1]
    S = {}
    for f in range(NFOLD):
        tr = fold != f
        S[("tr", f)] = (D[tr].T @ D[tr], -(D[tr].T @ rr[tr]))
        for nm, h in reg:
            m_ = h & ~tr
            if not m_.any():
                continue
            S[(nm, f)] = (int(m_.sum()), float((rr[m_] ** 2).sum()),
                          D[m_].T @ rr[m_], D[m_].T @ D[m_])
    base = {nm: float(np.sqrt((rr[h] ** 2).mean())) for nm, h in reg}
    rows = []
    for k in KS:
        if k > kmax:
            continue
        for rep in range(1 if k == kmax else REPEAT):
            c = np.sort(rng.choice(kmax, k, replace=False))
            wts = {f: nnls(S[("tr", f)][0][np.ix_(c, c)], S[("tr", f)][1][c])[0]
                   for f in range(NFOLD)}
            d = []
            for nm, _ in reg:
                n = sse = 0.0
                for f in range(NFOLD):
                    if (nm, f) not in S:
                        continue
                    cnt, s2, v, G = S[(nm, f)]
                    w = wts[f]
                    n += cnt
                    sse += s2 + 2 * w @ v[c] + w @ G[np.ix_(c, c)] @ w
                d.append(np.sqrt(max(sse, 0) / max(n, 1)) - base[nm])
            rows.append((fam_name, k, rep, float(np.mean(d)), sum(1 for x in d if x < 0)))
    return rows


def main():
    ev, wid, M, meta = load_sub()
    nW = int(wid.max() + 1)
    y = ev.ytrue.to_numpy(np.float64)
    B = ev.base.to_numpy(np.float64)
    Ak, Al, Au = M[0].astype(np.float64), M[1].astype(np.float64), M[2].astype(np.float64)
    gate = 0.5 * (per_well_z(np.log(np.clip(Au, 1e-3, None)), wid, nW)
                  + zscore(np.log(np.abs(Ak - Al).clip(1e-3))))
    best = None
    for m in LAT:
        for w in W0:
            for k in KG:
                T = (1 - m) * Ak + m * Al
                P = B + w * np.exp(k * gate) * (T - B)
                v = float(np.sqrt(((P - y) ** 2).mean()))
                if best is None or v < best[0]:
                    best = (v, m, w, k)
    _, m0, w0, k0 = best
    T0 = (1 - m0) * Ak + m0 * Al
    PA = B + w0 * np.exp(k0 * gate) * (T0 - B)
    print(f"{nW} wells, {len(ev):,} rows | baseline {best[0]:.4f} "
          f"(lat {m0:.2f} w0 {w0:.2f} k {k0:+.2f})", flush=True)

    wells = pd.factorize(ev.well)[1]
    reg = []
    for kindr in ("isopach", "xy"):
        wl, lab = regions(wells, kindr)
        pos = {w: i for i, w in enumerate(wl)}
        wlab = np.array([lab[pos[w]] for w in wells])
        for r in sorted(set(lab[~pd.isna(lab)])):
            reg.append((f"{kindr}{int(r)}", np.isin(wid, np.flatnonzero(wlab == r))))
    fold = np.random.default_rng(0).integers(0, NFOLD, nW)[wid]
    rng = np.random.default_rng(1)

    nmv, nsh = len(meta["mv"]), len(meta["shape"])
    rows = []
    rows += sweep("MV", np.arange(3, 3 + nmv), M, meta, ev, wid, nW, PA, y, reg, fold, rng)
    print("  MV done", flush=True)
    rows += sweep("shape", np.arange(3 + nmv, 3 + nmv + nsh), M, meta, ev, wid, nW,
                  PA, y, reg, fold, rng)
    print("  shape done", flush=True)

    d = pd.DataFrame(rows, columns=["family", "k", "rep", "gain", "wins"])
    d.to_csv("stride/figs/dofcurve2.csv", index=False)
    g = d.groupby(["family", "k"]).agg(gain=("gain", "mean"), sd=("gain", "std"),
                                       wins=("wins", "mean"), n=("rep", "count")).reset_index()
    for fam in ("MV", "shape"):
        gg = g[g.family == fam]
        print(f"\n=== {fam} ===\n{'k':>4} {'gain':>9} {'sd':>7} {'wins/8':>7} {'draws':>6}")
        for _, r in gg.iterrows():
            print(f"{int(r.k):>4} {r.gain:>9.4f} {0 if pd.isna(r.sd) else r.sd:>7.4f} "
                  f"{r.wins:>7.2f} {int(r.n):>6}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(0, color="k", lw=0.8)
    for fam, c in (("MV", "#2ca02c"), ("shape", "#1f77b4")):
        gg = g[g.family == fam]
        se = gg.sd / np.sqrt(gg.n.clip(1))
        ax.plot(gg.k, gg.gain, "o-", ms=3.5, lw=1.2, color=c, label=f"{fam} family")
        ax.fill_between(gg.k, gg.gain - se, gg.gain + se, color=c, alpha=0.2, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 50]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 50])
    ax.set_xlabel("k  (fitted weights, NNLS cross-fitted by well)")
    ax.set_ylabel("gain on held-out regions  (ft, negative is better)")
    ax.set_title("the gain collapses as fitted degrees of freedom grow")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig("stride/figs/dofcurve2.png", dpi=160)
    print("\nwrote stride/figs/dofcurve2.{csv,png}")


if __name__ == "__main__":
    main()
