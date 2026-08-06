"""What does building fewer members cost, when they enter as a MEAN at a fixed weight?

Run for all three deployed families, because a curve measured on one of them invites the objection
that it is a property of that family. The three enter the answer differently -- strong and lateral
are mixed against each other at weight (1-m-v) and m, MV rides at a fixed v -- so each family's
count is varied with the other two held at their full membership.

`dofcurve2` answers a different question -- k FITTED weights -- and finds the gain collapsing. The
deployed kernel does not fit anything: it averages whatever members it built and carries the average
at one fixed weight. So the question that decides the member budget is

    average n members, add at the fixed weight, read on the eight held-out blocks

which should IMPROVE with n and then saturate, because averaging reduces the mean's noise without
adding a parameter. Where it saturates is the answer to "is 40 of 207 enough".

Same protocol and blocks as everywhere else; runs off the in-RAM subframe.
"""
from __future__ import annotations
import numpy as np, pandas as pd

from stride.subframe import load_sub
from stride.regionout import regions

MIX, MVW, W0, GAM, CAP = 0.15, 0.10, 0.95, 0.0, 90.0
NS = (1, 2, 3, 4, 6, 8, 11, 16, 22, 30, 40, 50, 59, 76)
REPEAT = 100


def zscore(v, lo=-2.0, hi=3.0):
    return np.clip((v - v.mean()) / max(v.std(), 1e-9), lo, hi)


def per_well_z(v, wid, nW, lo=-2.0, hi=3.0):
    n = np.bincount(wid, minlength=nW).clip(1)
    m = np.bincount(wid, weights=v, minlength=nW) / n
    d = v - m[wid]
    s = np.sqrt(np.bincount(wid, weights=d * d, minlength=nW) / n).clip(1e-6)
    return np.clip(d / s[wid], lo, hi)


def main():
    ev, wid, M, meta = load_sub()
    nW = int(wid.max() + 1)
    y = ev.ytrue.to_numpy(np.float64)
    B = ev.base.to_numpy(np.float64)
    Ak, Al, Au = (M[i].astype(np.float64) for i in range(3))
    nmv = len(meta["mv"])
    MVm = M[3:3 + nmv].astype(np.float64)
    gate = 0.5 * (per_well_z(np.log(np.clip(Au, 1e-3, None)), wid, nW)
                  + zscore(np.log(np.abs(Ak - Al).clip(1e-3))))
    cnt = np.bincount(wid, minlength=nW).clip(1)
    st = np.cumsum(cnt) - cnt
    frac = ((np.arange(len(ev)) - st[wid]) / np.clip(cnt[wid] - 1, 1, None)).clip(0, 1)

    wells = pd.factorize(ev.well)[1]
    reg = []
    for kindr in ("isopach", "xy"):
        wl, lab = regions(wells, kindr)
        pos = {w: i for i, w in enumerate(wl)}
        wlab = np.array([lab[pos[w]] for w in wells])
        for r in sorted(set(lab[~pd.isna(lab)])):
            reg.append((f"{kindr}{int(r)}", np.isin(wid, np.flatnonzero(wlab == r))))

    nsh = len(meta["shape"])
    o = 3 + nmv + nsh
    STm = M[o:o + len(meta["strong"])].astype(np.float64)
    LAm = M[o + len(meta["strong"]):o + len(meta["strong"]) + len(meta["lateral"])].astype(np.float64)
    print(f"{nW} wells | pools: strong {len(STm)}, lateral {len(LAm)}, MV {len(MVm)}")

    def score(k_, l_, v_):
        T = (1 - MIX - MVW) * k_ + MIX * l_ + MVW * v_
        return B + np.clip(W0 * np.exp(GAM * gate) * (T - B), -CAP * frac, CAP * frac)

    FULL = {"strong": STm.mean(0), "lateral": LAm.mean(0), "MV": MVm.mean(0)}
    full = score(FULL["strong"], FULL["lateral"], FULL["MV"])
    base_rms = {nm: float(np.sqrt(((full - y)[h] ** 2).mean())) for nm, h in reg}
    print(f"  all members: {float(np.sqrt(((full-y)**2).mean())):.4f} ft\n")

    rng = np.random.default_rng(1)
    rows = []
    for fam, pool in (("strong", STm), ("lateral", LAm), ("MV", MVm)):
        for n in NS:
            if n > len(pool):
                continue
            for rep in range(1 if n == len(pool) else REPEAT):
                c = rng.choice(len(pool), n, replace=False)
                arg = dict(FULL); arg[fam] = pool[c].mean(0)
                P = score(arg["strong"], arg["lateral"], arg["MV"])
                d = [float(np.sqrt(((P - y)[h] ** 2).mean())) - base_rms[nm] for nm, h in reg]
                rows.append((fam, n, rep, float(np.mean(d)), sum(1 for x in d if x < 0)))
        # the full pool as the zero point of its own curve
        rows.append((fam, len(pool), 0, 0.0, 0))
    d = pd.DataFrame(rows, columns=["family", "n", "rep", "gain", "wins"])
    g = d.groupby(["family", "n"]).agg(gain=("gain", "mean"), sd=("gain", "std"),
                                       wins=("wins", "mean"), k=("rep", "count")).reset_index()
    for fam in ("strong", "lateral", "MV"):
        gg = g[g.family == fam]
        print(f"=== {fam} (pool {int(gg.n.max())}) ===")
        print(f"{'n':>4} {'gain vs full':>13} {'sd':>8}")
        for _, r in gg.iterrows():
            print(f"{int(r.n):>4} {r.gain:>+13.4f} {0 if pd.isna(r.sd) else r.sd:>8.4f}")
        print()
    d.to_csv("stride/figs/countcurve.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0, color="k", lw=0.8)
    for fam, c in (("strong", "#1f77b4"), ("lateral", "#d62728"), ("MV", "#2ca02c")):
        gg = g[g.family == fam]
        se = (gg.sd / np.sqrt(gg.k.clip(1))).fillna(0)
        ax.plot(gg.n, gg.gain, "o-", ms=4, lw=1.3, color=c, label=f"{fam} (pool {int(gg.n.max())})")
        ax.fill_between(gg.n, gg.gain - se, gg.gain + se, color=c, alpha=0.18, linewidth=0)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xscale("log"); ax.set_xticks([1,2,4,8,16,30,50,76]); ax.set_xticklabels([1,2,4,8,16,30,50,76])
    ax.set_xlabel("members averaged into the family mean")
    ax.set_ylabel("gain vs using the whole pool  (ft)")
    ax.set_title("averaging saturates early: the member budget is cheap")
    fig.tight_layout(); fig.savefig("stride/figs/countcurve.png", dpi=160)
    print("\nwrote stride/figs/countcurve.{csv,png}")


if __name__ == "__main__":
    main()
