"""Figure 3 (Section 2): what a member does, on real wells.

A schematic of the state space would say less than the data does. Three wells are drawn: the truth,
the carrier baseline the ensemble starts from, and the ensemble's answer. One is typical, one is a
tail well where the ensemble moves a long way and wins, one is a tail well where it moves and loses.

The point the figure has to carry into Section 4 is that a member's contribution is a MOVE from the
carrier, not an independent prediction -- so what an ensemble weight buys is the size and shape of
that move, and that is the quantity Section 4 counts degrees of freedom over.
"""
from __future__ import annotations
import numpy as np, pandas as pd

from stride.subframe import load_sub

MIX, MVW, W0, K, CAP = 0.15, 0.10, 0.95, 0.0, 90.0


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
    Av = M[3:3 + nmv].astype(np.float64).mean(0)
    gate = 0.5 * (per_well_z(np.log(np.clip(Au, 1e-3, None)), wid, nW)
                  + zscore(np.log(np.abs(Ak - Al).clip(1e-3))))
    cnt = np.bincount(wid, minlength=nW).clip(1)
    st = np.cumsum(cnt) - cnt
    frac = ((np.arange(len(ev)) - st[wid]) / np.clip(cnt[wid] - 1, 1, None)).clip(0, 1)
    T = (1 - MIX - MVW) * Ak + MIX * Al + MVW * Av
    P = B + np.clip(W0 * np.exp(K * gate) * (T - B), -CAP * frac, CAP * frac)

    eB = np.bincount(wid, weights=(B - y) ** 2, minlength=nW) / cnt
    eP = np.bincount(wid, weights=(P - y) ** 2, minlength=nW) / cnt
    move = np.bincount(wid, weights=np.abs(P - B), minlength=nW) / cnt
    d = pd.DataFrame({"w": np.arange(nW), "rmsB": np.sqrt(eB), "rmsP": np.sqrt(eP),
                      "move": move, "n": cnt})
    d["gain"] = d.rmsP - d.rmsB
    d = d[d.n > 2000]
    typ = d.iloc[(d.rmsP - d.rmsP.median()).abs().argsort()].iloc[0]
    win = d.sort_values("gain").iloc[0]
    lose = d.sort_values("gain").iloc[-1]
    print(f"{nW} wells | ensemble {np.sqrt(((P-y)**2).mean()):.4f}, carrier "
          f"{np.sqrt(((B-y)**2).mean()):.4f}")
    for nm, r in (("typical", typ), ("tail, wins", win), ("tail, loses", lose)):
        print(f"  {nm:<12} well {int(r.w):>3}  carrier {r.rmsB:6.2f} -> {r.rmsP:6.2f} "
              f"({r.gain:+.2f})  mean move {r.move:5.2f} ft")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=False)
    for ax, (nm, r) in zip(axes, (("typical well", typ), ("tail well, ensemble wins", win),
                                  ("tail well, ensemble loses", lose))):
        m = wid == int(r.w)
        x = np.arange(int(m.sum()))
        ax.plot(x, y[m], color="k", lw=1.4, label="truth")
        ax.plot(x, B[m], color="#7f7f7f", lw=1.0, ls="--", label="carrier (baseline)")
        ax.plot(x, P[m], color="#2ca02c", lw=1.2, label="ensemble")
        ax.set_title(f"{nm}\n{r.rmsB:.1f} → {r.rmsP:.1f} ft", fontsize=9)
        ax.set_xlabel("row along the hidden section")
    axes[0].set_ylabel("dTVT (ft)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("stride/figs/method.png", dpi=160)
    d.to_csv("stride/figs/method_wells.csv", index=False)
    print("\nwrote stride/figs/method.png and method_wells.csv")


if __name__ == "__main__":
    main()
