"""Figure 0: why this metric has very few effective degrees of freedom per well.

Section 1 claims two things and currently shows neither: that the error is concentrated in a few
wells, and that within a well it is mostly a constant and a slope. Both are oracle measurements --
the per-well fit is made ON the held-out rows, so it is an upper bound on what any method could
recover, not a result.

Left panel   cumulative share of squared error against wells ranked worst-first.
Right panel  the oracle ladder: what remains after removing, per well, a constant / a slope /
             both, as a fraction of the error the ensemble actually makes.

Runs off the in-RAM subframe, so seconds rather than minutes.
"""
from __future__ import annotations
import numpy as np, pandas as pd

from stride.subframe import load_sub

LAT = (0.10, 0.15, 0.20, 0.25)
W0 = (0.75, 0.85, 0.95)
KG = (0.0, -0.05, -0.10)


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
    gate = 0.5 * (per_well_z(np.log(np.clip(Au, 1e-3, None)), wid, nW)
                  + zscore(np.log(np.abs(Ak - Al).clip(1e-3))))
    # the SHIPPED design, so this figure and the frontier figure describe the same ensemble:
    # strong + lateral + MV at the deployed constants (m .15, mv .10, k 0, cap 90, w0 .95)
    nmv = len(meta["mv"])
    Av = M[3:3 + nmv].astype(np.float64).mean(0)
    m0, mv0, w0_, k0 = 0.15, 0.10, 0.95, 0.0
    cnt = np.bincount(wid, minlength=nW).clip(1)
    st = np.cumsum(cnt) - cnt
    frac = ((np.arange(len(ev)) - st[wid]) / np.clip(cnt[wid] - 1, 1, None)).clip(0, 1)
    T0 = (1 - m0 - mv0) * Ak + m0 * Al + mv0 * Av
    P = B + np.clip(w0_ * np.exp(k0 * gate) * (T0 - B), -90.0 * frac, 90.0 * frac)
    best = (float(np.sqrt(((P - y) ** 2).mean())), m0, w0_, k0)
    e = P - y
    sse = float((e ** 2).sum())
    print(f"{nW} wells, {len(ev):,} rows | ensemble {best[0]:.4f} ft", flush=True)

    # per-well oracle removals, fitted on the very rows they are scored on
    n = np.bincount(wid, minlength=nW).clip(1)
    x = np.arange(len(ev), dtype=np.float64)
    first = np.cumsum(n) - n
    t = (x - first[wid]) / np.clip(n[wid] - 1, 1, None)          # 0..1 along each well
    s = lambda v: np.bincount(wid, weights=v, minlength=nW)
    S1, St, Stt = s(np.ones_like(e)), s(t), s(t * t)
    Se, Ste = s(e), s(t * e)
    const = Se / S1
    det = (S1 * Stt - St * St)
    slope_only = Ste / np.where(Stt > 0, Stt, 1)
    b_ = (S1 * Ste - St * Se) / np.where(np.abs(det) > 1e-9, det, 1)
    a_ = (Se - b_ * St) / S1
    lad = {"ensemble": e,
           "− constant": e - const[wid],
           "− slope": e - slope_only[wid] * t,
           "− both": e - (a_[wid] + b_[wid] * t)}
    print(f"\n{'removal':>12} {'rms':>8} {'share of SSE':>14}")
    bars = []
    for k, v in lad.items():
        r = float(np.sqrt((v ** 2).mean()))
        share = 1 - float((v ** 2).sum()) / sse
        print(f"{k:>12} {r:>8.4f} {share*100:>13.1f}%")
        bars.append((k, r, share))

    pw = np.bincount(wid, weights=e ** 2, minlength=nW)
    order = np.argsort(-pw)
    cum = np.cumsum(pw[order]) / pw.sum()
    for q in (5, 20, 40, 100):
        print(f"  worst {q:>3} wells: {cum[q-1]*100:.1f}% of SSE")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax1.plot(np.arange(1, nW + 1), cum * 100, lw=1.6, color="#1f77b4")
    for q, c in ((20, "#d62728"), (40, "#ff7f0e")):
        ax1.plot([q, q], [0, cum[q - 1] * 100], ls=":", color=c, lw=1)
        ax1.annotate(f"worst {q}: {cum[q-1]*100:.0f}%", (q, cum[q - 1] * 100),
                     textcoords="offset points", xytext=(8, -12), fontsize=8, color=c)
    ax1.set_xlabel("wells, ranked worst first"); ax1.set_ylabel("cumulative share of SSE (%)")
    ax1.set_title("the error is concentrated"); ax1.set_ylim(0, 100)

    names = [b[0] for b in bars]; rms = [b[1] for b in bars]
    ax2.bar(names, rms, color=["#7f7f7f", "#1f77b4", "#1f77b4", "#2ca02c"])
    for i, (nm, r, sh) in enumerate(bars):
        ax2.annotate(f"{r:.2f}" + (f"\n−{sh*100:.0f}% SSE" if i else ""),
                     (i, r), ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("pooled RMSE (ft)")
    ax2.set_title("an oracle line per well removes most of it")
    ax2.set_ylim(0, max(rms) * 1.25)
    fig.tight_layout(); fig.savefig("stride/figs/motivation.png", dpi=160)
    pd.DataFrame(bars, columns=["removal", "rms", "share"]).to_csv(
        "stride/figs/motivation.csv", index=False)
    print("\nwrote stride/figs/motivation.{csv,png}")


if __name__ == "__main__":
    main()
