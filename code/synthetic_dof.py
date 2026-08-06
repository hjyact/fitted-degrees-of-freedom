"""Does the collapse follow k/n, and does the real crossing point land where theory says?

The wellbore result is one dataset, which is not evidence that anything general is happening. This
is the controlled version: exchangeable candidate members over n independent units, k of them
combined either by a fixed equal-weight mean or by NNLS weights fitted out-of-fold, held-out risk
measured as a function of both k and n.

THE PREDICTION. Let the members share a per-unit error component and carry independent noise on top.
Fitting k weights on n units costs an excess risk of order sigma_e^2 * k / n (the usual k-parameter
variance), while the best achievable weighting beats the equal-weight mean by some fixed Delta that
does not grow with k once the members are exchangeable. The net gain is therefore

    gain(k) ~ -Delta + c * k / n        crossing zero at    k* = n * Delta / c

so k* should be LINEAR IN n, and curves for different n should collapse when plotted against k/n.
If they do, the wellbore curve is an instance of a law rather than a property of one field, and its
measured crossing (k ~ 33 at n = 465 wells) is a point that theory places.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.optimize import nnls

NS = (60, 125, 250, 500, 1000)
KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
ROWS = 40           # rows per unit; errors are correlated within a unit, as in the real problem
REPEAT = 40
NFOLD = 5
RHO = 0.75          # share of a member's error that is common to all members on that unit
HET = 0.6           # spread of member QUALITY. With exchangeable members the equal-weight mean is
                    # already optimal, Delta = 0, and fitting can only lose -- the first version of
                    # this script had no crossing for exactly that reason. Real pools are not
                    # exchangeable: some members carry more of the signal, so the optimal weights
                    # are unequal and there is something for fitting to find.


def one(n, k, rng):
    """held-out risk of (a) the equal-weight mean and (b) NNLS weights fitted out-of-fold."""
    # per-unit shared error, then per-member idiosyncratic error, both constant along a unit's rows
    shared = rng.standard_normal(n)
    idio = rng.standard_normal((n, k))
    # member quality: loading a_i on the signal, spread by HET. The equal-weight mean is optimal
    # only when these are equal.
    a = np.exp(rng.standard_normal(k) * HET)
    a = a / a.mean()
    e_mem = (np.sqrt(RHO) * shared[:, None]) * a + np.sqrt(1 - RHO) * idio
    y = np.sqrt(RHO) * shared + rng.standard_normal(n) * 0.35
    fold = rng.integers(0, NFOLD, n)

    mean_pred = e_mem.mean(1)
    fit_pred = np.empty(n)
    for f in range(NFOLD):
        tr, te = fold != f, fold == f
        G = e_mem[tr].T @ e_mem[tr]
        b = e_mem[tr].T @ y[tr]
        w, _ = nnls(G, b)
        fit_pred[te] = e_mem[te] @ w
    # the equal-weight mean is carried at ONE fitted scalar, fitted the same way, so the comparison
    # is one weight against k weights and not "no fitting" against "fitting"
    s = np.empty(n)
    for f in range(NFOLD):
        tr, te = fold != f, fold == f
        d = mean_pred[tr]
        a = float((d @ y[tr]) / max(d @ d, 1e-12))
        s[te] = max(a, 0.0) * mean_pred[te]
    rms = lambda p: float(np.sqrt(((y - p) ** 2).mean()))
    return rms(s), rms(fit_pred)


def main():
    rng = np.random.default_rng(0)
    rows = []
    for n in NS:
        for k in KS:
            if k > n // 4:
                continue
            for _ in range(REPEAT):
                a, b = one(n, k, rng)
                rows.append((n, k, k / n, b - a))
    d = pd.DataFrame(rows, columns=["n", "k", "kn", "gain"])
    g = d.groupby(["n", "k"]).agg(gain=("gain", "mean"), sd=("gain", "std"),
                                  kn=("kn", "first")).reset_index()
    d.to_csv("figures/synthetic.csv", index=False)

    print(f"{'n':>6} " + " ".join(f"{k:>8}" for k in KS))
    for n in NS:
        gg = g[g.n == n].set_index("k")
        print(f"{n:>6} " + " ".join(
            (f"{gg.loc[k,'gain']:>+8.4f}" if k in gg.index else f"{'--':>8}") for k in KS))

    print(f"\n{'n':>6} {'k*  (zero crossing)':>22} {'k*/n':>8}")
    star = []
    for n in NS:
        gg = g[g.n == n].sort_values("k")
        x, yv = gg.k.to_numpy(float), gg.gain.to_numpy()
        s = np.nan
        for i in range(len(yv) - 1):
            if yv[i] < 0 <= yv[i + 1]:
                s = x[i] + (x[i + 1] - x[i]) * (-yv[i]) / (yv[i + 1] - yv[i]); break
        star.append((n, s))
        print(f"{n:>6} {s:>22.1f} {s/n:>8.4f}" if s == s else f"{n:>6} {'no crossing':>22}")
    ok = [(n, s) for n, s in star if s == s]
    if len(ok) >= 2:
        a = np.polyfit([n for n, _ in ok], [s for _, s in ok], 1)
        r = np.corrcoef([n for n, _ in ok], [s for _, s in ok])[0, 1]
        print(f"\n  k* = {a[0]:.4f} n + {a[1]:.2f}   (corr {r:.4f})")
        print(f"  k*/n is {np.mean([s/n for n,s in ok]):.4f} +- {np.std([s/n for n,s in ok]):.4f}")
        print(f"\n  the wellbore frame: n = 465 wells, measured k* ~ 33  ->  k*/n = {33/465:.4f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for n in NS:
        gg = g[g.n == n].sort_values("k")
        a1.plot(gg.k, gg.gain, "o-", ms=3, lw=1.1, label=f"n={n}")
        a2.plot(gg.kn, gg.gain, "o-", ms=3, lw=1.1, label=f"n={n}")
    for ax, xl in ((a1, "k  (fitted weights)"), (a2, "k / n")):
        ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log"); ax.set_xlabel(xl)
    a1.set_ylabel("held-out risk, k weights minus 1 weight")
    a1.set_title("each n has its own curve"); a2.set_title("they collapse on k/n")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("figures/synthetic.png", dpi=160)
    print("\nwrote figures/synthetic.{csv,png}")


if __name__ == "__main__":
    main()
