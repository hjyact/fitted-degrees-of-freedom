"""Figure 1 of the paper: every member lands on the break-even frontier.

`frontier_law` states the identity and lists five estimators. This regenerates the full scatter from
the member bank so the claim has its own data rather than a docstring.

For each member m, with e_m = m - y and e_S = S - y for the stack S:

    sigma_m      = rms(e_m)
    rho_m        = corr(e_m, e_S)
    break-even   = sigma_S / sigma_m

A member helps the stack only if rho_m is BELOW its break-even line. The claim is that essentially
every member -- across mechanisms that share no evidence -- sits on the line, so accuracy tuning
moves a member ALONG it and buys nothing; only new evidence moves off it.

Writes stride/figs/frontier.csv (one row per member) and stride/figs/frontier.png.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd, torch

from stride.gsel import Harness, load, DEV
from stride.score465 import families, row_frac
from stride.rowmod import per_well_z, zscore

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
M, K, CAP, W0, MVW = 0.15, 0.00, 90.0, 0.95, 0.10
MVP = ("MV", "MW", "MX", "MY", "UW", "UV", "SO", "SN")


def main():
    ev, wid, names, A = load("465")
    kill, lat = families(names)
    idx = {n: i for i, n in enumerate(names)}
    mv = [n for n in names if "__" not in n and n[:2] in MVP and n[2:].strip("0123456789") == ""]
    gg = lambda L: A[[idx[n] for n in L]].mean(0, keepdims=True)
    usd = [f"{n}__usd" for n in kill if f"{n}__usd" in idx]
    H = Harness(ev, wid, np.concatenate([gg(kill), gg(lat), gg(usd), gg(mv)]))

    gate = 0.5 * (per_well_z(H, torch.log(H.A[2].clamp_min(1e-3)))
                  + zscore(torch.log((H.A[0] - H.A[1]).abs().clamp_min(1e-3))))
    p = row_frac(H)
    T = (1 - M - MVW) * H.A[0] + M * H.A[1] + MVW * H.A[3]
    S = H.B + (W0 * gate.mul(K).exp() * (T - H.B)).clamp(-CAP * p, CAP * p)

    y = H.y
    eS = (S - y).double()
    sS = float(eS.pow(2).mean().sqrt())
    eS = eS - eS.mean()
    nS = float(eS.pow(2).sum().sqrt())
    print(f"{H.nW} wells, {len(ev):,} rows; stack sigma {sS:.4f}, base {float(H.rms(H.B)):.4f}")

    fam = {n: ("strong" if n in set(kill) else "lateral" if n in set(lat)
               else "MV" if n in set(mv) else "other") for n in names}
    rows = []
    members = [n for n in names if "__" not in n]
    for nm in members:
        t = torch.as_tensor(A[idx[nm]], device=DEV)
        e = (t - y).double()
        sm = float(e.pow(2).mean().sqrt())
        e = e - e.mean()
        rho = float((e * eS).sum() / (e.pow(2).sum().sqrt() * nS).clamp_min(1e-12))
        rows.append((nm, fam[nm], sm, rho, sS / sm))
    d = pd.DataFrame(rows, columns=["member", "family", "sigma", "rho", "breakeven"])
    d["ratio"] = d.rho / d.breakeven
    d.to_csv(OUT / "frontier.csv", index=False)
    print(f"\n{len(d)} members")
    print(f"  ratio rho/breakeven: median {d.ratio.median():.3f}, "
          f"min {d.ratio.min():.3f}, below 1: {int((d.ratio < 1).sum())}")
    for f in ("strong", "lateral", "MV", "other"):
        g = d[d.family == f]
        if len(g):
            print(f"  {f:8s} n={len(g):3d}  sigma {g.sigma.median():6.2f}  "
                  f"ratio median {g.ratio.median():.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.linspace(max(d.sigma.min() * 0.95, sS * 1.01), d.sigma.max() * 1.05, 200)
    ax.plot(x, sS / x, "k-", lw=1.2, label=r"break-even  $\rho=\sigma_S/\sigma_m$")
    for f, c, mk in (("strong", "#1f77b4", "o"), ("lateral", "#d62728", "s"),
                     ("MV", "#2ca02c", "^"), ("other", "#7f7f7f", ".")):
        g = d[d.family == f]
        if len(g):
            ax.scatter(g.sigma, g.rho, s=18, c=c, marker=mk, alpha=0.75, label=f, linewidths=0)
    ax.set_xlabel(r"member error $\sigma_m$  (ft)")
    ax.set_ylabel(r"$\mathrm{corr}(e_m,\,e_S)$")
    ax.set_title(f"{len(d)} members on one frontier  (stack $\\sigma_S$={sS:.2f} ft)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "frontier.png", dpi=160)
    print(f"\nwrote {OUT/'frontier.csv'} and {OUT/'frontier.png'}")


if __name__ == "__main__":
    main()
