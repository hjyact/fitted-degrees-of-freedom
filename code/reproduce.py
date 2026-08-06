"""Regenerate every figure and table in the paper, and check they came out the same.

A paper that argues for measurement discipline should be able to rebuild its own numbers with one
command. This runs the four figure scripts in order, times them, and diffs the resulting CSVs
against what is on disk so that a silent change in an upstream input shows up as a diff rather than
as a quietly different figure.

    python -m stride.reproduce            # rebuild and compare
    python -m stride.reproduce --check    # compare only, rebuild nothing

Everything runs off `clipbase/sub465.npy`, which `stride.subframe` builds once from the 29 GB track
matrix. If that cache is missing this rebuilds it first, and that step alone takes minutes; the rest
takes seconds.
"""
from __future__ import annotations
import argparse, hashlib, subprocess, sys, time
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "stride" / "figs"
CACHE = REPO / "stride" / "clipbase" / "sub465.npy"

STEPS = [
    ("subframe",         "stride.subframe",         [],                       "cache"),
    ("Figure 1  motivation", "stride.fig_motivation", ["motivation.csv"],      "fig"),
    ("Figure 2  method",     "stride.fig_method",     ["method_wells.csv"],    "fig"),
    ("Figure 3  dof curve",  "stride.dofcurve2",      ["dofcurve2.csv"],       "fig"),
    ("Figure 4  frontier",   "stride.frontier_scatter", ["frontier.csv"],      "fig"),
    ("Table 1a in/honest",   "stride.claim1",         ["claim1.csv"],          "tab"),
    ("Table 1b k-dof price", "stride.claim1_price",   ["claim1_price.csv"],    "tab"),
]
# frontier_scatter and claim1* still read the full 29 GB frame; flagged so the timing is not a
# surprise and so a future cleanup knows which scripts are worth porting to load_sub().
SLOW = {"stride.frontier_scatter", "stride.claim1", "stride.claim1_price"}


def digest(p: Path) -> str:
    if not p.exists():
        return "missing"
    d = pd.read_csv(p).round(6)
    return hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare only, rebuild nothing")
    ap.add_argument("--fast", action="store_true", help="skip the steps that read the 29 GB frame")
    a = ap.parse_args()

    before = {f: digest(FIGS / f) for _, _, outs, _ in STEPS for f in outs}
    if a.check:
        print(f"{'artefact':<22} {'digest':>14}")
        for f, d in before.items():
            print(f"  {f:<20} {d:>14}")
        return

    print(f"{'step':<24} {'seconds':>8}  {'artefacts':<28} {'status'}")
    for name, mod, outs, kind in STEPS:
        if kind == "cache" and CACHE.exists():
            print(f"{name:<24} {'--':>8}  {'sub465.npy':<28} present")
            continue
        if a.fast and mod in SLOW:
            print(f"{name:<24} {'--':>8}  {','.join(outs):<28} skipped (--fast)")
            continue
        t = time.time()
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", mod],
                           cwd=REPO, capture_output=True, text=True)
        dt = time.time() - t
        if r.returncode != 0:
            print(f"{name:<24} {dt:>8.0f}  {','.join(outs):<28} FAILED")
            print(r.stdout[-600:]); print(r.stderr[-600:])
            continue
        st = []
        for f in outs:
            now = digest(FIGS / f)
            st.append("same" if now == before.get(f) else
                      ("new" if before.get(f) == "missing" else "CHANGED"))
        print(f"{name:<24} {dt:>8.0f}  {','.join(outs):<28} {', '.join(st) or 'ok'}")

    print("\nA 'CHANGED' above means an input moved under the paper. Find out what before citing it.")


if __name__ == "__main__":
    main()
