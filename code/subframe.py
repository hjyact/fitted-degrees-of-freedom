"""A small in-RAM frame for the paper experiments.

`h465.load` maps a 29 GB track matrix on a machine with 29 GB of RAM, so every run pages the whole
thing off disk and costs 5-8 minutes before it computes anything. The paper's experiments do not
need 671 members: they need four family MEANS and the individual members of whichever family is
being swept.

This writes those rows once. `sub465.npy` is about 0.9 GB, fits in memory, and turns an 8-minute
startup into a few seconds.

    python -m stride.subframe          # build
    from stride.subframe import load_sub;  F = load_sub()

`F` carries: y, base, wid, the row fraction, the gate's ingredients, the four means, and the MV and
shape members by name.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

CB = Path(__file__).resolve().parent / "clipbase"
NPY, META = CB / "sub465.npy", CB / "sub465.json"
MVP = ("MV", "MW", "MX", "MY", "UW", "UV", "SO", "SN")
SHAPE = ("CH_n", "CN0", "CN1", "CN2", "CW0", "CW1", "CW4", "CW8",
         "SF25", "SFK", "SFsC", "S2a", "SK", "SL")


def build():
    from stride.gsel import load
    from stride.score465 import families, STRONG_EXTRA, LATERAL
    ev, wid, names, A = load("465")
    kill, lat = families(names)
    idx = {n: i for i, n in enumerate(names)}
    known = set(STRONG_EXTRA) | set(LATERAL)
    sh = [n for n in names if "__" not in n and n not in known and n.startswith(SHAPE)]
    mv = [n for n in names if "__" not in n and n[:2] in MVP and n[2:].strip("0123456789") == ""]
    us = [f"{n}__usd" for n in kill if f"{n}__usd" in idx]

    # LAYOUT IS APPEND-ONLY. Rows 0-2 are the three means and 3.. the MV then shape members;
    # anything added later goes at the END so that code indexing 3:3+len(mv) keeps working.
    rows, meta = [], {"means": ["strong", "lateral", "usd"], "mv": mv, "shape": sh,
                      "strong": kill, "lateral": lat,
                      "n_strong": len(kill), "n_lateral": len(lat), "n_usd": len(us)}
    for grp in (kill, lat, us):
        rows.append(A[[idx[n] for n in grp]].mean(0).astype(np.float32))
    for n in mv + sh + kill + lat:
        rows.append(np.asarray(A[idx[n]], dtype=np.float32))
    M = np.stack(rows)
    np.save(NPY, M)
    ev[["id", "well", "row", "ytrue", "base"]].to_parquet(CB / "sub465_ev.parquet", index=False)
    META.write_text(json.dumps(meta))
    print(f"wrote {NPY} {M.shape} ({M.nbytes/2**30:.2f} GB): 3 means + "
          f"{len(mv)} MV + {len(sh)} shape + {len(kill)} strong + {len(lat)} lateral")


def load_sub():
    """(ev, wid, M, meta) with M held in RAM, not mapped."""
    ev = pd.read_parquet(CB / "sub465_ev.parquet")
    meta = json.loads(META.read_text())
    M = np.load(NPY)
    wid = pd.factorize(ev.well)[0]
    return ev, wid, M, meta


if __name__ == "__main__":
    build()
