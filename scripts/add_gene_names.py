#!/usr/bin/env python
"""Add gene_names to an existing *_oracle_signatures.npz.

inverse_gap.py now stores them, but the archive on disk predates that. Re-running
the GPU job just to recover column labels is waste: the names live in the
dataloader's h5ad and can be read on CPU without loading X or the model.

Reads var in backed mode so the 3.2 GB X is never touched.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--h5ad", required=True)
    args = ap.parse_args()

    import anndata as ad
    a = ad.read_h5ad(args.h5ad, backed="r")
    var = a.var
    names = (var["gene_name"].astype(str).values if "gene_name" in var
             else np.asarray(a.var_names, dtype=str))
    names = np.asarray(names, dtype=str)
    print(f"h5ad var: {len(names)} genes, first 3: {list(names[:3])}")

    # allow_pickle=True only because this archive is one OUR OWN job wrote minutes
    # ago, and an older inverse_gap.py stored gene_names as a pandas-Categorical
    # object array. Everything is re-saved below with explicit string dtypes so
    # later readers never need pickle.
    z = dict(np.load(args.npz, allow_pickle=True))
    for k, v in list(z.items()):
        if v.dtype == object:
            z[k] = np.asarray(v, dtype=str)
            print(f"coerced {k} from object to {z[k].dtype}")
    ncol = z["empirical_sig"].shape[1]
    if len(names) != ncol:
        print(f"REFUSING: h5ad has {len(names)} genes but npz has {ncol} columns. "
              "These are different gene spaces; mislabelling would be worse than no labels.")
        return 1
    z["gene_names"] = np.asarray(names, dtype=str)
    np.savez_compressed(args.npz, **z)
    print(f"added gene_names ({len(names)}) to {args.npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
