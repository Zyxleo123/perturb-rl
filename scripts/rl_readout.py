#!/usr/bin/env python
"""Recover the readout gene names behind each RL cache's signature columns.

rl_cache.py stores Y = dF[:, mask] (K x G_masked) but not which genes the columns
are. Putting one line's signature next to the other's needs them, so this re-derives
them exactly as rl_headroom.build does -- adata.var["gene_name"] if present, else
var_names; mask = readout genes that are not any candidate's target -- and proves the
mapping against the cache before writing it:
  1. the masked count equals the cache's column count;
  2. dF for a few candidates, recomputed from the raw h5ad (mean of the gene's cells
     minus the mean of all control cells -- the full-cell signature does not depend on
     build's random half split), equals the cache's row to float32 precision.
A gene_name / var_names (Ensembl) mix-up fails check 2 loudly (see GAPS.md's
evaluation traps).

Writes <out>/rl_acquire/readout_<dataset>.npz (genes = the cache's column names, in
order) and readout_check.json. Usage:
  python -I rl_readout.py --gears-dir D --cache-dir D --out-dir D
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own scripts/ only
import numpy as np  # noqa: E402

import rl_headroom as RH  # noqa: E402

N_CHECK = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    import anndata as ad
    cc = json.load(open(os.path.join(a.cache_dir, "cache_check.json")))
    od = os.path.join(a.out_dir, "rl_acquire")
    check = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "datasets": {}}
    ok_all = True
    for name, info in cc["datasets"].items():
        z = np.load(info["path"], allow_pickle=False)
        Y, cands = z["Y"], [str(c) for c in z["cands"]]
        adata = ad.read_h5ad(os.path.join(a.gears_dir, name, "perturb_processed.h5ad"), backed="r")
        genes = np.asarray(adata.var["gene_name"].astype(str).values if "gene_name" in adata.var
                           else adata.var_names.astype(str), dtype=object)
        mask = np.array([g not in set(cands) for g in genes])
        conds = np.asarray([RH.canon(c) for c in adata.obs["condition"].astype(str).values], dtype=object)
        rows_ctrl = np.flatnonzero(conds == "ctrl")
        X = adata.X
        mu = np.asarray(RH.densify(X[np.sort(rows_ctrl)]), np.float64).mean(0)
        rng = np.random.default_rng(0)
        diffs = {}
        for i in rng.choice(len(cands), size=N_CHECK, replace=False):
            r = np.sort(np.flatnonzero(conds == cands[i]))
            d = (np.asarray(RH.densify(X[r]), np.float64).mean(0) - mu)[mask]
            diffs[cands[i]] = float(np.abs(d - Y[i]).max() / max(np.abs(Y[i]).max(), 1e-12))
        count_ok = int(mask.sum()) == Y.shape[1]
        rows_ok = all(v < 1e-4 for v in diffs.values())
        ok = count_ok and rows_ok
        ok_all &= ok
        check["datasets"][name] = {"ok": bool(ok), "n_genes": int(genes.size), "n_masked": int(mask.sum()),
                                   "cache_cols": int(Y.shape[1]), "count_ok": bool(count_ok),
                                   "row_max_rel_diff": diffs, "rows_ok": bool(rows_ok),
                                   "used_gene_name": bool("gene_name" in adata.var)}
        RH.log(f"{name}: {json.dumps(check['datasets'][name])}")
        if ok:
            np.savez(os.path.join(od, f"readout_{name}.npz"), genes=np.array(genes[mask], dtype=str))
        adata.file.close()
    a_sets = [set(np.load(os.path.join(od, f"readout_{n}.npz"))["genes"].tolist())
              for n in cc["datasets"] if check["datasets"][n]["ok"]]
    if len(a_sets) == 2:
        check["shared_readout_genes"] = len(a_sets[0] & a_sets[1])
    check["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(od, "readout_check.json"), "w") as f:
        json.dump(check, f, indent=2)
    RH.log("readout check:", "OK" if ok_all else "FAILED", "shared:", check.get("shared_readout_genes"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
