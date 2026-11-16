#!/usr/bin/env python
"""How much of "identify the perturbation" is just reading the knocked-down gene?

CDN (Wu et al., ICML 2025) reports that on five Perturb-seq datasets every
published deep-learning baseline — GEARS, PDGrapher, Linear, MLP, GenePt — sits at
0.31–0.57 normalized rank (0.5 = chance), while a plain Wilcoxon rank-sum test
reaches 0.79–0.89. Their explanation: for most perturbations the intended target
IS simply the gene with the largest log-fold change, because CRISPR knockdown
lowers the target's own transcript. They therefore report their headline table on
the "non-trivial" subset only.

If that holds here, then high retrieval accuracy on perturb-seq is mostly
barcode-reading, not inverse design — and it would NOT transfer to a design task,
where the target state is a desired phenotype that carries no such fingerprint.

This measures it directly from the signatures already on disk:
  for each condition, where does the perturbed gene's OWN Δ rank among all genes,
  sorted most-down-regulated first?

  trivial  := perturbed gene is rank 1 (the single most down-regulated gene)
  top10    := within the 10 most down-regulated
  invisible:= the perturbed gene is not in the measured gene set at all

CPU only; reads artifacts/npz/*_signatures.npz, writes artifacts/trivial_rule.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    args = ap.parse_args()

    npzdir = os.path.join(args.artifacts, "npz")
    # "*_oracle_signatures.npz" (written by inverse_gap.py) also ends in
    # _signatures.npz but holds oracle_sig/empirical_sig instead of signatures.
    files = sorted(
        f for f in os.listdir(npzdir)
        if f.endswith("_signatures.npz") and not f.endswith("_oracle_signatures.npz")
    )
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": ("is the perturbed gene itself the most down-regulated gene in its "
                     "own condition? (i.e. is identification just barcode-reading)"),
        "reference": ("CDN / Wu et al. ICML 2025 call these 'trivial' perturbations and "
                      "exclude them from their headline table"),
        "datasets": [],
    }

    for f in files:
        ds = f.replace("_signatures.npz", "")
        z = np.load(os.path.join(npzdir, f), allow_pickle=False)
        missing = [k for k in ("signatures", "conditions", "gene_names") if k not in z]
        if missing:
            log(f"{ds}: skipping, archive lacks {missing}")
            continue
        sig = z["signatures"].astype(np.float64)       # (K, G) Δ-from-control
        conds = [str(c) for c in z["conditions"]]
        genes = [str(g) for g in z["gene_names"]]
        gidx = {g: i for i, g in enumerate(genes)}

        # Direction matters: CRISPRi/knockdown lowers the target's transcript, but
        # Norman 2019 is a CRISPRa (activation) screen, which RAISES it. Ranking only
        # by "most down-regulated" therefore inverts the conclusion for Norman
        # (its targets land at the very bottom of a down-ranking, i.e. the top of an
        # up-ranking). So rank by |Δ| -- "is the perturbed gene among the most-changed
        # genes", which is direction-agnostic -- and keep the signed ranks alongside.
        rows = np.arange(sig.shape[0])[:, None]

        def _ranks(mat):
            o = np.argsort(mat, axis=1)
            r = np.empty_like(o)
            r[rows, o] = np.arange(mat.shape[1])[None, :] + 1
            return r

        rank_abs = _ranks(-np.abs(sig))   # rank 1 = largest |Δ|
        rank_down = _ranks(sig)           # rank 1 = most down-regulated
        rank_up = _ranks(-sig)            # rank 1 = most up-regulated
        rank_of = rank_abs

        recs, invisible = [], 0
        for i, c in enumerate(conds):
            targets = [p for p in c.split("+") if p != "ctrl"]
            present = [t for t in targets if t in gidx]
            if not present:
                invisible += 1
                continue
            # for a combo, take the best-ranked of its genes
            # for a combo, take the best-ranked of its genes
            best = min(present, key=lambda t: int(rank_abs[i, gidx[t]]))
            j = gidx[best]
            recs.append({"condition": c,
                         "own_gene_abs_rank": int(rank_abs[i, j]),
                         "own_gene_down_rank": int(rank_down[i, j]),
                         "own_gene_up_rank": int(rank_up[i, j]),
                         "own_gene_delta": float(sig[i, j]),
                         "n_targets_measured": len(present),
                         "n_targets": len(targets)})

        if not recs:
            out["datasets"].append({"dataset": ds, "n_conditions": len(conds),
                                    "all_targets_invisible": True})
            continue

        rk = np.array([r["own_gene_abs_rank"] for r in recs])
        dz = np.array([r["own_gene_delta"] for r in recs])
        G = len(genes)
        frac_down = float(np.mean(dz < 0))
        d = {
            "dataset": ds,
            "n_conditions": len(conds),
            "n_conditions_target_measured": len(recs),
            "n_conditions_target_invisible": invisible,
            "frac_target_invisible": invisible / max(1, len(conds)),
            "n_genes": G,
            # the trivial rule, direction-agnostic: rank by |Δ|
            "frac_trivial_rank1": float(np.mean(rk == 1)),
            "frac_top10": float(np.mean(rk <= 10)),
            "frac_top100": float(np.mean(rk <= 100)),
            "median_own_gene_abs_rank": float(np.median(rk)),
            # which direction is the fingerprint? CRISPRi/KO -> down, CRISPRa -> up
            "frac_own_gene_goes_down": frac_down,
            "direction": ("down (knockdown/knockout)" if frac_down > 0.6
                          else "up (activation)" if frac_down < 0.4
                          else "mixed"),
            "median_own_gene_delta": float(np.median(dz)),
            "median_own_gene_down_rank": float(
                np.median([r["own_gene_down_rank"] for r in recs])),
            "median_own_gene_up_rank": float(
                np.median([r["own_gene_up_rank"] for r in recs])),
            "chance_frac_top10": 10.0 / G,
        }
        out["datasets"].append(d)
        log(f"{ds}: rank1={d['frac_trivial_rank1']:.3f} "
            f"top10={d['frac_top10']:.3f} "
            f"top100={d['frac_top100']:.3f} "
            f"median|Δ|rank={d['median_own_gene_abs_rank']:.0f}/{G} "
            f"invisible={d['frac_target_invisible']:.3f} "
            f"dir={d['direction']}")

    p = os.path.join(args.artifacts, "trivial_rule.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    log("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
