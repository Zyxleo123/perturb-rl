#!/usr/bin/env python
"""Profile the downloaded perturb-seq datasets and measure how well-posed the
INVERSE problem (target cell state -> perturbation) actually is.

Everything is written to disk as JSON/JSONL/NPZ so that figures can be redrawn
without recomputation (see jobs/30_figures.sbatch).

Run with `python -I` and pass paths as arguments; never cd into the data dir.

Usage:
  python -I profile_datasets.py --gears-dir DIR --out-dir DIR [--pdgrapher-dir DIR]
                               [--datasets norman adamson ...] [--max-cells N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

# ---------------------------------------------------------------- config
# Pseudo-bulk sizes for the identifiability sweep: how many cells of the SAME
# unknown perturbation must you observe before you can name it?
BULK_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
TOPKS = (1, 5, 10, 20)
RNG_SEED = 0
# A perturbation needs at least this many cells to take part in the retrieval
# experiment (half for the centroid, half for queries).
MIN_CELLS_PER_COND = 20


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def densify(x):
    import scipy.sparse as sp
    return np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)


def canon_condition(c):
    """Canonical name for a perturbation: sorted gene set, 'ctrl' padding dropped.

    Norman encodes the SAME single-gene perturbation both ways -- 'SPI1+ctrl' and
    'ctrl+SPI1' are 46 such pairs out of 276. Treating them as distinct candidates
    inflates K and scores a correct answer as wrong whenever the model picks the
    twin, so every grouping here goes through this first.
    """
    parts = sorted(p for p in str(c).split("+") if p != "ctrl")
    return "+".join(parts) if parts else "ctrl"


# ---------------------------------------------------------------- basic profile
def basic_profile(adata, name):
    """Shape / action-space / annotation inventory for one dataset."""
    import pandas as pd

    obs = adata.obs
    cond_col = "condition" if "condition" in obs else None
    if cond_col is None:
        raise KeyError(f"{name}: no 'condition' column; obs has {list(obs.columns)}")

    raw_conds = obs[cond_col].astype(str)
    n_raw = int(raw_conds.nunique())
    conds = raw_conds.map(canon_condition)
    vc = conds.value_counts()
    ctrl_mask = (conds == "ctrl").values
    pert_conds = [c for c in vc.index if c != "ctrl"]
    n_dup_encodings = n_raw - int(conds.nunique())

    # GEARS encodes conditions as "GENE+ctrl" (single) or "GENEA+GENEB" (combo).
    def parse(c):
        parts = [p for p in c.split("+") if p != "ctrl"]
        return parts

    singles = [c for c in pert_conds if len(parse(c)) == 1]
    combos = [c for c in pert_conds if len(parse(c)) >= 2]
    pert_genes = sorted({g for c in pert_conds for g in parse(c)})

    measured = (
        adata.var["gene_name"].astype(str).tolist()
        if "gene_name" in adata.var
        else adata.var_names.astype(str).tolist()
    )
    measured_set = set(measured)
    # Cycle consistency needs the oracle to be able to SEE the gene it perturbed.
    pert_genes_measured = [g for g in pert_genes if g in measured_set]

    X = adata.X
    try:
        import scipy.sparse as sp
        nnz_frac = (X.nnz / (X.shape[0] * X.shape[1])) if sp.issparse(X) else float(np.mean(np.asarray(X) != 0))
    except Exception:
        nnz_frac = float("nan")

    sub = X[:200]
    sub = densify(sub)
    prof = {
        "dataset": name,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_control_cells": int(ctrl_mask.sum()),
        "n_conditions_total": int(len(vc)),
        "n_perturbed_conditions": int(len(pert_conds)),
        "n_conditions_raw_encodings": n_raw,
        "n_duplicate_encodings_merged": n_dup_encodings,
        "n_single_conditions": int(len(singles)),
        "n_combo_conditions": int(len(combos)),
        "n_unique_perturbed_genes": int(len(pert_genes)),
        "n_perturbed_genes_also_measured": int(len(pert_genes_measured)),
        "frac_perturbed_genes_measured": (
            float(len(pert_genes_measured) / len(pert_genes)) if pert_genes else float("nan")
        ),
        "cells_per_condition_median": float(vc[pert_conds].median()) if pert_conds else float("nan"),
        "cells_per_condition_min": int(vc[pert_conds].min()) if pert_conds else -1,
        "cells_per_condition_max": int(vc[pert_conds].max()) if pert_conds else -1,
        "n_conditions_ge_min_cells": int((vc[pert_conds] >= MIN_CELLS_PER_COND).sum()) if pert_conds else 0,
        "X_nonzero_frac": float(nnz_frac),
        "X_min": float(np.min(sub)),
        "X_max": float(np.max(sub)),
        "X_looks_log1p": bool(np.max(sub) < 20.0),
        "obs_columns": list(map(str, obs.columns)),
        "var_columns": list(map(str, adata.var.columns)),
        "cells_per_condition_hist": np.histogram(
            vc[pert_conds].values if pert_conds else [0], bins=30
        )[0].tolist(),
    }
    return prof, conds.values, ctrl_mask, pert_conds, np.array(measured, dtype=object)


# ---------------------------------------------------------------- signatures
def condition_signatures(X, conds, ctrl_mask, pert_conds, rng):
    """Split each condition's cells in half: one half builds the centroid
    signature, the other half supplies held-out queries.

    Returns delta-from-control representations throughout, because that is the
    quantity an inverse model is actually handed as its target."""
    ctrl_mean = densify(X[ctrl_mask]).mean(axis=0)

    kept, sig, idx_query = [], [], {}
    for c in pert_conds:
        rows = np.flatnonzero(conds == c)
        if rows.size < MIN_CELLS_PER_COND:
            continue
        rng.shuffle(rows)
        half = rows.size // 2
        ref, qry = rows[:half], rows[half:]
        sig.append(densify(X[ref]).mean(axis=0) - ctrl_mean)
        idx_query[c] = qry
        kept.append(c)
    return np.asarray(kept, dtype=object), np.vstack(sig).astype(np.float32), idx_query, ctrl_mean


def de_and_effect_stats(X, conds, ctrl_mask, kept, ctrl_mean, sig):
    """Per-condition effect size and a DE-gene count, plus the within-condition
    spread that sets the noise floor for identifiability."""
    ctrl = densify(X[ctrl_mask])
    ctrl_sd = ctrl.std(axis=0) + 1e-8
    rows = []
    for i, c in enumerate(kept):
        cells = densify(X[conds == c])
        d = sig[i]
        # z-like score per gene against control variability (not a formal test;
        # a cheap, consistent effect-size proxy across datasets)
        z = np.abs(d) / ctrl_sd
        # within-condition spread of single-cell deltas around their own mean
        within = float(np.linalg.norm(cells - ctrl_mean - d, axis=1).mean())
        rows.append(
            {
                "condition": str(c),
                "n_cells": int(cells.shape[0]),
                "l2_from_control": float(np.linalg.norm(d)),
                "n_de_z2": int((z > 2).sum()),
                "n_de_z5": int((z > 5).sum()),
                "frac_de_z2": float((z > 2).mean()),
                "max_abs_delta": float(np.abs(d).max()),
                "within_cond_l2_spread": within,
                # >1 means the perturbation signal stands out above cell-to-cell noise
                "effect_to_noise": float(np.linalg.norm(d) / (within + 1e-8)),
                "n_genes_in_pert": int(len([p for p in str(c).split("+") if p != "ctrl"])),
            }
        )
    return rows


def nearest_neighbour_confusability(kept, sig):
    """How close is each perturbation's signature to the NEXT perturbation's?
    If the nearest neighbour is closer than the within-condition spread, a target
    state cannot identify a unique perturbation -- the inverse map is one-to-many
    and both regression and cycle-consistency objectives become degenerate."""
    S = sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-8)
    C = S @ S.T
    np.fill_diagonal(C, -np.inf)
    nn = np.argmax(C, axis=1)
    return [
        {"condition": str(kept[i]), "nn_condition": str(kept[nn[i]]), "nn_cosine": float(C[i, nn[i]])}
        for i in range(len(kept))
    ]


# ---------------------------------------------------------------- retrieval
def retrieval_sweep(X, kept, sig, idx_query, ctrl_mean, rng, n_draws=200):
    """Nearest-centroid inverse model: given the observed state of n cells that
    share one unknown perturbation, name the perturbation.

    This is the free, no-training lower bound every learned inverse model must
    beat, and the cleanest measure of how much information a target cell state
    carries about its cause."""
    S = sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-8)
    K = len(kept)
    cond_to_i = {c: i for i, c in enumerate(kept)}
    out = []
    for n in BULK_SIZES:
        hits = {k: 0 for k in TOPKS}
        total = 0
        ranks = []
        for c in kept:
            qry = idx_query[c]
            if qry.size < n:
                continue
            for _ in range(max(1, n_draws // max(1, K))):
                pick = rng.choice(qry, size=n, replace=False)
                q = densify(X[pick]).mean(axis=0) - ctrl_mean
                q = q / (np.linalg.norm(q) + 1e-8)
                scores = S @ q
                order = np.argsort(-scores)
                r = int(np.flatnonzero(order == cond_to_i[c])[0]) + 1
                ranks.append(r)
                for k in TOPKS:
                    if r <= k:
                        hits[k] += 1
                total += 1
        if total == 0:
            continue
        out.append(
            {
                "n_cells_observed": int(n),
                "n_queries": int(total),
                "n_candidate_conditions": int(K),
                "chance_top1": float(1.0 / K),
                **{f"top{k}_acc": hits[k] / total for k in TOPKS},
                "median_rank": float(np.median(ranks)),
                "mean_reciprocal_rank": float(np.mean(1.0 / np.asarray(ranks))),
            }
        )
    return out


# ---------------------------------------------------------------- pdgrapher
def profile_pdgrapher(pdg_dir):
    """Inventory PDGrapher's processed data: which cell lines, how many paired
    samples, and what its splits actually contain."""
    info = {"root": pdg_dir, "exists": os.path.isdir(pdg_dir)}
    if not info["exists"]:
        return info
    for sub in ("splits", "torch_data"):
        p = os.path.join(pdg_dir, "processed", sub)
        entry = {"path": p, "exists": os.path.isdir(p)}
        if entry["exists"]:
            try:
                names = sorted(os.listdir(p))
                entry["n_entries"] = len(names)
                entry["entries_sample"] = names[:40]
                # walk one level for cell-line style dirs
                deep = {}
                for n in names[:40]:
                    q = os.path.join(p, n)
                    if os.path.isdir(q):
                        try:
                            deep[n] = sorted(os.listdir(q))[:12]
                        except OSError:
                            pass
                entry["children"] = deep
            except OSError as e:
                entry["error"] = str(e)
        info[sub] = entry
    return info


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pdgrapher-dir", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-cells", type=int, default=250_000)
    args = ap.parse_args()

    import anndata as ad

    os.makedirs(args.out_dir, exist_ok=True)
    npz_dir = os.path.join(args.out_dir, "npz")
    os.makedirs(npz_dir, exist_ok=True)

    names = args.datasets or [
        d
        for d in sorted(os.listdir(args.gears_dir))
        if os.path.isfile(os.path.join(args.gears_dir, d, "perturb_processed.h5ad"))
    ]
    log("datasets found:", names)

    profiles, cond_rows, retr_rows, errors = [], [], [], []
    gene_sets = {}

    for name in names:
        h5 = os.path.join(args.gears_dir, name, "perturb_processed.h5ad")
        log("=" * 70)
        log("loading", name, "from", h5)
        try:
            adata = ad.read_h5ad(h5)
            log(f"  {name}: {adata.shape}")
            if adata.n_obs > args.max_cells:
                rng0 = np.random.default_rng(RNG_SEED)
                keep = rng0.choice(adata.n_obs, size=args.max_cells, replace=False)
                keep.sort()
                adata = adata[keep].copy()
                log(f"  subsampled to {adata.shape}")

            prof, conds, ctrl_mask, pert_conds, measured = basic_profile(adata, name)
            gene_sets[name] = measured.tolist()
            log(
                f"  {prof['n_perturbed_conditions']} perturbed conditions, "
                f"{prof['n_unique_perturbed_genes']} unique perturbed genes, "
                f"{prof['n_control_cells']} controls"
            )

            if prof["n_control_cells"] < 10:
                prof["skip_reason"] = "fewer than 10 control cells"
                profiles.append(prof)
                errors.append({"dataset": name, "error": "no controls"})
                del adata
                continue

            rng = np.random.default_rng(RNG_SEED)
            X = adata.X
            kept, sig, idx_query, ctrl_mean = condition_signatures(
                X, conds, ctrl_mask, pert_conds, rng
            )
            log(f"  {len(kept)} conditions with >= {MIN_CELLS_PER_COND} cells enter retrieval")
            prof["n_conditions_in_retrieval"] = int(len(kept))

            eff = de_and_effect_stats(X, conds, ctrl_mask, kept, ctrl_mean, sig)
            nnc = {d["condition"]: d for d in nearest_neighbour_confusability(kept, sig)}
            for row in eff:
                row.update({k: v for k, v in nnc[row["condition"]].items() if k != "condition"})
                row["dataset"] = name
                cond_rows.append(row)

            # headline aggregate gap numbers
            etn = np.array([r["effect_to_noise"] for r in eff])
            prof["effect_to_noise_median"] = float(np.median(etn))
            prof["frac_conditions_etn_below_1"] = float(np.mean(etn < 1.0))
            prof["nn_cosine_median"] = float(np.median([r["nn_cosine"] for r in nnc.values()]))
            prof["n_de_z2_median"] = float(np.median([r["n_de_z2"] for r in eff]))
            prof["frac_de_z2_median"] = float(np.median([r["frac_de_z2"] for r in eff]))

            rs = retrieval_sweep(X, kept, sig, idx_query, ctrl_mean, rng)
            for r in rs:
                r["dataset"] = name
                retr_rows.append(r)
            if rs:
                prof["top1_acc_1cell"] = rs[0]["top1_acc"]
                prof["chance_top1"] = rs[0]["chance_top1"]
                prof["top1_acc_best_bulk"] = max(r["top1_acc"] for r in rs)
                log(
                    f"  retrieval: 1-cell top1={rs[0]['top1_acc']:.3f} "
                    f"(chance {rs[0]['chance_top1']:.4f}), "
                    f"best bulk top1={prof['top1_acc_best_bulk']:.3f}"
                )

            np.savez_compressed(
                os.path.join(npz_dir, f"{name}_signatures.npz"),
                conditions=np.asarray([str(c) for c in kept]),
                signatures=sig,
                ctrl_mean=ctrl_mean.astype(np.float32),
                gene_names=measured.astype(str),
            )
            profiles.append(prof)
            del adata, X, sig
        except Exception as e:
            log(f"  ERROR on {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            errors.append({"dataset": name, "error": f"{type(e).__name__}: {e}"})

    # -------------------------------------------------- cross-dataset overlap
    overlap = []
    ns = list(gene_sets)
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            a, b = set(gene_sets[ns[i]]), set(gene_sets[ns[j]])
            overlap.append(
                {
                    "a": ns[i], "b": ns[j],
                    "n_a": len(a), "n_b": len(b),
                    "n_shared": len(a & b),
                    "jaccard": len(a & b) / max(1, len(a | b)),
                }
            )

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gears_dir": args.gears_dir,
        "min_cells_per_cond": MIN_CELLS_PER_COND,
        "bulk_sizes": list(BULK_SIZES),
        "profiles": profiles,
        "gene_overlap": overlap,
        "errors": errors,
    }
    if args.pdgrapher_dir:
        out["pdgrapher"] = profile_pdgrapher(args.pdgrapher_dir)

    with open(os.path.join(args.out_dir, "dataset_profiles.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.out_dir, "condition_stats.jsonl"), "w") as f:
        for r in cond_rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, "retrieval.jsonl"), "w") as f:
        for r in retr_rows:
            f.write(json.dumps(r) + "\n")

    log("wrote dataset_profiles.json, condition_stats.jsonl, retrieval.jsonl to", args.out_dir)
    log("errors:", errors if errors else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
