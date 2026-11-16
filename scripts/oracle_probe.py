#!/usr/bin/env python
"""Does the GEARS forward oracle actually carry enough signal to be an RL reward?

The project plan uses GEARS as (a) the reward oracle and (b) the cycle-consistency
map. Both only make sense if GEARS beats the trivial predictors that ignore which
perturbation was applied. This script measures that on the official Norman
checkpoint, against two baselines:

  * "control mean"        -- predict no change at all
  * "mean perturbed cell" -- predict the average perturbation effect, ignoring
                             the identity of the perturbation (non-conditional)

Metric is GEARS's own headline: Pearson r over the top-20 non-dropout DE genes,
both on raw expression and on the delta from control. The delta version is the
one that matters: raw r is inflated to ~0.99 by the shared mean expression
profile, which is identical for every predictor.

IMPORTANT CAVEAT, reported in the output: the official checkpoint is
`gears_misc_umi_no_test`, trained with split='no_test', i.e. on every
perturbation in Norman. So these numbers are TRAIN-SET numbers and are an
optimistic upper bound on oracle quality. Honest held-out numbers need a
retrain with split='simulation' (jobs/22_retrain_oracle.sbatch).

Writes <out>/oracle_probe.json. Always exits 0 so the figure job still runs.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback

import numpy as np


def log(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.strftime('%H:%M:%S')} rss={rss:.1f}GB]", *a, flush=True)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 3:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def densify(x):
    import scipy.sparse as sp
    return np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataloader-dir", required=True, help="norman_umi_go directory")
    ap.add_argument("--ckpt-dir", required=True, help="model_ckpt directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", required=True, help="scratch dir PertData may write into")
    ap.add_argument("--split", default="no_test")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-conditions", type=int, default=0, help="0 = all")
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--out-name", default="oracle_probe.json")
    ap.add_argument(
        "--skip-cell-graphs",
        action="store_true",
        help="skip unpickling data_pyg/cell_graphs.pkl (3.7 GB, >30 min over a contended "
             "NFS mount). predict() never reads it; set_pert_genes() and prepare_split() "
             "still run on the real PertData so the perturbation indexing is GEARS's own.",
    )
    ap.add_argument(
        "--held-out-only",
        action="store_true",
        help="restrict to the split's test conditions -- the honest setting, and the "
             "only one that says anything about perturbations the oracle has not seen",
    )
    ap.add_argument(
        "--load-dataloader",
        action="store_true",
        help="also load the 3.7 GB cell-graph pickle (not needed for predict())",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": args.ckpt_dir,
        "dataloader": args.dataloader_dir,
        "split": args.split,
        "metric": "pearson r over top-20 non-dropout DE genes",
        "caveat": (
            "Official checkpoint is gears_misc_umi_no_test (split='no_test'), trained on ALL "
            "Norman perturbations. These are train-set numbers: an OPTIMISTIC upper bound on "
            "reward-oracle quality, not held-out performance."
        ),
        "per_condition": [],
        "baselines": {},
        "status": "started",
    }
    out_path = os.path.join(args.out_dir, args.out_name)
    out["held_out_only"] = bool(args.held_out_only)
    if args.held_out_only:
        out["caveat"] = (
            "Restricted to the split's TEST conditions: perturbations the oracle never saw "
            "in training. This is the honest estimate of reward-oracle quality in the regime "
            "RL post-training will actually explore."
        )

    def dump():
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    try:
        import torch
        from gears import GEARS, PertData

        log("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            log("gpu:", torch.cuda.get_device_name(0),
                f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        else:
            args.device = "cpu"
            log("no CUDA -> falling back to CPU (slower but fine for inference)")
        out["device"] = args.device

        if args.skip_cell_graphs:
            # PertData.load() unpickles data_pyg/cell_graphs.pkl unconditionally
            # (pertdata.py ~l.200); it is 3.7 GB and predict() never reads it.
            # Intercept that one filename only -- prepare_split needs its own pickles.
            import pickle as _pickle

            _real_load = _pickle.load

            def _skip_cell_graphs(f, *a, **k):
                if str(getattr(f, "name", "")).endswith("cell_graphs.pkl"):
                    log("skipping cell_graphs.pkl (unused by predict())")
                    return None
                return _real_load(f, *a, **k)

            _pickle.load = _skip_cell_graphs

        log("loading PertData from", args.dataloader_dir)
        pert_data = PertData(args.work_dir)
        pert_data.load(data_path=args.dataloader_dir)
        log("adata:", pert_data.adata.shape)

        log("prepare_split", args.split, "seed", args.seed)
        pert_data.prepare_split(split=args.split, seed=args.seed)

        if args.skip_cell_graphs:
            _pickle.load = _real_load
            # Normally set by get_dataloader(); copied verbatim from GEARS
            # pertdata.py l.429-430 so the gene ordering is identical.
            pert_data.node_map = {
                x: it for it, x in enumerate(pert_data.adata.var.gene_name)
            }
            pert_data.gene_names = pert_data.adata.var.gene_name
        out["skip_cell_graphs"] = bool(args.skip_cell_graphs)

        if args.load_dataloader:
            log("get_dataloader (loads the big cell-graph pickle)")
            pert_data.get_dataloader(batch_size=32, test_batch_size=128)
        else:
            # GEARS.__init__ only reads this attribute and predict() builds its own
            # cell graphs from ctrl_adata, so we can skip building the loaders.
            # NOTE: this does NOT avoid the big cell-graph pickle -- PertData.load()
            # unpickles data_pyg/cell_graphs.pkl unconditionally (pertdata.py ~l.200),
            # which for norman_umi_go is 3.7 GB and dominates startup. Skipping
            # get_dataloader only avoids the DataLoader/batching pass on top of it.
            pert_data.dataloader = None
            log("skipped get_dataloader (predict() does not use it)")

        log("constructing GEARS and loading pretrained weights")
        model = GEARS(pert_data, device=args.device, weight_bias_track=False,
                      proj_name="gears", exp_name="oracle_probe")
        model.load_pretrained(args.ckpt_dir)
        out["model_config"] = {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in (model.config or {}).items()
        }
        log("model loaded. config:", out["model_config"])

        adata = model.adata
        genes = np.asarray(adata.var["gene_name"].astype(str).values)
        # uns['top_non_dropout_de_20'] stores ENSEMBL IDs (ENSG...) in the
        # norman_umi_go release, but gene NAMES in some other GEARS releases.
        # Index both so either resolves; mismatching them silently empties the DE
        # set and the whole probe reports nothing.
        gene_to_i = {g: i for i, g in enumerate(genes)}
        if "gene_id" in adata.var:
            for i, gid in enumerate(adata.var["gene_id"].astype(str).values):
                gene_to_i.setdefault(gid, i)
        for i, vn in enumerate(adata.var_names.astype(str)):
            gene_to_i.setdefault(vn, i)
        conds = adata.obs["condition"].astype(str).values
        cond_names = adata.obs["condition_name"].astype(str).values
        ctrl_mask = conds == "ctrl"

        X = adata.X
        ctrl_mean = densify(X[ctrl_mask]).mean(axis=0)
        pert_mean = densify(X[~ctrl_mask]).mean(axis=0)  # non-conditional baseline
        log(f"ctrl cells={ctrl_mask.sum()} pert cells={(~ctrl_mask).sum()} genes={genes.size}")

        de20 = adata.uns["top_non_dropout_de_20"]
        # map condition -> its condition_name key used in uns
        cond_to_name = {}
        for c, cn in zip(conds, cond_names):
            if c != "ctrl":
                cond_to_name.setdefault(c, cn)

        import pandas as pd
        vc = pd.Series(conds).value_counts()
        todo = [c for c in vc.index if c != "ctrl" and vc[c] >= args.min_cells]
        todo = [c for c in todo if cond_to_name.get(c) in de20]
        # every gene in the perturbation must be in GEARS's perturbation graph
        def parse(c):
            return [p for p in c.split("+") if p != "ctrl"]
        skipped_not_in_graph = [c for c in todo if any(g not in model.pert_list for g in parse(c))]
        todo = [c for c in todo if all(g in model.pert_list for g in parse(c))]

        if args.held_out_only:
            s2c = model.set2conditions or {}
            test_conds = set()
            for key in ("test", "val"):
                if key in s2c:
                    test_conds |= {str(c) for c in s2c[key]}
                    out[f"n_{key}_conditions_in_split"] = len(s2c[key])
            if not test_conds:
                log("WARNING: split has no test/val conditions; "
                    "falling back to all conditions (numbers are NOT held out)")
                out["held_out_only"] = False
                out["held_out_warning"] = "split exposed no test conditions"
            else:
                before = len(todo)
                todo = [c for c in todo if c in test_conds]
                log(f"held-out filter: {before} -> {len(todo)} conditions")

        if args.max_conditions:
            todo = todo[: args.max_conditions]
        log(f"evaluating {len(todo)} conditions "
            f"({len(skipped_not_in_graph)} skipped: gene absent from pert graph)")
        out["n_conditions_evaluated"] = len(todo)
        out["n_skipped_not_in_pert_graph"] = len(skipped_not_in_graph)
        out["skipped_examples"] = skipped_not_in_graph[:20]

        rows = []
        n_de_unresolved = 0
        for n, c in enumerate(todo):
            try:
                de_genes = [str(g) for g in de20[cond_to_name[c]]]
                de_idx = np.array([gene_to_i[g] for g in de_genes if g in gene_to_i])
                if de_idx.size < 5:
                    n_de_unresolved += 1
                    if n_de_unresolved <= 3:
                        log(f"  {c}: only {de_idx.size}/{len(de_genes)} DE genes resolved "
                            f"(e.g. {de_genes[:3]}) -- skipping")
                    continue
                truth = densify(X[conds == c]).mean(axis=0)
                pl = parse(c)
                pred = model.predict([pl])["_".join(pl)]
                pred = np.asarray(pred).ravel()

                t, p = truth[de_idx], pred[de_idx]
                cm, pm = ctrl_mean[de_idx], pert_mean[de_idx]
                rows.append(
                    {
                        "condition": c,
                        "n_cells": int((conds == c).sum()),
                        "n_genes_in_pert": len(pl),
                        "n_de": int(de_idx.size),
                        # raw expression (inflated by the shared mean profile)
                        "gears_pearson_de": pearson(p, t),
                        "ctrl_pearson_de": pearson(cm, t),
                        "meanpert_pearson_de": pearson(pm, t),
                        # delta from control: the metric that actually separates models
                        "gears_pearson_delta_de": pearson(p - cm, t - cm),
                        "meanpert_pearson_delta_de": pearson(pm - cm, t - cm),
                        "gears_mse_de": float(np.mean((p - t) ** 2)),
                        "ctrl_mse_de": float(np.mean((cm - t) ** 2)),
                        "meanpert_mse_de": float(np.mean((pm - t) ** 2)),
                        "truth_effect_l2_de": float(np.linalg.norm(t - cm)),
                    }
                )
                if n % 25 == 0:
                    log(f"  [{n+1}/{len(todo)}] {c}: "
                        f"gears_delta={rows[-1]['gears_pearson_delta_de']:.3f} "
                        f"meanpert_delta={rows[-1]['meanpert_pearson_delta_de']:.3f}")
            except Exception as e:
                log(f"  WARN {c}: {type(e).__name__}: {e}")

        out["per_condition"] = rows
        out["n_skipped_de_unresolved"] = n_de_unresolved
        if rows:
            def med(k):
                v = np.array([r[k] for r in rows if np.isfinite(r.get(k, np.nan))])
                return float(np.median(v)) if v.size else float("nan")

            out["baselines"] = {
                "median_gears_pearson_de": med("gears_pearson_de"),
                "median_ctrl_pearson_de": med("ctrl_pearson_de"),
                "median_meanpert_pearson_de": med("meanpert_pearson_de"),
                "median_gears_pearson_delta_de": med("gears_pearson_delta_de"),
                "median_meanpert_pearson_delta_de": med("meanpert_pearson_delta_de"),
                "median_gears_mse_de": med("gears_mse_de"),
                "median_ctrl_mse_de": med("ctrl_mse_de"),
                "median_meanpert_mse_de": med("meanpert_mse_de"),
            }
            g = np.array([r["gears_pearson_delta_de"] for r in rows], dtype=float)
            m = np.array([r["meanpert_pearson_delta_de"] for r in rows], dtype=float)
            ok = np.isfinite(g) & np.isfinite(m)
            out["baselines"]["frac_conditions_gears_beats_meanpert_delta"] = (
                float(np.mean(g[ok] > m[ok])) if ok.any() else float("nan")
            )
            log("SUMMARY:", json.dumps(out["baselines"], indent=2))
        out["status"] = "ok"
    except Exception as e:
        out["status"] = f"error: {type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        log("ERROR:", out["status"])
        traceback.print_exc()

    dump()
    log("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
