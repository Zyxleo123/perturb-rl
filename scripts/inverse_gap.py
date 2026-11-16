#!/usr/bin/env python
"""How much performance headroom does inverse design actually have?

Measures one ladder on the SAME task — given a target cell state, name the
perturbation that produced it — so the rungs are directly comparable:

  CEILING   empirical centroid: rank candidates by the Δ-signature measured from
            real labelled cells of that perturbation. Needs labelled data for
            every candidate, so it is not a usable designer, but it bounds what
            any method can extract from a target state.
  ORACLE    forward-model search: rank candidates by the Δ-signature GEARS
            PREDICTS for them. This is what every forward-oracle-driven inverse
            method is limited by -- PDGrapher's inverse map, LabCompass's MAP /
            Langevin / guidance, and any RL policy whose reward is this oracle.
            No method built on this oracle can beat the oracle's own ranking by
            more than noise.
  MEAN      non-conditional control: rank by the average perturbation effect,
            ignoring which perturbation it is. Anything at or below this line
            carries no perturbation-specific information.
  CHANCE    1 / K.

CEILING minus ORACLE is the oracle-induced performance gap: the part of the task
that is lost to the reward model rather than to the generator. If that gap is
large, the bottleneck is the oracle, and post-training the generator cannot fix it.

Also reports per-condition oracle fidelity -- cos(GEARS Δ, measured Δ) against
cos(mean-perturbation Δ, measured Δ) -- which is the forward-model version of the
same question.

Writes <out>/inverse_gap.json and <out>/npz/<dataset>_oracle_signatures.npz.
Always exits 0.
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

POOL_SIZES = (1, 8, 32, 128)
TOPKS = (1, 5, 10, 20)
RNG_SEED = 0


def log(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.strftime('%H:%M:%S')} rss={rss:.1f}GB]", *a, flush=True)


def densify(x):
    import scipy.sparse as sp
    return np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)


def canon_condition(c):
    """Canonical name: sorted gene set, 'ctrl' padding dropped.

    Norman lists the same single-gene perturbation twice ('SPI1+ctrl' and
    'ctrl+SPI1'); 46 of 276 candidates are such twins. Scoring them as distinct
    inflates K and marks a correct answer wrong when the model picks the twin, so
    conditions are merged here before anything is ranked.
    """
    parts = sorted(p for p in str(c).split("+") if p != "ctrl")
    return "+".join(parts) if parts else "ctrl"


def unit(M):
    M = np.atleast_2d(np.asarray(M, dtype=np.float32))
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def rank_metrics(scores_rows, true_idx, K):
    """scores_rows: (n_queries, K). Returns top-k accs + rank stats."""
    out = {}
    order = np.argsort(-scores_rows, axis=1)
    ranks = np.array([int(np.flatnonzero(order[i] == t)[0]) + 1
                      for i, t in enumerate(true_idx)])
    for k in TOPKS:
        out[f"top{k}_acc"] = float(np.mean(ranks <= k))
    out["median_rank"] = float(np.median(ranks))
    out["mean_reciprocal_rank"] = float(np.mean(1.0 / ranks))
    out["n_queries"] = int(ranks.size)
    out["n_candidates"] = int(K)
    out["chance_top1"] = float(1.0 / K)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataloader-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", default="no_test")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset-label", default="norman")
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--max-candidates", type=int, default=0, help="0 = all")
    ap.add_argument("--queries-per-cond", type=int, default=3)
    ap.add_argument(
        "--skip-cell-graphs",
        action="store_true",
        help="skip unpickling data_pyg/cell_graphs.pkl (3.7 GB, ~20 min over NFS). "
             "predict() never touches it. set_pert_genes() and prepare_split() still run "
             "on the real PertData, so the perturbation index mapping stays GEARS's own.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "npz"), exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.dataset_label,
        "checkpoint": args.ckpt_dir,
        "task": "given a target cell state, rank candidate perturbations",
        "rungs": {
            "ceiling": "empirical Δ-centroid from labelled cells (not a usable designer)",
            "oracle": "GEARS-predicted Δ-signature (bounds every oracle-driven method)",
            "mean": "average perturbation effect, non-conditional",
            "chance": "1/K",
        },
        "caveat": (
            "Checkpoint gears_misc_umi_no_test was trained on ALL Norman perturbations, so the "
            "ORACLE rung is measured in-distribution and is an OPTIMISTIC upper bound. The true "
            "oracle-induced gap on unseen perturbations is at least this large."
        ),
        "status": "started",
    }
    out_path = os.path.join(args.out_dir, "inverse_gap.json")

    def dump():
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    dump()
    try:
        import pandas as pd
        import torch
        from gears import GEARS, PertData

        log("torch", torch.__version__, "cuda:", torch.cuda.is_available())
        if not torch.cuda.is_available():
            args.device = "cpu"
            log("no CUDA -> cpu")

        if args.skip_cell_graphs:
            # PertData.load() unpickles data_pyg/cell_graphs.pkl unconditionally
            # (pertdata.py ~l.200). It is 3.7 GB and predict() never reads it, so
            # intercept that one filename and leave every other pickle.load alone
            # (prepare_split needs its own split pickles).
            import pickle as _pickle

            _real_load = _pickle.load

            def _skip_cell_graphs(f, *a, **k):
                if str(getattr(f, "name", "")).endswith("cell_graphs.pkl"):
                    log("skipping cell_graphs.pkl (unused by predict())")
                    return None
                return _real_load(f, *a, **k)

            _pickle.load = _skip_cell_graphs

        log("loading PertData" + (" [skip-cell-graphs]" if args.skip_cell_graphs
                                  else " (unpickles 3.7 GB cell-graph cache; slow)"))
        pert_data = PertData(args.work_dir)
        pert_data.load(data_path=args.dataloader_dir)
        pert_data.prepare_split(split=args.split, seed=args.seed)

        if args.skip_cell_graphs:
            _pickle.load = _real_load
            # These two are normally set by get_dataloader(); copied verbatim from
            # GEARS pertdata.py l.429-430 so the gene ordering is identical.
            pert_data.node_map = {
                x: it for it, x in enumerate(pert_data.adata.var.gene_name)
            }
            pert_data.gene_names = pert_data.adata.var.gene_name

        pert_data.dataloader = None
        model = GEARS(pert_data, device=args.device, weight_bias_track=False,
                      proj_name="gears", exp_name="inverse_gap")
        model.load_pretrained(args.ckpt_dir)
        out["skip_cell_graphs"] = bool(args.skip_cell_graphs)
        log("GEARS ready")

        adata = model.adata
        raw_conds = adata.obs["condition"].astype(str).values
        conds = np.asarray([canon_condition(c) for c in raw_conds])
        n_dup = len(set(raw_conds)) - len(set(conds))
        out["n_duplicate_encodings_merged"] = int(n_dup)
        log(f"merged {n_dup} duplicate condition encodings "
            f"({len(set(raw_conds))} raw -> {len(set(conds))} canonical)")
        ctrl_mask = conds == "ctrl"
        X = adata.X
        ctrl_mean = densify(X[ctrl_mask]).mean(axis=0)
        mean_pert_delta = densify(X[~ctrl_mask]).mean(axis=0) - ctrl_mean
        log(f"cells={adata.n_obs} genes={adata.n_vars} ctrl={ctrl_mask.sum()}")

        def parse(c):
            return [p for p in c.split("+") if p != "ctrl"]

        vc = pd.Series(conds).value_counts()
        cands = [c for c in vc.index
                 if c != "ctrl" and vc[c] >= args.min_cells
                 and all(g in model.pert_list for g in parse(c))]
        if args.max_candidates:
            cands = cands[: args.max_candidates]
        K = len(cands)
        log(f"{K} candidate perturbations")
        out["n_candidates"] = K

        # ---- split each condition's cells: half -> centroid, half -> queries
        rng = np.random.default_rng(RNG_SEED)
        emp, qidx = [], {}
        for c in cands:
            rows = np.flatnonzero(conds == c)
            rng.shuffle(rows)
            h = rows.size // 2
            emp.append(densify(X[rows[:h]]).mean(axis=0) - ctrl_mean)
            qidx[c] = rows[h:]
        emp = np.vstack(emp).astype(np.float32)

        # ---- GEARS-predicted signature for every candidate
        log("predicting Δ-signature for every candidate with GEARS")
        orc = np.zeros_like(emp)
        t0 = time.time()
        failed = []
        for i, c in enumerate(cands):
            pl = parse(c)
            try:
                p = np.asarray(model.predict([pl])["_".join(pl)]).ravel()
                orc[i] = p - ctrl_mean
            except Exception as e:
                failed.append(c)
                log(f"  predict failed for {c}: {type(e).__name__}: {e}")
            if i % 50 == 0:
                el = time.time() - t0
                log(f"  [{i+1}/{K}] {el:.0f}s elapsed, ~{el/(i+1)*(K-i-1):.0f}s left")
        out["n_predict_failed"] = len(failed)
        log(f"predictions done in {time.time()-t0:.0f}s ({len(failed)} failed)")

        # gene_names must travel with these: the dataloader's gene space
        # (norman_umi_go, 5054 genes) differs from the GEARS-processed release used
        # by profile_datasets.py (norman, 5045), so names are NOT interchangeable.
        np.savez_compressed(
            os.path.join(args.out_dir, "npz", f"{args.dataset_label}_oracle_signatures.npz"),
            conditions=np.asarray(cands), oracle_sig=orc, empirical_sig=emp,
            ctrl_mean=ctrl_mean.astype(np.float32),
            mean_pert_delta=mean_pert_delta.astype(np.float32),
            # dtype=str matters: adata.var["gene_name"] is a pandas Categorical, so
            # .astype(str).values yields an OBJECT array, which np.savez pickles and
            # np.load(allow_pickle=False) then refuses to read.
            gene_names=np.asarray(
                adata.var["gene_name"].astype(str).values
                if "gene_name" in adata.var else adata.var_names.astype(str),
                dtype=str,
            ),
        )

        # ---- oracle fidelity: does GEARS beat the non-conditional baseline?
        U_emp, U_orc = unit(emp), unit(orc)
        u_mean = unit(mean_pert_delta)[0]
        fid = np.sum(U_orc * U_emp, axis=1)
        fid_mean = U_emp @ u_mean
        # how much of GEARS's similarity is just the shared average effect?
        out["oracle_fidelity"] = {
            "median_cos_gears_vs_measured": float(np.median(fid)),
            "median_cos_meanpert_vs_measured": float(np.median(fid_mean)),
            "frac_conditions_gears_beats_meanpert": float(np.mean(fid > fid_mean)),
            "median_cos_gears_vs_meanpert": float(np.median(U_orc @ u_mean)),
        }
        log("oracle fidelity:", json.dumps(out["oracle_fidelity"], indent=2))

        # ---- the ladder
        log("running retrieval ladder")
        cond_to_i = {c: i for i, c in enumerate(cands)}
        results = {"ceiling": [], "oracle": [], "mean": []}
        for n in POOL_SIZES:
            Q, true_i = [], []
            for c in cands:
                q = qidx[c]
                if q.size < n:
                    continue
                for _ in range(args.queries_per_cond):
                    pick = rng.choice(q, size=n, replace=False)
                    Q.append(densify(X[pick]).mean(axis=0) - ctrl_mean)
                    true_i.append(cond_to_i[c])
            if not Q:
                continue
            Q = unit(np.vstack(Q))
            true_i = np.asarray(true_i)
            for name, S in (("ceiling", U_emp), ("oracle", U_orc)):
                m = rank_metrics(Q @ S.T, true_i, K)
                m["n_cells_pooled"] = int(n)
                results[name].append(m)
            # non-conditional: identical score for every candidate -> true chance;
            # score each candidate by how much the target looks like the average effect
            Smean = np.repeat(u_mean[None, :], K, axis=0)
            m = rank_metrics(Q @ Smean.T + 1e-9 * rng.standard_normal((Q.shape[0], K)),
                             true_i, K)
            m["n_cells_pooled"] = int(n)
            results["mean"].append(m)
            log(f"  n={n:3d}  ceiling top1={results['ceiling'][-1]['top1_acc']:.3f}  "
                f"oracle top1={results['oracle'][-1]['top1_acc']:.3f}  "
                f"mean top1={results['mean'][-1]['top1_acc']:.3f}  "
                f"chance={1.0/K:.4f}")
        out["ladder"] = results

        # ---- headline gap
        if results["ceiling"] and results["oracle"]:
            ce = {r["n_cells_pooled"]: r for r in results["ceiling"]}
            oc = {r["n_cells_pooled"]: r for r in results["oracle"]}
            shared = sorted(set(ce) & set(oc))
            out["headline"] = {
                "per_pool_size": {
                    str(n): {
                        "ceiling_top1": ce[n]["top1_acc"],
                        "oracle_top1": oc[n]["top1_acc"],
                        "absolute_gap": ce[n]["top1_acc"] - oc[n]["top1_acc"],
                        "frac_of_ceiling_retained": (
                            oc[n]["top1_acc"] / ce[n]["top1_acc"]
                            if ce[n]["top1_acc"] > 0 else float("nan")
                        ),
                        "ceiling_median_rank": ce[n]["median_rank"],
                        "oracle_median_rank": oc[n]["median_rank"],
                    }
                    for n in shared
                }
            }
            log("HEADLINE:", json.dumps(out["headline"], indent=2))
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
