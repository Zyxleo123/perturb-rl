#!/usr/bin/env python
"""Choosing the neighbour SET, not just ranking neighbours.

`repr_probe.py` leaves a gap that the similarity alone cannot explain. With the same
k = 10 and a PERFECT pairwise similarity, the kNN average reaches 12.5% / 14.1% top-1;
greedily choosing 10 atlas genes whose mean best reconstructs the target reaches
38.9% / 41.4%. Both know the answer. The difference is only how the ten are combined,
so roughly two thirds of the attainable headroom lives in the COMBINER, not in the
ordering.

The mechanism this script tests: the top-k neighbours of a gene are near-duplicates of
EACH OTHER -- same complex, same pathway -- so averaging them double-counts one direction
and leaves the target's distinctive component under-weighted. Greedy avoids that because
each new pick is chosen given the ones already taken.

The fix does not need the answer. Write the prediction as a weighted combination of atlas
signatures, y_hat = sum_j w_j y_j. Then

    cos(y_hat, y_g) = (sum_j w_j <y_j, y_g>) / sqrt(w^T G w),     G_jl = <y_j, y_l>

and maximising it over w has the closed form  w* ∝ G^{-1} b,  b_j = <y_j, y_g>. The Gram
matrix G is between ATLAS genes, which are measured, so it is known exactly. Only b is
unknown, and b_j = s(g,j) ||y_j|| is exactly what a gene representation estimates. So:

    knn_avg   w = normalised similarity over the top k      [what GAPS uses]
    gls       w = (G_SS + lambda I)^{-1} b_S over the top m [this script]
    mmr       greedy top-k with a redundancy penalty, the discrete version

`gls` is ordinary ridge / generalised least squares: it whitens the neighbourhood, so two
redundant neighbours share the weight one of them would have had. λ and m are chosen on a
held-out fold of genes, never on the gene being predicted.

Each combiner is run on three similarity sources, to separate the two causes cleanly:
  oracle   s = the true signature cosine         -- how much the COMBINER alone is worth
  learned  s = repr_predict's out-of-fold non-negative combination of prior kernels
  go       s = GO Jaccard                        -- the GAPS baseline

Same task, atlas, predictor shape and tie-aware metric as GAPS gap 0 throughout.
Writes <art>/repr_select/{result_<dataset>.npz, summary.json, report.md}. Exits 0.

Usage: python -I repr_select.py --art-dir D --gears-dir D --datasets A B [--knn 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_headroom as RH          # noqa: E402
import rl_acquire as RA           # noqa: E402
import repr_probe as RP           # noqa: E402
import repr_predict as RD         # noqa: E402

SEED = 0
N_FOLDS = 5
M_GRID = (10, 25, 50, 100)              # neighbourhood size the combiner may use
LAM_GRID = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)   # ridge, relative to mean diag(G)
MMR_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ================================================================== combiners
def predict_knn(S, YA, k):
    """The GAPS predictor: normalised-similarity average of the top k."""
    return RP.knn_from_sim(S, YA, k=k)


def predict_gls(S, YA, G, m, lam, nrm):
    """y_hat_g = Y_S^T (G_SS + lam I)^{-1} b_S over the m most similar atlas genes.

    b_j = s(g,j) * ||y_j|| is the estimate of <y_j, y_g> that the representation gives.
    Only the held-out gene's signature is unknown; G is between measured genes."""
    K = S.shape[0]
    out = np.zeros_like(YA)
    mm = min(m, K - 1)
    top = np.argpartition(-S, mm - 1, axis=1)[:, :mm]
    ridge = lam * float(np.mean(np.diag(G)))
    for i in range(K):
        idx = top[i]
        b = np.maximum(S[i, idx], 0.0) * nrm[idx]
        A = G[np.ix_(idx, idx)] + ridge * np.eye(mm)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        out[i] = w.astype(np.float32) @ YA[idx]
    return out


def predict_mmr(S, YA, Catlas, k, beta):
    """Greedy top-k with a redundancy penalty: pick argmax s(g,j) - beta * max_{l in S}
    cos(y_j, y_l). The discrete sibling of `gls`; both use only measured quantities."""
    K = S.shape[0]
    out = np.zeros_like(YA)
    for i in range(K):
        s = S[i].copy()
        s[i] = -np.inf
        chosen, pen = [], np.zeros(K, np.float32)
        for _ in range(k):
            v = s - beta * pen
            if chosen:
                v[np.array(chosen)] = -np.inf
            j = int(np.argmax(v))
            chosen.append(j)
            pen = np.maximum(pen, np.abs(Catlas[j]))
        c = np.array(chosen)
        w = np.maximum(S[i, c], 0.0)
        out[i] = (w / w.sum()).astype(np.float32) @ YA[c] if w.sum() > 0 else YA.mean(0)
    return out


# ================================================================== selection
def tune(make_pred, grid, YA, Qu, own, K, folds, val_fold):
    """Choose a hyper-parameter on genes OUTSIDE the fold being predicted.

    One parameter per fold: for fold f, score every setting on the genes of a different
    fold (val_fold[f]) and keep the winner. No gene's prediction is ever produced by a
    setting selected on that gene."""
    # Only the per-gene reciprocal rank is kept during the sweep; holding every
    # setting's full K x G prediction would cost GBs for no reason. The few winning
    # settings are recomputed afterwards.
    val = {}
    for g in grid:
        P = make_pred(g)
        S = (Qu @ RH.unit(P).T).astype(np.float32)
        _, rank, rr = RP.score_matrix(S, own)
        val[g] = RP.per_gene(rank, rr, own, K)[1]
        del P, S
    picked = {f: max(grid, key=lambda g: float(val[g][folds[val_fold[f]]].mean()))
              for f in range(len(folds))}
    out = np.zeros_like(YA)
    for g in set(picked.values()):                 # recompute each winner once
        P = make_pred(g)
        for f, te in enumerate(folds):
            if picked[f] == g:
                out[te] = P[te]
        del P
    return out, picked


def run(ds, cache_path, priors_path, gears_dir, out_dir, knn):
    W = RA.World(cache_path)
    K = W.K
    log(f"{ds}: K={K}, knn={knn}")

    D = RH.build(ds, os.path.join(gears_dir, ds, "perturb_processed.h5ad"),
                 np.random.default_rng(RH.SEED))
    mk = D["mask"]
    if not (list(D["cands"]) == [str(x) for x in W.cands]
            and np.allclose(D["dF"][:, mk], W.Y, atol=1e-6)):
        raise RuntimeError("rl_headroom.build no longer reproduces the cache; refusing to mix")
    YA = D["dA"][:, mk].astype(np.float32)
    del D
    YAu = RH.unit(YA)
    Csig = (YAu @ YAu.T).astype(np.float32)          # true gene-gene signature cosine
    G = (YA.astype(np.float64) @ YA.astype(np.float64).T)
    nrm = np.sqrt(np.maximum(np.diag(G), 1e-24))

    Qu = RH.unit(W.Q)
    own = W.owner.astype(np.int64)
    folds = RH.folds_of(K, np.random.default_rng(SEED), k=N_FOLDS)
    val_fold = {f: (f + 1) % N_FOLDS for f in range(N_FOLDS)}

    # ------------------------------------------------------ similarity sources
    Jg = W.JJ.copy()
    np.fill_diagonal(Jg, 0.0)
    views = {"go_jaccard": Jg.astype(np.float32)}
    rich = W.rich()
    views["go_idf"], views["go_emb"] = rich["idf"].astype(np.float32), rich["emb"].astype(np.float32)
    z = np.load(priors_path, allow_pickle=False)
    pnames = [str(x) for x in z["names"]]
    if [str(x) for x in z["cands"]] != [str(x) for x in W.cands]:
        raise RuntimeError("prior kernels are not aligned to this cache's candidates")
    # How many genes did the prior sources not cover? scripts/related_work.py mean-imputes
    # those rows, which makes them identical to each other -- a block of spuriously maximal
    # similarity that is unrelated to any signature. RPE1 coverage is 84%, K562 99.5%, so
    # this is quantified here rather than assumed harmless.
    imputed = {}
    for i, nm in enumerate(pnames):
        R = z["kernels"][i]
        _, inv, cnt = np.unique(np.round(R, 5), axis=0, return_inverse=True, return_counts=True)
        imputed[nm] = float((cnt[inv] > 1).mean())
    log("  duplicate (mean-imputed) kernel rows: "
        + ", ".join(f"{k}={v*100:.1f}%" for k, v in imputed.items()))

    prior_views = []
    for i, nm in enumerate(pnames):
        if nm in RD.MEASUREMENT_KERNELS:           # another line's MEASUREMENT, not a prior
            continue
        v = RD.cosnorm(z["kernels"][i].astype(np.float64))
        np.fill_diagonal(v, 0.0)
        views[nm] = v
        prior_views.append(nm)
    names = ["go_jaccard", "go_idf", "go_emb"] + prior_views
    Slearn, Wts = RD.learned_similarity([views[n] for n in names], Csig, folds, K,
                                        np.random.default_rng(SEED))
    log(f"  learned similarity over {len(names)} prior views (no measurement kernels)")

    sources = {"oracle": Csig.copy(), "learned": Slearn, "go": Jg.astype(np.float32)}
    for S in sources.values():
        np.fill_diagonal(S, -np.inf)

    # ------------------------------------------------------------- combiners
    res, preds = {}, {}
    for sname, S in sources.items():
        Catlas = Csig.copy()
        np.fill_diagonal(Catlas, 0.0)
        jobs = {
            "knn_avg": (lambda kk, S=S: predict_knn(S, YA, kk), (knn,)),
            "gls": (lambda p, S=S: predict_gls(S, YA, G, p[0], p[1], nrm),
                    [(m, l) for m in M_GRID for l in LAM_GRID]),
            "mmr": (lambda b, S=S, C=Catlas: predict_mmr(S, YA, C, knn, b), MMR_GRID),
        }
        for cname, (fn, grid) in jobs.items():
            t0 = time.time()
            P, picked = tune(fn, list(grid), YA, Qu, own, K, folds, val_fold)
            Sq = (Qu @ RH.unit(P).T).astype(np.float32)
            m, rank, rr = RP.score_matrix(Sq, own)
            key = f"{sname}__{cname}"
            preds[key] = RP.per_gene(rank, rr, own, K)[0]
            res[key] = {"metrics": m, "chosen_per_fold": {str(k): str(v) for k, v in picked.items()},
                        "secs": round(time.time() - t0, 1)}
            log(f"  {key:22s} top1={m['top1']:.4f} top10={m['top10']:.4f} mrr={m['mrr']:.4f} "
                f"medrank={m['median_rank']:.0f}  picked={sorted(set(map(str, picked.values())))}"
                f"  [{res[key]['secs']}s]")

    # ceilings for the table
    Pg, _ = RP.greedy_reconstruct(G, YA, k=knn)
    for nm, P in (("ceiling__oracle_greedy", Pg), ("ceiling__measured", YA)):
        m, rank, rr = RP.score_matrix((Qu @ RH.unit(P).T).astype(np.float32), own)
        preds[nm] = RP.per_gene(rank, rr, own, K)[0]
        res[nm] = {"metrics": m, "chosen_per_fold": {}, "secs": 0.0}
        log(f"  {nm:22s} top1={m['top1']:.4f} mrr={m['mrr']:.4f} medrank={m['median_rank']:.0f}")

    np.savez_compressed(os.path.join(out_dir, f"result_{ds}.npz"),
                        cands=np.array([str(x) for x in W.cands]),
                        view_names=np.array(names), learned_weights=Wts.astype(np.float32),
                        **{f"rank_{k}": v.astype(np.float32) for k, v in preds.items()})
    return {"dataset": ds, "K": int(K), "knn": int(knn), "n_queries": int(Qu.shape[0]),
            "views": names, "imputed_kernel_row_fraction": imputed, "results": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--knn", type=int, default=10)
    a = ap.parse_args()

    out_dir = os.path.join(a.art_dir, "repr_select")
    os.makedirs(out_dir, exist_ok=True)
    check = json.load(open(os.path.join(a.art_dir, "rl_acquire", "cache_check.json")))
    rel_dir = os.path.join(a.art_dir, "rl_acquire", "related")

    summary = {"knn": a.knn, "datasets": {}, "errors": []}
    for ds in a.datasets:
        try:
            t0 = time.time()
            s = run(ds, check["datasets"][ds]["path"],
                    os.path.join(rel_dir, f"priors_{ds}.npz"), a.gears_dir, out_dir, a.knn)
            s["secs"] = round(time.time() - t0, 1)
            summary["datasets"][ds] = s
            log(f"{ds}: done in {s['secs']}s")
        except Exception as e:
            summary["errors"].append({"dataset": ds, "error": f"{type(e).__name__}: {e}",
                                      "traceback": traceback.format_exc()[-2000:]})
            log(f"{ds}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc()

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    L = ["# Unseen-gene retrieval: ranking the neighbours vs combining them", "",
         "Rows are `similarity source` x `combiner`. `knn_avg` is the predictor every",
         "number in GAPS.md uses. Hyper-parameters are chosen on a different fold of genes",
         "than the one being predicted.", ""]
    for ds, s in summary["datasets"].items():
        L += [f"## {ds} (K={s['K']}, {s['n_queries']} queries)", "",
              "| similarity | combiner | top-1 | top-10 | MRR | median rank |",
              "|---|---|---:|---:|---:|---:|"]
        for k, v in s["results"].items():
            src, comb = k.split("__", 1)
            m = v["metrics"]
            L.append(f"| {src} | {comb} | {m['top1']*100:.1f}% | {m['top10']*100:.1f}% "
                     f"| {m['mrr']:.3f} | {m['median_rank']:.0f} |")
        L.append("")
    if summary["errors"]:
        L += ["## Errors", ""] + [f"- {e['dataset']}: {e['error']}" for e in summary["errors"]]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    log("wrote", os.path.join(out_dir, "report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
