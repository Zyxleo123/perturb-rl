#!/usr/bin/env python
"""Does a better gene representation close the unseen-gene gap?

`repr_probe.py` measures that the unseen-gene failure is a SIMILARITY failure: the
neighbour-averaging predictor class can reach the atlas ceiling, but GO Jaccard orders
neighbours almost at random with respect to the thing that matters (Spearman ~0.12
against true signature similarity). This script asks the obvious next question with the
representations already on disk: swap the kNN's similarity and measure what happens.

Everything except the similarity is held fixed. Same task as `story_cases.case_gap` and
GAPS gap 0 -- each gene held out in turn, its signature predicted from the OTHER genes'
half-A signatures, its held-out queries ranked against all K predictions -- and the same
predictor, `rl_headroom.go_knn_predict`'s weighted average of the k=10 most similar
atlas signatures. Only the matrix that says "most similar" changes.

Views scored (all K x K, cosine-normalised so weights are comparable):
  go_jaccard        GO term Jaccard. Every number in GAPS.md uses this.         [baseline]
  <8 prior kernels> IterPert's released multimodal priors (Huang et al., RECOMB 2024),
                    already aligned to our candidates by scripts/related_work.py: PoPS,
                    ESM protein language model, BioGPT text, node2vec on the GO graph,
                    three optical pooled screens, and the OTHER CELL LINE's Perturb-seq.
  iterpert_mean     IterPert's own combination: mean of max-normalised kernels.
  learned           a non-negative combination of the views above, fitted OUT OF FOLD to
                    predict true signature similarity. This is the method: it asks
                    whether the information is present in the priors and merely badly
                    weighted by Jaccard.
  oracle_sim        similarity = the true signature cosine.          [ceiling of ordering]
  oracle_greedy     the 10 atlas genes whose mean best reconstructs the held-out
                    signature.                                  [ceiling of the class]
  measured / mean   the atlas ceiling and the constant floor.

IMPORTANT, and reported separately: `rpe1_kernel` (on K562) and `k562_perturbseq_kernel`
(on RPE1) are the other cell line's MEASURED Perturb-seq, not prior knowledge. A method
using them is answering "has this gene been measured somewhere else?", which is GAPS's
unseen-CONTEXT axis, not the unseen-GENE one. They are scored because the comparison is
informative, and flagged as `is_measurement` everywhere so no table mixes them up.

Out-of-fold discipline: 5 folds over candidates; the learned combination for a gene in
fold f is fitted only on pairs whose BOTH members lie outside fold f, so no gene is ever
ranked by a model that has seen it. The atlas itself is the same leave-one-out atlas
every GAPS number uses (a gene never predicts itself).

Writes <art>/repr_predict/{result_<dataset>.npz, summary.json, report.md}. Exits 0.

Usage: python -I repr_predict.py --art-dir D --gears-dir D --datasets A B [--knn 10]
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

N_FOLDS = 5
SEED = 0
MAX_PAIRS = 600_000               # training pairs per fold for the learned combination
# Kernels that are a MEASUREMENT of perturbation effect in another cell line, not a prior.
MEASUREMENT_KERNELS = {"rpe1_kernel", "k562_perturbseq_kernel"}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def cosnorm(Kmat):
    """A kernel is feat feat^T; dividing by the diagonal makes it the cosine between the
    feature rows, so every view is on the same 0-1-ish scale before they are combined."""
    d = np.sqrt(np.maximum(np.diag(Kmat), 1e-12))
    out = Kmat / d[:, None] / d[None, :]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def maxnorm(Kmat):
    """IterPert's own normalisation before averaging kernels (iterpert.py)."""
    m = np.abs(Kmat).max()
    return (Kmat / m).astype(np.float32) if m > 0 else Kmat.astype(np.float32)


# ------------------------------------------------------------------ learned view
def learned_similarity(views, Csig, folds, K, rng):
    """Non-negative least squares combination of the views, fitted out of fold.

    For each fold f: sample pairs (i, j) with i, j both OUTSIDE f, regress the true
    signature cosine on the views' values for those pairs, and use the fitted weights to
    score every (i in f, j != i) pair. Non-negative so the result is still a similarity
    and the weights read as "how much this view is worth".
    """
    from scipy.optimize import nnls
    V = np.stack(views, 0)                       # n_views x K x K
    out = np.zeros((K, K), np.float32)
    W = []
    for f, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(K), te)
        a = rng.choice(tr, size=min(MAX_PAIRS, tr.size * (tr.size - 1) // 2), replace=True)
        b = rng.choice(tr, size=a.size, replace=True)
        keep = a != b
        a, b = a[keep], b[keep]
        X = V[:, a, b].T.astype(np.float64)
        y = Csig[a, b].astype(np.float64)
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        w, _ = nnls(X1, y - y.min())             # shift so an all-zero fit is admissible
        W.append(w)
        out[te] = np.tensordot(w[:-1].astype(np.float32), V[:, te, :], axes=(0, 0))
    return out, np.array(W)


# ------------------------------------------------------------------- ordering
def ordering_quality(Sim, Csig, K, knn):
    """How well a view orders a gene's neighbours by TRUE signature similarity: the
    mechanism column. A view can only help the predictor through this."""
    sim = Sim.copy()
    np.fill_diagonal(sim, -np.inf)
    cs = Csig.copy()
    np.fill_diagonal(cs, -np.inf)
    kk = min(knn, K - 1)
    vt = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    tt = np.argpartition(-cs, kk - 1, axis=1)[:, :kk]
    rec = np.mean([len(set(vt[i].tolist()) & set(tt[i].tolist())) / kk for i in range(K)])
    # mean true-signature cosine of the neighbours the view actually picks
    picked = float(np.mean(cs[np.arange(K)[:, None], vt]))
    best = float(np.mean(cs[np.arange(K)[:, None], tt]))
    iu = np.triu_indices(K, 1)
    sub = np.random.default_rng(1).choice(iu[0].size, size=min(300_000, iu[0].size), replace=False)
    sp = RP.spearman(Sim[iu[0][sub], iu[1][sub]], Csig[iu[0][sub], iu[1][sub]])
    return {"spearman_vs_signature": sp, "recall_at_k": float(rec),
            "picked_neighbour_cos": picked, "best_neighbour_cos": best}


# ===================================================================== driver
def run(ds, cache_path, priors_path, gears_dir, out_dir, knn):
    W = RA.World(cache_path)
    K = W.K
    rng = np.random.default_rng(SEED)
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
    Csig = (YAu @ YAu.T).astype(np.float32)
    Gram = YA.astype(np.float64) @ YA.astype(np.float64).T

    Qu = RH.unit(W.Q)
    own = W.owner.astype(np.int64)
    q1 = np.array([W.q_of[i][0] for i in range(K)])
    Qu1, own1 = RH.unit(W.Q[q1]), np.arange(K)

    # -------------------------------------------------------------- views
    views, flags = {}, {}
    Jg = W.JJ.copy()
    np.fill_diagonal(Jg, 0.0)
    views["go_jaccard"] = Jg.astype(np.float32)
    flags["go_jaccard"] = False

    z = np.load(priors_path, allow_pickle=False)
    pnames = [str(x) for x in z["names"]]
    if [str(x) for x in z["cands"]] != [str(x) for x in W.cands]:
        raise RuntimeError("prior kernels are not aligned to this cache's candidates")
    raw = {}
    for i, nm in enumerate(pnames):
        Kk = z["kernels"][i].astype(np.float64)
        raw[nm] = Kk
        v = cosnorm(Kk)
        np.fill_diagonal(v, 0.0)
        views[nm] = v
        flags[nm] = nm in MEASUREMENT_KERNELS
    log(f"  views: go_jaccard + {len(pnames)} prior kernels "
        f"({sum(flags.values())} of them are another line's MEASUREMENT)")

    # the two richer GO views the blind-regime section tested on the POLICY and found
    # no better than Jaccard; never tested on the PREDICTOR, which is where gap 0 says
    # the representation binds.
    rich = W.rich()
    for nm, key in (("go_idf", "idf"), ("go_emb", "emb")):
        v = rich[key].astype(np.float32).copy()
        views[nm] = v
        flags[nm] = False

    mean_k = np.mean([maxnorm(raw[nm]) for nm in pnames], 0).astype(np.float32)
    np.fill_diagonal(mean_k, 0.0)
    views["iterpert_mean"] = mean_k
    flags["iterpert_mean"] = any(flags[nm] for nm in pnames)

    # learned combination, out of fold, over the PRIOR views only (no measurement kernel)
    prior_names = ["go_jaccard", "go_idf", "go_emb"] + [n for n in pnames if not flags[n]]
    folds = RH.folds_of(K, np.random.default_rng(SEED), k=N_FOLDS)
    Slearn, Wts = learned_similarity([views[n] for n in prior_names], Csig, folds, K, rng)
    np.fill_diagonal(Slearn, 0.0)
    views["learned_priors"] = Slearn
    flags["learned_priors"] = False
    log(f"  learned combination over {len(prior_names)} prior views, {N_FOLDS}-fold "
        f"out of fold; mean weights " +
        ", ".join(f"{n}={Wts[:, i].mean():.3f}" for i, n in enumerate(prior_names)))

    # learned combination INCLUDING the measurement kernel, for the separate table
    all_names = ["go_jaccard", "go_idf", "go_emb"] + list(pnames)
    Slearn_all, Wts_all = learned_similarity([views[n] for n in all_names], Csig, folds, K, rng)
    np.fill_diagonal(Slearn_all, 0.0)
    views["learned_all"] = Slearn_all
    flags["learned_all"] = any(flags.values())

    # ---------------------------------------------------------- predictors
    # A view with negative entries would otherwise spend a neighbour slot on the gene
    # itself (diagonal 0 beats a negative cosine), so the diagonal is barred everywhere.
    preds, order = {}, {}
    for nm, Sim in views.items():
        Sx = Sim.copy()
        np.fill_diagonal(Sx, -np.inf)
        preds[nm] = RP.knn_from_sim(Sx, YA, k=knn)
        order[nm] = ordering_quality(Sim, Csig, K, knn)

    # reconciliation: our go_jaccard row must be rl_headroom's own GO-kNN predictor,
    # the one every number in GAPS.md uses. A drift here invalidates the comparison.
    P_ref = RH.go_knn_predict(Jg, YA, YA.mean(0), k=knn)
    drift = float(np.abs(RH.unit(P_ref) - RH.unit(preds["go_jaccard"])).max())
    log(f"  go_jaccard vs rl_headroom.go_knn_predict: max |delta| = {drift:.2e}")

    Cs = Csig.copy()
    np.fill_diagonal(Cs, -np.inf)
    preds["oracle_sim"] = RP.knn_from_sim(Cs, YA, k=knn)
    order["oracle_sim"] = ordering_quality(Csig, Csig, K, knn)
    flags["oracle_sim"] = True
    Pg, sel = RP.greedy_reconstruct(Gram, YA, k=knn)
    preds["oracle_greedy"] = Pg
    flags["oracle_greedy"] = True
    preds["measured"] = YA
    flags["measured"] = True
    preds["mean"] = np.repeat(YA.mean(0)[None], K, 0).astype(np.float32)
    flags["mean"] = False

    res, ranks = {}, {}
    for nm, P in preds.items():
        S = (Qu @ RH.unit(P).T).astype(np.float32)
        m, rank, rr = RP.score_matrix(S, own)
        rg, _ = RP.per_gene(rank, rr, own, K)
        ranks[nm] = rg
        m1, _, _ = RP.score_matrix((Qu1 @ RH.unit(P).T).astype(np.float32), own1)
        res[nm] = {"all_queries": m, "first_query": m1,
                   "is_measurement_or_oracle": bool(flags.get(nm, False)),
                   "ordering": order.get(nm)}
        tag = "  [uses measurement/oracle]" if flags.get(nm) else ""
        log(f"  {nm:22s} top1={m['top1']:.4f} top10={m['top10']:.4f} mrr={m['mrr']:.4f} "
            f"medrank={m['median_rank']:.0f}{tag}")

    np.savez_compressed(
        os.path.join(out_dir, f"result_{ds}.npz"),
        cands=np.array([str(x) for x in W.cands]),
        view_names=np.array(list(views.keys())),
        prior_names=np.array(prior_names), learned_weights=Wts.astype(np.float32),
        all_names=np.array(all_names), learned_weights_all=Wts_all.astype(np.float32),
        folds=np.concatenate([np.full(f.size, i) for i, f in enumerate(folds)])[
            np.argsort(np.concatenate(folds))].astype(np.int16),
        **{f"rank_{k}": v.astype(np.float32) for k, v in ranks.items()},
    )
    return {"dataset": ds, "K": int(K), "knn": int(knn),
            "go_knn_reconciliation_max_delta": drift,
            "n_queries": int(Qu.shape[0]), "predictors": res,
            "prior_names": prior_names,
            "learned_weights_mean": {n: float(Wts[:, i].mean()) for i, n in enumerate(prior_names)},
            "learned_weights_all_mean": {n: float(Wts_all[:, i].mean())
                                         for i, n in enumerate(all_names)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--knn", type=int, default=10)
    a = ap.parse_args()

    out_dir = os.path.join(a.art_dir, "repr_predict")
    os.makedirs(out_dir, exist_ok=True)
    check = json.load(open(os.path.join(a.art_dir, "rl_acquire", "cache_check.json")))
    rel_dir = os.path.join(a.art_dir, "rl_acquire", "related")

    summary = {"knn": a.knn, "datasets": {}, "errors": []}
    for ds in a.datasets:
        try:
            t0 = time.time()
            pp = os.path.join(rel_dir, f"priors_{ds}.npz")
            if not os.path.exists(pp):
                raise FileNotFoundError(f"no aligned priors: {pp} (run ./submit.sh related_fetch)")
            s = run(ds, check["datasets"][ds]["path"], pp, a.gears_dir, out_dir, a.knn)
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

    L = ["# Unseen-gene retrieval under different gene representations", "",
         f"Same task, same predictor (k={a.knn} weighted average of atlas signatures); only the",
         "similarity that picks the neighbours changes. Rows marked * use a measurement of",
         "perturbation effect (another cell line, or the held-out gene itself) and are not",
         "prior knowledge.", ""]
    for ds, s in summary["datasets"].items():
        L += [f"## {ds} (K={s['K']}, {s['n_queries']} queries)", "",
              "| representation | top-1 | top-10 | MRR | median rank | Spearman vs signature | recall@k |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for k, v in s["predictors"].items():
            m, o = v["all_queries"], v["ordering"]
            star = "*" if v["is_measurement_or_oracle"] else ""
            L.append(f"| {k}{star} | {m['top1']*100:.1f}% | {m['top10']*100:.1f}% | {m['mrr']:.3f} "
                     f"| {m['median_rank']:.0f} | "
                     + (f"{o['spearman_vs_signature']:.3f} | {o['recall_at_k']*100:.1f}% |"
                        if o else " — | — |"))
        L += ["", "Learned weights (mean over folds): "
              + ", ".join(f"`{k}`={v:.3f}" for k, v in s["learned_weights_mean"].items()), ""]
    if summary["errors"]:
        L += ["## Errors", ""] + [f"- {e['dataset']}: {e['error']}" for e in summary["errors"]]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    log("wrote", os.path.join(out_dir, "report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
