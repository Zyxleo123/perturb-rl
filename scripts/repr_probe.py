#!/usr/bin/env python
"""Why does the unseen-gene predictor fail, and is the failure attackable?

GAPS gap 0 measures one number per predictor (0.8-1.9% top-1 against a 32.5-88.9%
ceiling) and diagnoses it as "fidelity without discrimination". That diagnosis names
the symptom, not the cause, and it does not say whether anything can be done. This
script splits the failure into causes that have different consequences:

  (a) the PREDICTOR CLASS is wrong -- no weighted average of measured signatures can
      name an unseen gene, in which case better gene features are pointless;
  (b) the SIMILARITY is wrong -- a weighted average CAN name it, but GO Jaccard picks
      the wrong genes to average, in which case the gap is a metric-learning problem
      and we can measure exactly how much is on the table;
  (c) the SCORING GEOMETRY is wrong -- the right prediction is made but loses on a
      hubness-style artefact of cosine retrieval in high dimensions.

The task is `story_cases.case_gap`'s, number for number, so results compare directly
with GAPS: every gene is held out in turn, its signature predicted from the OTHER
genes' half-A signatures, and its held-out query ranked against all K predictions.

Predictors scored on that identical task:
  go_knn        GO-Jaccard kNN, k=10                      -- what GAPS reports
  mean          the mean effect (no gene information)     -- the constant floor
  oracle_sim    kNN whose similarity IS the true signature cosine between the held-out
                gene and each atlas gene. A gene-gene metric learned from priors can at
                best recover this ordering, so it is the CEILING OF PREDICTOR CLASS (b).
  oracle_greedy greedily picks the 10 atlas genes whose mean best reconstructs the held
                out signature: the ceiling of the averaging class under any weighting.
  measured      the gene's own measured half-A signature   -- the atlas ceiling

and two corrections applied to go_knn's scores alone, which add no information:
  centred       column-centre the score matrix (GAPS Framing A's one win)
  csls          Cross-domain Similarity Local Scaling, the standard hubness correction
                for retrieval between two embedding spaces (Conneau et al., ICLR 2018)

Diagnostics written alongside:
  * hubness of the predicted space (N10 counts and their skew) vs the measured space
  * how far GO Jaccard is from the true signature ordering (Spearman, recall@10,
    GO rank of the true nearest neighbour)
  * everything stratified by the gene's own split-half reliability and effect size,
    so "the gap" can be separated from "the gene was never measurable"
  * each gene's best available neighbour, to show whether oracle_sim only works where
    a near-duplicate (complex partner / paralog) happens to be in the atlas

Writes <art>/repr_probe/{probe_<dataset>.npz, summary.json, report.md}.
Always exits 0 after writing what it could, so dependents are never stranded.

Usage: python -I repr_probe.py --art-dir D --gears-dir D --datasets A B
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

KNN = 10
HUB_K = 10


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ===================================================================== scoring
def tie_rank(S, own):
    """story_cases.tie_rank: rl_headroom.score's tie-aware rank and reciprocal rank."""
    S = np.round(S, 6)
    t = S[np.arange(S.shape[0]), own]
    g = (S > t[:, None]).sum(1)
    e = (S == t[:, None]).sum(1) - 1
    H = RH.harmonic(S.shape[1])
    return g + 1.0 + e / 2.0, (H[g + e + 1] - H[g]) / (e + 1)


def score_matrix(S, own):
    """Per-query rank/rr -> the summary GAPS reports."""
    rank, rr = tie_rank(S, own)
    return {"mrr": float(rr.mean()), "top1": float((rank <= 1).mean()),
            "top5": float((rank <= 5).mean()), "top10": float((rank <= 10).mean()),
            "median_rank": float(np.median(rank))}, rank, rr


def per_gene(rank, rr, own, K):
    """Average a per-query vector down to one value per gene."""
    cnt = np.maximum(np.bincount(own, minlength=K), 1)
    return (np.bincount(own, weights=rank, minlength=K) / cnt,
            np.bincount(own, weights=rr, minlength=K) / cnt)


# ================================================================= predictors
def knn_from_sim(Sim, YA, k=KNN):
    """Weighted average of the k most similar atlas signatures, by a given similarity.

    Identical in form to rl_headroom.go_knn_predict; only the similarity differs, so a
    difference in the result is a difference in the REPRESENTATION, not the predictor.
    Negative similarities are clipped to 0 (a weighted average, not a contrast)."""
    K = Sim.shape[0]
    base = YA.mean(0)
    out = np.repeat(base[None], K, 0).astype(np.float32)
    kk = min(k, K - 1)
    top = np.argpartition(-Sim, kk - 1, axis=1)[:, :kk]
    for i in range(K):
        w = np.maximum(Sim[i, top[i]], 0.0)
        if w.sum() > 0:
            out[i] = (w / w.sum()) @ YA[top[i]]
    return out


def greedy_reconstruct(G, YA, k=KNN):
    """For each gene i, greedily pick k other genes whose UNIFORM mean maximises
    cos(mean, YA_i). Pure Gram algebra: with G = YA YA^T,
        cos(sum_{j in S} YA_j, YA_i) = (sum_j G[j,i]) / sqrt(sum_{j,l in S} G[j,l]).
    This is the ceiling of the averaging class -- the best 10 atlas genes, chosen with
    full knowledge of the answer."""
    K = G.shape[0]
    sel = np.zeros((K, k), np.int64)
    for i in range(K):
        num = np.zeros(K)            # sum_j G[j,i] if j added
        den = np.zeros(K)            # sum_{j,l in S} G[j,l] if j added
        chosen = []
        cur_num, cur_den = 0.0, 0.0
        cross = np.zeros(K)          # sum_{l in S} G[j,l]
        for step in range(k):
            num = cur_num + G[:, i]
            den = cur_den + 2.0 * cross + np.diag(G)
            val = num / np.sqrt(np.maximum(den, 1e-24))
            val[i] = -np.inf
            if chosen:
                val[np.array(chosen)] = -np.inf
            j = int(np.argmax(val))
            chosen.append(j)
            cur_num = num[j]
            cur_den = den[j]
            cross = cross + G[:, j]
            sel[i, step] = j
    P = np.stack([YA[sel[i]].mean(0) for i in range(K)]).astype(np.float32)
    return P, sel


# ================================================================ corrections
def csls(S, k=HUB_K):
    """Cross-domain Similarity Local Scaling (Conneau et al., ICLR 2018), the standard
    fix for hubness in retrieval between two embedding spaces: penalise a candidate by
    how similar it is to everything, and a query likewise. No information added."""
    kq = min(k, S.shape[1] - 1)
    kc = min(k, S.shape[0] - 1)
    rq = np.sort(S, axis=1)[:, -kq:].mean(1)
    rc = np.sort(S, axis=0)[-kc:, :].mean(0)
    return 2.0 * S - rq[:, None] - rc[None, :]


def hubness(S, k=HUB_K):
    """N_k: how often each candidate appears in a query's top k. A heavy right skew is
    the signature of hub candidates absorbing queries they do not own."""
    kk = min(k, S.shape[1])
    top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    n = np.bincount(top.ravel(), minlength=S.shape[1]).astype(np.float64)
    m, sd = n.mean(), n.std()
    skew = float(((n - m) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    return n, skew


# ============================================================== rank helpers
def rank_of_true_neighbour(Sim, Csig):
    """Where GO ranks each gene's true nearest-signature neighbour, and how many of the
    true top-10 neighbours GO's top-10 recovers."""
    K = Sim.shape[0]
    sim = Sim.copy()
    np.fill_diagonal(sim, -np.inf)
    cs = Csig.copy()
    np.fill_diagonal(cs, -np.inf)
    true1 = np.argmax(cs, axis=1)
    # rank of true1 under Sim, ties counted at their midpoint
    t = sim[np.arange(K), true1]
    gt = (sim > t[:, None]).sum(1)
    eq = (sim == t[:, None]).sum(1) - 1
    rank1 = gt + 1.0 + eq / 2.0
    kk = min(KNN, K - 1)
    go_top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    tr_top = np.argpartition(-cs, kk - 1, axis=1)[:, :kk]
    rec = np.array([len(set(go_top[i].tolist()) & set(tr_top[i].tolist())) / kk for i in range(K)])
    return rank1, rec, true1, cs[np.arange(K), true1]


def spearman(a, b):
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def strat(vals, by, qs=(0.25, 0.5, 0.75)):
    """Median of `vals` inside quartiles of `by`."""
    edges = np.quantile(by, qs)
    g = np.digitize(by, edges)
    return [{"q": int(i), "n": int((g == i).sum()),
             "by_median": float(np.median(by[g == i])),
             "val_median": float(np.median(vals[g == i]))} for i in range(4)]


# ===================================================================== driver
def probe(ds, cache_path, gears_dir, out_dir, g2g_path):
    W = RA.World(cache_path)
    K = W.K
    log(f"{ds}: K={K} candidates, {W.Y.shape[1]} readout genes, {W.F.shape[1]} GO terms")

    D = RH.build(ds, os.path.join(gears_dir, ds, "perturb_processed.h5ad"),
                 np.random.default_rng(RH.SEED))
    mk = D["mask"]
    same = (list(D["cands"]) == [str(x) for x in W.cands]
            and np.allclose(D["dF"][:, mk], W.Y, atol=1e-6))
    if not same:
        raise RuntimeError("rl_headroom.build no longer reproduces the cache; refusing to mix")
    log(f"  rebuild matches cache: {same}")
    YA = D["dA"][:, mk].astype(np.float32)          # leak-free: half A, no query cells
    YB = D["dB"][:, mk].astype(np.float32)
    del D

    # ---- queries: every held-out query, owned by its gene (GAPS averages per gene)
    Qu = RH.unit(W.Q)
    own = W.owner.astype(np.int64)
    # first-query subset, to reconcile exactly with story.json's numbers
    q1 = np.array([W.q_of[i][0] for i in range(K)])
    Qu1, own1 = RH.unit(W.Q[q1]), np.arange(K)

    YAu = RH.unit(YA)
    Csig = (YAu @ YAu.T).astype(np.float32)         # true gene-gene signature cosine
    Gram = (YA.astype(np.float64) @ YA.astype(np.float64).T)
    J = W.JJ.copy()
    np.fill_diagonal(J, 0.0)                        # a gene never predicts itself
    Cs = Csig.copy()
    np.fill_diagonal(Cs, -np.inf)

    # ---------------------------------------------------------- predictors
    preds = {}
    preds["mean"] = np.repeat(YA.mean(0)[None], K, 0).astype(np.float32)
    preds["go_knn"] = RH.go_knn_predict(J, YA, YA.mean(0))
    preds["oracle_sim"] = knn_from_sim(Cs, YA)
    Pg, sel = greedy_reconstruct(Gram, YA)
    preds["oracle_greedy"] = Pg
    preds["measured"] = YA

    res, rank_g, rr_g = {}, {}, {}
    for name, P in preds.items():
        S = (Qu @ RH.unit(P).T).astype(np.float32)
        m, rank, rr = score_matrix(S, own)
        rg, rrg = per_gene(rank, rr, own, K)
        rank_g[name], rr_g[name] = rg, rrg
        m1, _, _ = score_matrix((Qu1 @ RH.unit(P).T).astype(np.float32), own1)
        res[name] = {"all_queries": m, "first_query": m1}
        log(f"  {name:14s} top1={m['top1']:.4f} mrr={m['mrr']:.4f} "
            f"medrank={m['median_rank']:.1f}   (first-query top1={m1['top1']:.4f})")

    # ------------------------------------------- scoring-geometry corrections
    Sgo = (Qu @ RH.unit(preds["go_knn"]).T).astype(np.float32)
    geom = {}
    for name, Sx in (("centred", Sgo - Sgo.mean(0, keepdims=True)), ("csls", csls(Sgo))):
        m, rank, rr = score_matrix(Sx, own)
        geom[name] = m
        log(f"  {name:14s} top1={m['top1']:.4f} mrr={m['mrr']:.4f} medrank={m['median_rank']:.1f}")

    # ------------------------------------------------------------- hubness
    n10_pred, skew_pred = hubness(Sgo)
    Smeas = (Qu @ RH.unit(YA).T).astype(np.float32)
    n10_meas, skew_meas = hubness(Smeas)
    log(f"  hubness N10 skew: predicted {skew_pred:.2f}  measured {skew_meas:.2f} "
        f"(max N10 pred {int(n10_pred.max())} of {Sgo.shape[0]} queries)")

    # -------------------------------------------- how wrong is GO's ordering
    rank1, rec10, true1, best_cos = rank_of_true_neighbour(J, Csig)
    iu = np.triu_indices(K, 1)
    sub = np.random.default_rng(0).choice(iu[0].size, size=min(400_000, iu[0].size), replace=False)
    sp = spearman(J[iu[0][sub], iu[1][sub]], Csig[iu[0][sub], iu[1][sub]])
    log(f"  GO vs signature: Spearman {sp:.3f} over pairs; median GO rank of the true "
        f"nearest neighbour {np.median(rank1):.0f}/{K}; recall@10 {rec10.mean():.3f}")

    # ------------------------------------------------- reliability / strength
    rel = np.einsum("ij,ij->i", YAu, RH.unit(YB)).astype(np.float64)   # split-half cosine
    strength = np.linalg.norm(YA, axis=1)
    fid_go = np.einsum("ij,ij->i", RH.unit(preds["go_knn"]), YAu)
    fid_or = np.einsum("ij,ij->i", RH.unit(preds["oracle_sim"]), YAu)

    summ = {
        "dataset": ds, "K": int(K), "n_queries": int(Qu.shape[0]),
        "readout_genes": int(W.Y.shape[1]), "go_terms": int(W.F.shape[1]),
        "predictors": res, "geometry": geom,
        "hubness": {"skew_predicted": skew_pred, "skew_measured": skew_meas,
                    "max_N10_predicted": int(n10_pred.max()),
                    "max_N10_measured": int(n10_meas.max()),
                    "n_queries": int(Sgo.shape[0])},
        "go_vs_signature": {"spearman_pairs": sp,
                            "median_go_rank_of_true_nn": float(np.median(rank1)),
                            "recall_at_10": float(rec10.mean()),
                            "median_best_neighbour_cos": float(np.median(best_cos))},
        "fidelity": {"go_knn_median": float(np.median(fid_go)),
                     "oracle_sim_median": float(np.median(fid_or)),
                     "split_half_median": float(np.median(rel))},
        "stratified": {
            "go_knn_rank_by_reliability": strat(rank_g["go_knn"], rel),
            "oracle_sim_rank_by_reliability": strat(rank_g["oracle_sim"], rel),
            "measured_rank_by_reliability": strat(rank_g["measured"], rel),
            "go_knn_rank_by_best_neighbour_cos": strat(rank_g["go_knn"], best_cos),
            "oracle_sim_rank_by_best_neighbour_cos": strat(rank_g["oracle_sim"], best_cos),
        },
    }

    np.savez_compressed(
        os.path.join(out_dir, f"probe_{ds}.npz"),
        cands=np.array([str(x) for x in W.cands]),
        reliability=rel.astype(np.float32), strength=strength.astype(np.float32),
        best_neighbour=true1.astype(np.int32), best_neighbour_cos=best_cos.astype(np.float32),
        go_rank_of_true_nn=rank1.astype(np.float32), recall_at_10=rec10.astype(np.float32),
        fid_go=fid_go.astype(np.float32), fid_oracle=fid_or.astype(np.float32),
        n10_predicted=n10_pred.astype(np.int32), n10_measured=n10_meas.astype(np.int32),
        oracle_greedy_sel=sel.astype(np.int32),
        **{f"rank_{k}": v.astype(np.float32) for k, v in rank_g.items()},
        **{f"rr_{k}": v.astype(np.float32) for k, v in rr_g.items()},
    )
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    a = ap.parse_args()

    out_dir = os.path.join(a.art_dir, "repr_probe")
    os.makedirs(out_dir, exist_ok=True)
    check = json.load(open(os.path.join(a.art_dir, "rl_acquire", "cache_check.json")))

    summary = {"datasets": {}, "errors": []}
    for ds in a.datasets:
        try:
            t0 = time.time()
            s = probe(ds, check["datasets"][ds]["path"], a.gears_dir, out_dir, None)
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

    # ------------------------------------------------------------- report.md
    L = ["# Why the unseen-gene predictor fails", ""]
    for ds, s in summary["datasets"].items():
        L += [f"## {ds} (K={s['K']}, {s['n_queries']} queries)", "",
              "| predictor | top-1 | top-10 | MRR | median rank |", "|---|---:|---:|---:|---:|"]
        for k, v in s["predictors"].items():
            m = v["all_queries"]
            L.append(f"| {k} | {m['top1']*100:.1f}% | {m['top10']*100:.1f}% | "
                     f"{m['mrr']:.3f} | {m['median_rank']:.0f} |")
        for k, m in s["geometry"].items():
            L.append(f"| go_knn + {k} | {m['top1']*100:.1f}% | {m['top10']*100:.1f}% | "
                     f"{m['mrr']:.3f} | {m['median_rank']:.0f} |")
        g = s["go_vs_signature"]
        h = s["hubness"]
        L += ["", f"GO vs signature: Spearman **{g['spearman_pairs']:.3f}** over gene pairs; "
                  f"the true nearest neighbour sits at GO rank **{g['median_go_rank_of_true_nn']:.0f}** "
                  f"of {s['K']}; GO's top-10 recovers **{g['recall_at_10']*100:.1f}%** of the true "
                  f"top-10. Median best available neighbour cosine **{g['median_best_neighbour_cos']:.3f}**.",
              "", f"Hubness N10 skew: predicted **{h['skew_predicted']:.2f}** vs measured "
                  f"**{h['skew_measured']:.2f}**; the biggest hub takes "
                  f"{h['max_N10_predicted']}/{h['n_queries']} queries.", ""]
    if summary["errors"]:
        L += ["## Errors", ""] + [f"- {e['dataset']}: {e['error']}" for e in summary["errors"]]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    log("wrote", os.path.join(out_dir, "report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
