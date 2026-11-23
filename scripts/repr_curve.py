#!/usr/bin/env python
"""How good does a gene representation have to be before it is worth anything?

`repr_select.py` leaves the project with a verdict but not a target. The combiner is
worth 3.5x given a perfect similarity (12.5% -> 44.4% top-1) and nothing given GO
(2.9% -> 3.3%). "Get a better representation" is not actionable; *how much* better is.

This sweeps similarity QUALITY continuously and reads off retrieval at each level. The
sweep is a controlled corruption of the true signature similarity,

    S_alpha = alpha * z(C_sig) + sqrt(1 - alpha^2) * z(E),   E symmetric Gaussian noise,

so alpha traces a path from pure noise to the oracle without ever changing the task, the
predictor or the metric. For each alpha we record the similarity's ACHIEVED ordering
quality (Spearman against true signature cosine, recall@10) and the retrieval it produces
under both combiners. Plotting retrieval against achieved Spearman gives a transfer
function, and the real representations -- GO Jaccard, the learned combination, the other
cell line's Perturb-seq -- are placed on it as reference points.

What this is for: it converts the negative into a specification. A future gene
representation is worth building only if it clears the Spearman at which the curve leaves
the floor, and the curve says what retrieval it would buy.

Measured correction to the obvious reading (2026-11-22). Indexed by Spearman the curve is
a PESSIMISTIC bound, not an optimistic one. At alpha = 0.15 the corrupted oracle reaches
Spearman 0.132 -- essentially GO Jaccard's 0.123 -- yet only 0.30% top-1 against GO's
2.87%. The reason is in the recall column: at that Spearman the corruption recovers 2.5%
of a gene's true top-10 neighbours while GO recovers 14.3%. Isotropic noise spreads its
error over the whole ordering; GO is wrong almost everywhere but concentrates what
accuracy it has at the TOP of the list, which is the only part a kNN or a ridge solve
reads. So global rank correlation is the wrong summary of a gene representation, and
recall@k is the axis these curves should be read on. That is itself the actionable
finding: a representation should be trained on a top-k ranking objective, not fitted to
reproduce the whole similarity matrix.

Writes <art>/repr_curve/{curve_<dataset>.npz, summary.json, report.md}. Exits 0.

Usage: python -I repr_curve.py --art-dir D --gears-dir D --datasets A B [--knn 10]
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
import repr_select as RS          # noqa: E402

SEED = 0
N_FOLDS = 5
ALPHAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0)
M_GLS = 100
LAM_GRID = (1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def zsym(M, rng=None, K=None):
    """Standardise the off-diagonal entries of a symmetric matrix to mean 0, sd 1."""
    if M is None:
        E = rng.standard_normal((K, K)).astype(np.float32)
        M = (E + E.T) / np.sqrt(2.0)
    iu = np.triu_indices(M.shape[0], 1)
    v = M[iu]
    out = (M - v.mean()) / max(v.std(), 1e-12)
    return out.astype(np.float32)


def run(ds, cache_path, priors_path, gears_dir, out_dir, knn):
    W = RA.World(cache_path)
    K = W.K
    rng = np.random.default_rng(SEED)
    log(f"{ds}: K={K}, knn={knn}, {len(ALPHAS)} quality levels")

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
    G = YA.astype(np.float64) @ YA.astype(np.float64).T
    nrm = np.sqrt(np.maximum(np.diag(G), 1e-24))

    Qu = RH.unit(W.Q)
    own = W.owner.astype(np.int64)
    folds = RH.folds_of(K, np.random.default_rng(SEED), k=N_FOLDS)
    val_fold = {f: (f + 1) % N_FOLDS for f in range(N_FOLDS)}

    Cz = zsym(Csig)
    Ez = zsym(None, rng, K)

    def evaluate(S, tag):
        S = S.copy()
        np.fill_diagonal(S, -np.inf)
        o = RD.ordering_quality(np.where(np.isfinite(S), S, 0.0), Csig, K, knn)
        row = {"spearman": o["spearman_vs_signature"], "recall_at_k": o["recall_at_k"]}
        Pk = RP.knn_from_sim(S, YA, k=knn)
        m, _, _ = RP.score_matrix((Qu @ RH.unit(Pk) .T).astype(np.float32), own)
        row["knn_avg"] = m
        Pg, picked = RS.tune(lambda l: RS.predict_gls(S, YA, G, M_GLS, l, nrm),
                             list(LAM_GRID), YA, Qu, own, K, folds, val_fold)
        m2, _, _ = RP.score_matrix((Qu @ RH.unit(Pg).T).astype(np.float32), own)
        row["gls"] = m2
        row["gls_lambda"] = sorted({str(v) for v in picked.values()})
        log(f"  {tag:26s} spearman={row['spearman']:+.3f} recall@{knn}={row['recall_at_k']*100:5.1f}%"
            f"   knn_avg top1={m['top1']*100:5.2f}%   gls top1={m2['top1']*100:5.2f}%")
        return row

    curve = {}
    for a in ALPHAS:
        S = (a * Cz + np.sqrt(max(1.0 - a * a, 0.0)) * Ez).astype(np.float32)
        curve[f"alpha={a:g}"] = evaluate(S, f"alpha={a:g}")

    # ---- the real representations, placed on the same axes
    real = {}
    Jg = W.JJ.copy()
    np.fill_diagonal(Jg, 0.0)
    cands = {"go_jaccard": Jg.astype(np.float32)}
    rich = W.rich()
    views = dict(cands)
    views["go_idf"], views["go_emb"] = rich["idf"].astype(np.float32), rich["emb"].astype(np.float32)
    z = np.load(priors_path, allow_pickle=False)
    pnames = [str(x) for x in z["names"]]
    prior_names, meas_names = [], []
    for i, nm in enumerate(pnames):
        v = RD.cosnorm(z["kernels"][i].astype(np.float64))
        np.fill_diagonal(v, 0.0)
        views[nm] = v
        (meas_names if nm in RD.MEASUREMENT_KERNELS else prior_names).append(nm)
    names_p = ["go_jaccard", "go_idf", "go_emb"] + prior_names
    Slearn, _ = RD.learned_similarity([views[n] for n in names_p], Csig, folds, K,
                                      np.random.default_rng(SEED))
    np.fill_diagonal(Slearn, 0.0)
    cands["learned_priors"] = Slearn
    if meas_names:
        names_x = names_p + meas_names
        Sx, _ = RD.learned_similarity([views[n] for n in names_x], Csig, folds, K,
                                      np.random.default_rng(SEED))
        np.fill_diagonal(Sx, 0.0)
        cands[f"xline({meas_names[0]})"] = views[meas_names[0]].copy()
        cands["learned_with_xline"] = Sx
    for nm, S in cands.items():
        real[nm] = evaluate(S, nm)

    np.savez_compressed(
        os.path.join(out_dir, f"curve_{ds}.npz"),
        alphas=np.array(ALPHAS, np.float32),
        spearman=np.array([curve[f"alpha={a:g}"]["spearman"] for a in ALPHAS], np.float32),
        recall=np.array([curve[f"alpha={a:g}"]["recall_at_k"] for a in ALPHAS], np.float32),
        knn_top1=np.array([curve[f"alpha={a:g}"]["knn_avg"]["top1"] for a in ALPHAS], np.float32),
        gls_top1=np.array([curve[f"alpha={a:g}"]["gls"]["top1"] for a in ALPHAS], np.float32),
        knn_mrr=np.array([curve[f"alpha={a:g}"]["knn_avg"]["mrr"] for a in ALPHAS], np.float32),
        gls_mrr=np.array([curve[f"alpha={a:g}"]["gls"]["mrr"] for a in ALPHAS], np.float32),
        real_names=np.array(list(real.keys())),
        real_spearman=np.array([v["spearman"] for v in real.values()], np.float32),
        real_recall=np.array([v["recall_at_k"] for v in real.values()], np.float32),
        real_knn_top1=np.array([v["knn_avg"]["top1"] for v in real.values()], np.float32),
        real_gls_top1=np.array([v["gls"]["top1"] for v in real.values()], np.float32),
    )
    return {"dataset": ds, "K": int(K), "knn": int(knn), "curve": curve, "real": real}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--knn", type=int, default=10)
    a = ap.parse_args()

    out_dir = os.path.join(a.art_dir, "repr_curve")
    os.makedirs(out_dir, exist_ok=True)
    check = json.load(open(os.path.join(a.art_dir, "rl_acquire", "cache_check.json")))
    rel = os.path.join(a.art_dir, "rl_acquire", "related")

    summary = {"knn": a.knn, "alphas": list(ALPHAS), "datasets": {}, "errors": []}
    for ds in a.datasets:
        try:
            t0 = time.time()
            s = run(ds, check["datasets"][ds]["path"], os.path.join(rel, f"priors_{ds}.npz"),
                    a.gears_dir, out_dir, a.knn)
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

    L = ["# How good does a gene representation have to be?", "",
         "A controlled corruption of the true signature similarity traces retrieval against",
         "ordering quality, holding the task, the predictor and the metric fixed. The real",
         "representations are placed on the same axes. The corruption is isotropic while a real",
         "representation fails structurally, so the curve is an optimistic bound on what a given",
         "Spearman buys.", ""]
    for ds, s in summary["datasets"].items():
        L += [f"## {ds} (K={s['K']})", "",
              "| similarity | Spearman vs signature | recall@10 | top-1, knn_avg | top-1, gls |",
              "|---|---:|---:|---:|---:|"]
        for k, v in s["curve"].items():
            L.append(f"| {k} | {v['spearman']:+.3f} | {v['recall_at_k']*100:.1f}% "
                     f"| {v['knn_avg']['top1']*100:.2f}% | {v['gls']['top1']*100:.2f}% |")
        for k, v in s["real"].items():
            L.append(f"| **{k}** | **{v['spearman']:+.3f}** | **{v['recall_at_k']*100:.1f}%** "
                     f"| **{v['knn_avg']['top1']*100:.2f}%** | **{v['gls']['top1']*100:.2f}%** |")
        L.append("")
    if summary["errors"]:
        L += ["## Errors", ""] + [f"- {e['dataset']}: {e['error']}" for e in summary["errors"]]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    log("wrote", os.path.join(out_dir, "report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
