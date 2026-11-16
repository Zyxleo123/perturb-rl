#!/usr/bin/env python
"""Is there an RL-shaped problem inside the unseen-gene gap?

GAPS.md gap 0 measured the gap: given a gene never perturbed here, every cheap
predictor ranks it ~1% top-1 against a 32-89% ceiling, while predicting the
EFFECT at ~70% of the noise ceiling. Fidelity is fine; discrimination is at the
floor. Two different things could cause that, and they imply two different RL
formulations -- or none. This script separates them.

PART 1 -- "is the signal there but mis-scored?"  (objective headroom)
  A predictor trained to minimise squared error spends its capacity on the
  component every perturbation shares. If the gene-specific part survives in the
  residual, then simply CHANGING WHAT WE OPTIMISE recovers it, and the honest
  framing is a ranking objective. We test the cheapest possible version of that:
  centre the predictions across candidates, then whiten them in a subspace fitted
  on TRAINING genes only, and re-rank. No new information enters -- only the
  scoring geometry changes. If top-1 moves a lot, the information was already
  there and an objective change is the lever (RL with a rank reward, or its
  differentiable cousin, contrastive learning -- which is then the baseline to
  beat, not a detail).

PART 2 -- "does it matter WHICH genes were measured?"  (acquisition headroom)
  The atlas is a choice, not a given. If held-out discrimination depends strongly
  on which genes happen to be in the training set, then choosing them is a
  sequential decision problem with a budget -- genuine RL, and simulatable
  offline against real measurements. If it does not, no acquisition policy can
  help and that framing is dead.

  Measured honestly: subsets are SELECTED on one held-out gene set and REPORTED on
  a disjoint one, so "best-of-N" is transferable gain, not overfitting. Heuristics
  use only prior information (GO), never the unmeasured signatures.

Every predictor here is out-of-fold and every number is in the masked gene space
(every candidate's own target gene removed), because a design target is a
phenotype. Writes <out>/rl_headroom/{objective.jsonl,acquisition.jsonl,summary.json}.
Always exits 0 after writing what it has.

Usage: python -I rl_headroom.py --gears-dir D --gene2go F --out-dir D
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import sys
import time
import traceback

import numpy as np

POOL = 32
N_DRAWS = 2
MIN_CELLS = 12
N_FOLDS = 5
SEED = 0
TOPKS = (1, 5, 10)
BUDGETS = (25, 50, 100, 200, 400)
N_SUBSETS = 200
WHITEN_R = (5, 10, 25, 50, 100, 200)
DATASETS = ("replogle_k562_essential", "replogle_rpe1_essential")


def log(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.strftime('%H:%M:%S')} rss={rss:.1f}GB]", *a, flush=True)


def densify(x):
    import scipy.sparse as sp
    return np.asarray(x.todense(), dtype=np.float32) if sp.issparse(x) else np.asarray(x, dtype=np.float32)


def canon(c):
    p = sorted(g for g in str(c).split("+") if g != "ctrl")
    return "+".join(p) if p else "ctrl"


def unit(M):
    M = np.atleast_2d(np.asarray(M, dtype=np.float32))
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return np.where(n > 0, M / np.maximum(n, 1e-12), 0.0).astype(np.float32)


class _NoGlobals(pickle.Unpickler):
    """Downloaded pickle: a dict of sets of strings needs no globals, so refuse
    every one. Nothing in the file can then import or call anything."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def load_gene2go(path):
    with open(path, "rb") as f:
        d = _NoGlobals(f).load()
    return {str(k): frozenset(map(str, v)) for k, v in d.items()}


_H = None


def harmonic(N):
    global _H
    if _H is None or _H.size < N + 2:
        _H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, N + 2))])
    return _H


def score(P, Q, owner):
    """Tie-aware macro-averaged retrieval: a predictor giving many candidates the
    same score gets the expected rank under random tie-breaking, so a constant
    predictor lands exactly on chance instead of being flattered by argsort order."""
    K = P.shape[0]
    S = np.round(unit(Q) @ unit(P).T, 6)
    t = S[np.arange(S.shape[0]), owner]
    g = (S > t[:, None]).sum(1)
    e = (S == t[:, None]).sum(1) - 1
    H = harmonic(K)
    per = {f"top{k}": np.clip((k - g) / (e + 1), 0, 1) for k in TOPKS}
    per["mrr"] = (H[g + e + 1] - H[g]) / (e + 1)
    per["rank"] = g + 1 + e / 2.0
    cnt = np.bincount(owner, minlength=K)
    m = cnt > 0
    out = {k: np.bincount(owner, weights=v, minlength=K)[m] / cnt[m] for k, v in per.items()}
    r = {f"top{k}": float(out[f"top{k}"].mean()) for k in TOPKS}
    r["mrr"] = float(out["mrr"].mean())
    r["median_rank"] = float(np.median(out["rank"]))
    r["K"] = int(K)
    r["chance_top1"] = 1.0 / K
    r["chance_mrr"] = float(H[K] / K)
    return r


# ================================================================ data
def build(name, h5, rng):
    """Single-gene candidates only, with an honest split: half the cells build the
    candidate signature, the other half supplies queries, controls split too."""
    import anndata as ad
    import pandas as pd

    log(f"loading {name}")
    adata = ad.read_h5ad(h5)
    X = adata.X
    if hasattr(X, "tocsr"):
        X = X.tocsr()
    genes = np.asarray(adata.var["gene_name"].astype(str).values if "gene_name" in adata.var
                       else adata.var_names.astype(str), dtype=object)
    conds = np.asarray([canon(c) for c in adata.obs["condition"].astype(str).values], dtype=object)
    codes, uniq = pd.factorize(conds)
    order = np.argsort(codes, kind="stable")
    b = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    rows_of = {uniq[i]: order[b[i]:b[i + 1]] for i in range(len(uniq))}

    cr = rows_of["ctrl"].copy()
    rng.shuffle(cr)
    Xc = densify(X[cr])
    h = cr.size // 2
    muA, muB, mu = Xc[:h].mean(0), Xc[h:].mean(0), Xc.mean(0)
    del Xc

    cands = sorted(c for c in rows_of
                   if c != "ctrl" and "+" not in c and rows_of[c].size >= MIN_CELLS)
    K, G = len(cands), len(genes)
    dA = np.zeros((K, G), np.float32)
    dB = np.zeros((K, G), np.float32)
    dF = np.zeros((K, G), np.float32)
    Q, owner = [], []
    for i, c in enumerate(cands):
        r = rows_of[c].copy()
        rng.shuffle(r)
        hh = r.size // 2
        XA, XB = densify(X[r[:hh]]), densify(X[r[hh:]])
        dA[i], dB[i] = XA.mean(0) - muA, XB.mean(0) - muB
        dF[i] = (XA.sum(0) + XB.sum(0)) / r.size - mu
        p = min(POOL, XB.shape[0])
        for _ in range(N_DRAWS if XB.shape[0] > POOL else 1):
            Q.append(XB[rng.choice(XB.shape[0], size=p, replace=False)].mean(0) - muB)
            owner.append(i)
    targets = set(cands)
    mask = np.array([g not in targets for g in genes])
    log(f"  K={K} single-gene candidates, {int(mask.sum())}/{G} genes after masking targets")
    del adata, X
    return {"name": name, "genes": genes, "cands": cands, "dA": dA, "dB": dB, "dF": dF,
            "Q": np.vstack(Q).astype(np.float32), "owner": np.asarray(owner), "mask": mask,
            "gidx": {g: j for j, g in enumerate(genes)}}


def folds_of(K, rng, k=N_FOLDS):
    p = rng.permutation(K)
    return [np.sort(p[i::k]) for i in range(k)]


# ================================================================ GO features
def go_features(cands, g2g, min_df=5, max_frac=0.5):
    """Binary GO-term matrix over candidate genes, terms of useful frequency only."""
    from collections import Counter
    cnt = Counter(t for c in cands for t in g2g.get(c, ()))
    n = len(cands)
    terms = sorted(t for t, k in cnt.items() if k >= min_df and k <= max_frac * n)
    tix = {t: j for j, t in enumerate(terms)}
    F = np.zeros((n, len(terms)), np.float32)
    for i, c in enumerate(cands):
        for t in g2g.get(c, ()):
            j = tix.get(t)
            if j is not None:
                F[i, j] = 1.0
    log(f"  GO features: {len(terms)} terms, {float((F.sum(1) > 0).mean()):.2f} of genes covered")
    return F


def ridge_fit_predict(F_tr, Y_tr, F_te, lam):
    """Ridge from GO indicators to signature; dual form, n_train << n_features."""
    fm, ym = F_tr.mean(0), Y_tr.mean(0)
    A = (F_tr - fm).astype(np.float64)
    Gm = A @ A.T
    al = np.linalg.solve(Gm + lam * np.trace(Gm) / max(1, A.shape[0]) * np.eye(A.shape[0]),
                         (Y_tr - ym).astype(np.float64))
    return (ym + (F_te - fm).astype(np.float64) @ (A.T @ al)).astype(np.float32)


def go_knn_predict(J_te_tr, Y_tr, base, k=10):
    """Weighted average of the k most GO-similar TRAINING signatures."""
    out = np.repeat(base[None], J_te_tr.shape[0], 0).astype(np.float32)
    kk = min(k, J_te_tr.shape[1])
    if kk == 0:
        return out
    top = np.argpartition(-J_te_tr, kk - 1, axis=1)[:, :kk]
    for i in range(J_te_tr.shape[0]):
        w = J_te_tr[i, top[i]]
        if w.sum() > 0:
            out[i] = (w / w.sum()) @ Y_tr[top[i]]
    return out


def jaccard(A, B):
    """Jaccard between rows of two binary matrices."""
    inter = A @ B.T
    a, b = A.sum(1)[:, None], B.sum(1)[None, :]
    return np.asarray(inter / np.maximum(a + b - inter, 1e-9), dtype=np.float32)


# ================================================================ part 1
def objective_headroom(D, F, rng, rows):
    """Does re-scoring the SAME predictions recover discrimination?

    centred  : subtract the mean training signature from predictions and from
               targets -- removes the component every perturbation shares.
    whitened : additionally project onto the top-r subspace of the TRAINING
               signatures and divide by each direction's singular value, so
               directions that barely vary between perturbations stop dominating
               the cosine.

    Both transforms are fitted on training folds only and applied identically to
    predictions and to the query. No new information enters: if top-1 moves, the
    gene-specific signal was already in the prediction and the squared-error
    objective was simply not scoring it."""
    K = len(D["cands"])
    mk = D["mask"]
    Y = D["dF"][:, mk]
    Qm = D["Q"][:, mk]
    log(f"[objective] {D['name']}: K={K}, {int(mk.sum())} genes after masking")

    F_folds = folds_of(K, rng)
    preds = {"go_ridge": np.zeros_like(Y), "go_knn": np.zeros_like(Y)}
    centred = {p: np.zeros_like(Y) for p in preds}
    whit = {(p, r): np.zeros_like(Y) for p in preds for r in WHITEN_R}
    for te in F_folds:
        tr = np.setdiff1d(np.arange(K), te)
        preds["go_ridge"][te] = ridge_fit_predict(F[tr], Y[tr], F[te], lam=1.0)
        preds["go_knn"][te] = go_knn_predict(jaccard(F[te], F[tr]), Y[tr], Y[tr].mean(0))
        mu = Y[tr].mean(0)
        _, sv, Vt = np.linalg.svd((Y[tr] - mu).astype(np.float64), full_matrices=False)
        for p in preds:
            centred[p][te] = preds[p][te] - mu
            for r in WHITEN_R:
                rr = min(r, Vt.shape[0])
                V = Vt[:rr].T.astype(np.float32)
                sd = np.maximum(sv[:rr], 1e-6).astype(np.float32)
                z = ((preds[p][te] - mu) @ V) / sd
                whit[(p, r)][te, :rr] = z            # store in the first rr columns
                whit[(p, r)][te, rr:] = 0.0

    # The query must be transformed the same way. A query's own candidate is held
    # out of its fold, but the transform itself is fitted on all other candidates,
    # so fitting the query transform on the full candidate set leaks nothing about
    # which candidate the query is.
    mu_all = Y.mean(0)
    _, sv_a, Vt_a = np.linalg.svd((Y - mu_all).astype(np.float64), full_matrices=False)
    cen_q = Qm - mu_all
    whq = {}
    for r in WHITEN_R:
        rr = min(r, Vt_a.shape[0])
        V = Vt_a[:rr].T.astype(np.float32)
        sd = np.maximum(sv_a[:rr], 1e-6).astype(np.float32)
        z = np.zeros((Qm.shape[0], Y.shape[1]), np.float32)
        z[:, :rr] = ((Qm - mu_all) @ V) / sd
        whq[r] = z

    ceil = score(D["dA"][:, mk], Qm, D["owner"])
    rows.append({"dataset": D["name"], "part": "objective", "predictor": "ceiling",
                 "scoring": "raw", **ceil})
    log(f"    {'ceiling':9s} {'raw':14s} top1={ceil['top1']:.4f} mrr={ceil['mrr']:.4f}")
    for p, P in preds.items():
        for sc, PP, QQ in (("raw", P, Qm), ("centred", centred[p], cen_q)):
            v = score(PP, QQ, D["owner"])
            rows.append({"dataset": D["name"], "part": "objective", "predictor": p,
                         "scoring": sc, **v})
            log(f"    {p:9s} {sc:14s} top1={v['top1']:.4f} mrr={v['mrr']:.4f} "
                f"(chance {v['chance_top1']:.4f})")
        for r in WHITEN_R:
            v = score(whit[(p, r)], whq[r], D["owner"])
            rows.append({"dataset": D["name"], "part": "objective", "predictor": p,
                         "scoring": f"whitened_r{r}", "whiten_r": int(r), **v})
            log(f"    {p:9s} {'whitened_r'+str(r):14s} top1={v['top1']:.4f} mrr={v['mrr']:.4f}")
    return ceil


# ================================================================ part 2
def acquisition_headroom(D, F, g2g, rng, rows, ceil):
    """Does it matter WHICH genes the atlas contains?

    pool -> the genes a policy may choose to measure
    sel  -> held-out genes used to CHOOSE a subset (the policy's training signal)
    rep  -> disjoint held-out genes the chosen subset is REPORTED on
    so best-of-N measures transferable acquisition gain, not overfitting."""
    K = len(D["cands"])
    mk = D["mask"]
    Y = D["dF"][:, mk]
    perm = rng.permutation(K)
    n_ev = max(60, K // 5)
    rep_i, sel_i, pool_i = np.sort(perm[:n_ev]), np.sort(perm[n_ev:2 * n_ev]), np.sort(perm[2 * n_ev:])
    log(f"[acquisition] {D['name']}: pool={pool_i.size} sel={sel_i.size} rep={rep_i.size}")

    def queries(idx):
        m = np.isin(D["owner"], idx)
        remap = {o: n for n, o in enumerate(idx)}
        return D["Q"][m][:, mk], np.array([remap[o] for o in D["owner"][m]])

    Q_sel, o_sel = queries(sel_i)
    Q_rep, o_rep = queries(rep_i)
    J_sel = jaccard(F[sel_i], F[pool_i])
    J_rep = jaccard(F[rep_i], F[pool_i])
    Y_pool = Y[pool_i]

    def evaluate(chosen):
        base = Y_pool[chosen].mean(0)
        ps = go_knn_predict(J_sel[:, chosen], Y_pool[chosen], base)
        pr = go_knn_predict(J_rep[:, chosen], Y_pool[chosen], base)
        return (score(ps, Q_sel, o_sel), score(pr, Q_rep, o_rep))

    # heuristics that use ONLY prior information (GO), never unmeasured signatures
    JPP = jaccard(F[pool_i], F[pool_i])
    np.fill_diagonal(JPP, 0.0)
    ndeg = F[pool_i].sum(1)

    def farthest_point(B):
        """GO-diversity: repeatedly add the pool gene most dissimilar to every gene
        already chosen (min-max on Jaccard similarity)."""
        chosen = [int(np.argmax(ndeg))]
        sim = JPP[chosen[0]].copy()          # similarity to the nearest chosen gene
        while len(chosen) < B:
            sim[chosen] = np.inf
            nxt = int(np.argmin(sim))
            chosen.append(nxt)
            sim = np.maximum(sim, JPP[nxt])
        return np.array(chosen)

    def coverage(B):
        """Greedy max-coverage of GO terms."""
        Fp = F[pool_i] > 0
        covered = np.zeros(Fp.shape[1], bool)
        chosen = []
        for _ in range(B):
            gain = (Fp & ~covered).sum(1).astype(float)
            gain[chosen] = -1
            i = int(np.argmax(gain))
            chosen.append(i)
            covered |= Fp[i]
        return np.array(chosen)

    def hubs(B):
        return np.argsort(-ndeg)[:B]

    for B in BUDGETS:
        if B > pool_i.size:
            continue
        t0 = time.time()
        srng = np.random.default_rng(1234 + B)
        rnd = []
        for _ in range(N_SUBSETS):
            ch = srng.choice(pool_i.size, size=B, replace=False)
            s, r = evaluate(ch)
            rnd.append((s["mrr"], r["mrr"], r["top1"]))
        rnd = np.array(rnd)
        best = int(np.argmax(rnd[:, 0]))          # chosen on SEL
        rows.append({
            "dataset": D["name"], "part": "acquisition", "strategy": "random", "budget": int(B),
            "n_subsets": N_SUBSETS, "pool": int(pool_i.size), "K_report": int(rep_i.size),
            "rep_mrr_median": float(np.median(rnd[:, 1])),
            "rep_mrr_p05": float(np.percentile(rnd[:, 1], 5)),
            "rep_mrr_p95": float(np.percentile(rnd[:, 1], 95)),
            "rep_mrr_max": float(rnd[:, 1].max()),
            "rep_top1_median": float(np.median(rnd[:, 2])),
            "rep_top1_max": float(rnd[:, 2].max()),
            "bestofN_on_sel_rep_mrr": float(rnd[best, 1]),
            "bestofN_on_sel_rep_top1": float(rnd[best, 2]),
            "ceiling_top1": ceil["top1"], "chance_top1": 1.0 / rep_i.size,
        })
        for nm, fn in (("go_diversity", farthest_point), ("go_coverage", coverage), ("go_hubs", hubs)):
            try:
                s, r = evaluate(fn(B))
                rows.append({"dataset": D["name"], "part": "acquisition", "strategy": nm,
                             "budget": int(B), "pool": int(pool_i.size),
                             "K_report": int(rep_i.size), "rep_mrr_median": r["mrr"],
                             "rep_top1_median": r["top1"], "sel_mrr": s["mrr"],
                             "ceiling_top1": ceil["top1"], "chance_top1": 1.0 / rep_i.size})
            except Exception as e:
                log(f"    {nm} failed: {type(e).__name__}: {e}")
        log(f"    B={B:4d} random rep mrr p05/med/p95/max = "
            f"{np.percentile(rnd[:,1],5):.4f}/{np.median(rnd[:,1]):.4f}/"
            f"{np.percentile(rnd[:,1],95):.4f}/{rnd[:,1].max():.4f}  "
            f"best-of-{N_SUBSETS} picked on sel -> rep {rnd[best,1]:.4f}  ({time.time()-t0:.0f}s)")



# ================================================================ part 3
def adaptivity_headroom(D, F, rng, rows, ceil):
    """Does it help to CHOOSE LATER GENES BASED ON WHAT EARLIER ONES SHOWED?

    Part 2 compared only NON-adaptive strategies: random subsets and GO heuristics
    both commit to the whole set up front, before seeing a single measurement. That
    is a static set-cover problem, and if it is all there is, RL has nothing to add.
    Adaptivity is the entire reason a problem is sequential, so it is the thing that
    decides whether this is an MDP or a one-shot combinatorial choice.

    Budget B is spent in R rounds. After each round the chosen genes' real
    signatures are revealed and may inform the next round.

      random / coverage   non-adaptive controls (round structure, no feedback used)
      adapt_residual      refit on what is measured, find which HELD-OUT genes are
                          predicted worst, and measure pool genes GO-similar to them
      adapt_greedy        greedy on the policy's own reward: pick the candidate that
                          most improves retrieval on the SELECTION genes. Legitimate
                          -- sel is the training signal a policy would also get.
      oracle_greedy       the same greedy, scored directly on the REPORT genes.
                          CHEATING, and reported only as the bound on what any
                          adaptive policy could reach.

    All strategies report on `rep`, which only oracle_greedy is allowed to look at."""
    K = len(D["cands"])
    mk = D["mask"]
    Y = D["dF"][:, mk]
    perm = rng.permutation(K)
    n_ev = max(60, K // 5)
    rep_i, sel_i, pool_i = np.sort(perm[:n_ev]), np.sort(perm[n_ev:2 * n_ev]), np.sort(perm[2 * n_ev:])

    def queries(idx):
        m = np.isin(D["owner"], idx)
        remap = {o: n for n, o in enumerate(idx)}
        return D["Q"][m][:, mk], np.array([remap[o] for o in D["owner"][m]])

    Q_sel, o_sel = queries(sel_i)
    Q_rep, o_rep = queries(rep_i)
    J_sel, J_rep = jaccard(F[sel_i], F[pool_i]), jaccard(F[rep_i], F[pool_i])
    Y_pool, Y_sel = Y[pool_i], Y[sel_i]
    Fp = F[pool_i] > 0
    log(f"[adaptivity] {D['name']}: pool={pool_i.size} sel={sel_i.size} rep={rep_i.size}")

    def predict(J, chosen):
        c = np.asarray(chosen)
        return go_knn_predict(J[:, c], Y_pool[c], Y_pool[c].mean(0))

    def ev(chosen, which):
        J, Q, o = (J_sel, Q_sel, o_sel) if which == "sel" else (J_rep, Q_rep, o_rep)
        return score(predict(J, chosen), Q, o)

    R_ROUNDS = 4
    N_CAND = 120
    for B in (25, 50, 100, 200):
        if B > pool_i.size:
            continue
        for strat in ("random", "coverage", "adapt_residual", "adapt_greedy", "oracle_greedy"):
            t0 = time.time()
            srng = np.random.default_rng(7000 + B)
            chosen, covered = [], np.zeros(Fp.shape[1], bool)
            per_round = []
            for rd in range(R_ROUNDS):
                b = B // R_ROUNDS + (1 if rd < B % R_ROUNDS else 0)
                avail = np.setdiff1d(np.arange(pool_i.size), chosen)
                if strat == "random":
                    pick = srng.choice(avail, size=min(b, avail.size), replace=False)
                elif strat == "coverage":
                    pick = []
                    for _ in range(b):
                        gain = (Fp & ~covered).sum(1).astype(float)
                        gain[chosen + list(pick)] = -1
                        i = int(np.argmax(gain))
                        pick.append(i)
                        covered |= Fp[i]
                    pick = np.array(pick)
                elif strat == "adapt_residual":
                    if not chosen:
                        pick = srng.choice(avail, size=b, replace=False)
                    else:
                        P = predict(J_sel, chosen)
                        c = np.sum(unit(P) * unit(Y_sel), axis=1)
                        worst = sel_i[np.argsort(c)[: max(5, sel_i.size // 5)]]
                        aff = jaccard(F[worst], F[pool_i][avail]).max(0)
                        pick = avail[np.argsort(-aff)[:b]]
                else:
                    which = "sel" if strat == "adapt_greedy" else "rep"
                    pick = []
                    for _ in range(b):
                        cand = srng.choice(np.setdiff1d(avail, pick),
                                           size=min(N_CAND, avail.size - len(pick)), replace=False)
                        base = chosen + list(pick)
                        sc = [ev(base + [int(c)], which)["mrr"] if base else
                              ev([int(c)] + list(srng.choice(avail, size=2, replace=False)), which)["mrr"]
                              for c in cand]
                        pick.append(int(cand[int(np.argmax(sc))]))
                    pick = np.array(pick)
                chosen += [int(x) for x in pick]
                per_round.append(float(ev(chosen, "rep")["mrr"]))
            r = ev(chosen, "rep")
            rows.append({"dataset": D["name"], "part": "adaptivity", "strategy": strat,
                         "budget": int(B), "rounds": R_ROUNDS, "pool": int(pool_i.size),
                         "K_report": int(rep_i.size), "rep_mrr": r["mrr"], "rep_top1": r["top1"],
                         "per_round_rep_mrr": per_round, "ceiling_top1": ceil["top1"],
                         "cheats": strat == "oracle_greedy", "secs": round(time.time() - t0, 1)})
            log(f"    B={B:4d} {strat:15s} rep mrr={r['mrr']:.4f} top1={r['top1']:.4f} "
                f"rounds={['%.3f' % x for x in per_round]} ({time.time()-t0:.0f}s)")



# ================================================================ part 4
def reward_noise_headroom(D, F, rng, rows, ceil):
    """Is the acquisition prize UNFINDABLE, or just hidden behind a noisy reward?

    Part 3 left the decisive ambiguity. Greedy selecting on a held-out gene set
    reaches ~1.2x random on a DISJOINT set, while the same greedy selecting on the
    set it reports on reaches ~2x. Two readings:

      (a) the 2x is pure overfitting to those particular genes -- nothing to find,
          and the acquisition framing tops out near the static GO heuristic; or
      (b) the reward is an unbiased but HIGH-VARIANCE estimate, greedy chases its
          noise, and a better estimate transfers further.

    They differ in a testable way: under (b), growing the reward's gene set must
    raise the score on a FIXED disjoint report set; under (a) it stays flat. That
    is the difference between "no project" and "a variance-reduction problem",
    which is what learned baselines and policy generalisation are for.

    Splits are three-way and disjoint: `act` (genes a policy may measure), `rew`
    (genes the reward is computed on, subsampled to n_rew), `rep` (fixed report
    set, never seen by any strategy except the explicitly-cheating bound)."""
    K = len(D["cands"])
    mk = D["mask"]
    Y = D["dF"][:, mk]
    perm = rng.permutation(K)
    n_rep = max(60, K // 6)
    n_act = min((K - n_rep) // 2, 437)
    rep_i = np.sort(perm[:n_rep])
    act_i = np.sort(perm[n_rep:n_rep + n_act])
    rew_pool = np.sort(perm[n_rep + n_act:])

    def queries(idx):
        m = np.isin(D["owner"], idx)
        remap = {o: n for n, o in enumerate(idx)}
        return D["Q"][m][:, mk], np.array([remap[o] for o in D["owner"][m]])

    Q_rep, o_rep = queries(rep_i)
    J_rep = jaccard(F[rep_i], F[act_i])
    Y_act = Y[act_i]
    log(f"[reward-noise] {D['name']}: act={act_i.size} rew_pool={rew_pool.size} rep={rep_i.size}")

    def predict(J, chosen):
        c = np.asarray(chosen)
        return go_knn_predict(J[:, c], Y_act[c], Y_act[c].mean(0))

    B, R_ROUNDS, N_CAND = 100, 4, 120
    if B > act_i.size:
        return

    def run_greedy(rew_i, tag, srng):
        """Greedy on a reward computed over `rew_i`; reported on the fixed rep set."""
        Q_r, o_r = queries(rew_i)
        J_r = jaccard(F[rew_i], F[act_i])
        chosen = []
        for rd in range(R_ROUNDS):
            b = B // R_ROUNDS + (1 if rd < B % R_ROUNDS else 0)
            avail = np.setdiff1d(np.arange(act_i.size), chosen)
            pick = []
            for _ in range(b):
                cand = srng.choice(np.setdiff1d(avail, pick),
                                   size=min(N_CAND, avail.size - len(pick)), replace=False)
                base = chosen + list(pick)
                sc = [score(predict(J_r, base + [int(c)]), Q_r, o_r)["mrr"] if base else
                      score(predict(J_r, [int(c)] + list(srng.choice(avail, size=2, replace=False))),
                            Q_r, o_r)["mrr"] for c in cand]
                pick.append(int(cand[int(np.argmax(sc))]))
            chosen += pick
        r = score(predict(J_rep, chosen), Q_rep, o_rep)
        own = score(predict(J_r, chosen), Q_r, o_r)
        rows.append({"dataset": D["name"], "part": "reward_noise", "strategy": tag,
                     "n_reward_genes": int(rew_i.size), "budget": B, "rounds": R_ROUNDS,
                     "act_pool": int(act_i.size), "K_report": int(rep_i.size),
                     "rep_mrr": r["mrr"], "rep_top1": r["top1"],
                     "own_reward_mrr": own["mrr"], "ceiling_top1": ceil["top1"],
                     "cheats": tag == "cheat_on_rep"})
        log(f"    n_rew={rew_i.size:4d} {tag:14s} own={own['mrr']:.4f} -> rep mrr={r['mrr']:.4f} "
            f"top1={r['top1']:.4f}")
        return r

    # Random must be a DISTRIBUTION, not one draw: part 2 showed a single random
    # subset spans roughly +/-15% around the median, which is the same size as the
    # effect being tested. Comparing a strategy to one draw measures luck.
    N_RAND = 100
    srng = np.random.default_rng(9100)
    rv = []
    for _ in range(N_RAND):
        ch = srng.choice(act_i.size, size=B, replace=False)
        rr = score(predict(J_rep, ch), Q_rep, o_rep)
        rv.append((rr["mrr"], rr["top1"]))
    rv = np.array(rv)
    rows.append({"dataset": D["name"], "part": "reward_noise", "strategy": "random",
                 "n_reward_genes": 0, "budget": B, "n_draws": N_RAND,
                 "act_pool": int(act_i.size), "K_report": int(rep_i.size),
                 "rep_mrr": float(np.median(rv[:, 0])),
                 "rep_mrr_p05": float(np.percentile(rv[:, 0], 5)),
                 "rep_mrr_p95": float(np.percentile(rv[:, 0], 95)),
                 "rep_mrr_max": float(rv[:, 0].max()),
                 "rep_top1": float(np.median(rv[:, 1])),
                 "ceiling_top1": ceil["top1"], "cheats": False})
    log(f"    {'random x%d' % N_RAND:25s} rep mrr med={np.median(rv[:,0]):.4f} "
        f"p95={np.percentile(rv[:,0],95):.4f} max={rv[:,0].max():.4f} "
        f"top1 med={np.median(rv[:,1]):.4f}")
    if os.environ.get("PJ_RL_RANDOM_ONLY"):
        log("    PJ_RL_RANDOM_ONLY set: skipping the greedy sweep")
        return

    for n_rew in (25, 50, 100, 200, 400):
        if n_rew > rew_pool.size:
            continue
        sub = np.sort(np.random.default_rng(555).choice(rew_pool, size=n_rew, replace=False))
        run_greedy(sub, "greedy_on_rew", np.random.default_rng(9200 + n_rew))
    # the cheating bound, in THIS split, so it is comparable to the rows above
    run_greedy(rep_i, "cheat_on_rep", np.random.default_rng(9300))


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--gene2go", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--parts", nargs="*", type=int, default=[1, 2, 3])
    a = ap.parse_args()

    od = os.path.join(a.out_dir, "rl_headroom")
    os.makedirs(od, exist_ok=True)
    obj, acq, adp, rwn, status = [], [], [], [], {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "errors": []}

    def dump():
        # Only rewrite a part's file if that part actually ran. Writing an empty
        # list over a previous run's results silently destroys them.
        for nm, rs, part in (("objective.jsonl", obj, 1), ("acquisition.jsonl", acq, 2),
                             ("adaptivity.jsonl", adp, 3),
                             ("reward_noise.jsonl", rwn, 4)):
            if part not in a.parts:
                continue
            with open(os.path.join(od, nm), "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
        with open(os.path.join(od, "summary.json"), "w") as f:
            json.dump({**status, "config": {"pool_cells": POOL, "budgets": list(BUDGETS),
                                            "n_subsets": N_SUBSETS, "whiten_r": list(WHITEN_R),
                                            "folds": N_FOLDS, "seed": SEED}}, f, indent=2)

    g2g = load_gene2go(a.gene2go)
    log(f"gene2go: {len(g2g)} genes")
    for name in a.datasets:
        try:
            D = build(name, os.path.join(a.gears_dir, name, "perturb_processed.h5ad"),
                      np.random.default_rng(SEED))
            F = go_features(D["cands"], g2g)
            mk = D["mask"]
            ceil = (objective_headroom(D, F, np.random.default_rng(SEED + 1), obj)
                    if 1 in a.parts else score(D["dA"][:, mk], D["Q"][:, mk], D["owner"]))
            dump()
            if 2 in a.parts:
                acquisition_headroom(D, F, g2g, np.random.default_rng(SEED + 2), acq, ceil)
                dump()
            if 3 in a.parts:
                adaptivity_headroom(D, F, np.random.default_rng(SEED + 2), adp, ceil)
                dump()
            if 4 in a.parts:
                reward_noise_headroom(D, F, np.random.default_rng(SEED + 4), rwn, ceil)
                dump()
            del D, F
        except Exception as e:
            traceback.print_exc()
            status["errors"].append(f"{name}: {type(e).__name__}: {e}")
            dump()
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump()
    log("errors:", status["errors"] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
