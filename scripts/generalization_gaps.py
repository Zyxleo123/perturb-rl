#!/usr/bin/env python
"""Where does inverse design actually have headroom?

For a perturbation whose effect has been MEASURED, inverse design is a lookup:
the nearest measured centroid names it (86.6% top-1 on Norman at 128 cells).
Nobody needs a model there. Headroom exists only where the atlas has no entry
for a candidate, and there are three ways that happens:

  axis "gene"     the gene was never perturbed in this context
  axis "combo"    the singles were measured, the pair was not
  axis "context"  the gene was perturbed, but in a different cell line / lab

For each axis this script fixes ONE task -- a target state (pseudo-bulk of up to
32 held-out cells) must be matched to the right candidate among ALL candidates --
and scores every predictor on identical queries:

  ceiling    the candidate's measured centroid (half A of its cells; the queries
             come from the disjoint half B, against an independent control half)
  chance     1 / K
  cheap      the strongest cheap predictors available for that axis, each made
             out-of-fold so a candidate is never predicted by a model fitted on it

Headroom is reported against the BEST CHEAP predictor, not against a published
model: a gap a linear baseline closes is not a gap.

Two gene spaces throughout:
  full     every measured gene
  masked   every gene that is the target of ANY candidate removed. A design target
           is a phenotype and carries no knocked-down-gene fingerprint, so this is
           the design-relevant number (see GAPS.md gap 2).

Writes  <out>/generalization/results.jsonl   one row per (axis, setting, space, predictor)
        <out>/generalization/per_condition.jsonl
        <out>/generalization/summary.json
Always exits 0 on a data problem, after writing what it could.

Usage: python -I generalization_gaps.py --gears-dir D --umi-dir D --oracle-npz F
                                        --gene2go F --out-dir D
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

POOL = 32          # cells per target pseudo-bulk (fewer if half B is smaller)
N_DRAWS = 3        # query draws per candidate when half B has > POOL cells
MIN_CELLS = 12     # each half then has >= 6 cells
N_FOLDS = 5
SEED = 0
TOPKS = (1, 5, 10)
FEWSHOT = (10, 25, 50, 100, 200, 400)
FEWSHOT_SEEDS = 3

SINGLE_DATASETS = ("replogle_k562_essential", "replogle_rpe1_essential", "adamson", "norman")
CONTEXT_PAIRS = (
    ("replogle_k562_essential", "replogle_rpe1_essential"),
    ("replogle_rpe1_essential", "replogle_k562_essential"),
    ("adamson", "replogle_k562_essential"),   # same line (K562), different experiment
    ("replogle_k562_essential", "adamson"),
    ("adamson", "replogle_rpe1_essential"),   # different line AND experiment
)
MIN_SHARED_FOR_LEARNING = 100


def log(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.strftime('%H:%M:%S')} rss={rss:.1f}GB]", *a, flush=True)


def densify(x):
    import scipy.sparse as sp
    return np.asarray(x.todense(), dtype=np.float32) if sp.issparse(x) else np.asarray(x, dtype=np.float32)


def canon(c):
    parts = sorted(p for p in str(c).split("+") if p != "ctrl")
    return "+".join(parts) if parts else "ctrl"


def parts(c):
    return [p for p in c.split("+") if p != "ctrl"]


def unit(M):
    M = np.atleast_2d(np.asarray(M, dtype=np.float32))
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return np.where(n > 0, M / np.maximum(n, 1e-12), 0.0).astype(np.float32)


class _NoGlobals(pickle.Unpickler):
    """gene2go_all.pkl is a downloaded pickle. A dict of sets of strings needs no
    globals at all, so refuse every one: nothing in the file can import or call."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def load_gene2go(path):
    if not path or not os.path.exists(path):
        log("gene2go not found; GO predictor disabled:", path)
        return None
    try:
        with open(path, "rb") as f:
            d = _NoGlobals(f).load()
        g2g = {str(k): frozenset(map(str, v)) for k, v in d.items()}
        log(f"gene2go: {len(g2g)} genes")
        return g2g
    except Exception as e:
        log(f"gene2go unreadable without globals ({type(e).__name__}: {e}); GO predictor disabled")
        return None


# ================================================================ data
def build_dataset(name, h5, rng, ctrl_mean_override=None, split_conditions=None, split_rng=None):
    import anndata as ad
    import pandas as pd

    log(f"loading {name}: {h5}")
    adata = ad.read_h5ad(h5)
    X = adata.X
    if hasattr(X, "tocsr"):
        X = X.tocsr()
    genes = np.asarray(
        adata.var["gene_name"].astype(str).values if "gene_name" in adata.var
        else adata.var_names.astype(str), dtype=object)
    conds = np.asarray([canon(c) for c in adata.obs["condition"].astype(str).values], dtype=object)
    log(f"  {adata.shape}, {len(set(conds))} canonical conditions")

    codes, uniq = pd.factorize(conds)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    rows_of = {uniq[i]: order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))}

    ctrl_rows = rows_of["ctrl"].copy()
    rng.shuffle(ctrl_rows)
    h = ctrl_rows.size // 2
    Xc = densify(X[ctrl_rows])
    muA, muB = Xc[:h].mean(0), Xc[h:].mean(0)
    mu = Xc.mean(0) if ctrl_mean_override is None else ctrl_mean_override.astype(np.float32)
    del Xc

    if split_conditions is not None:
        cands = [c for c in split_conditions if c in rows_of]
    else:
        cands = sorted(c for c in rows_of if c != "ctrl" and rows_of[c].size >= MIN_CELLS)
    K, G = len(cands), len(genes)
    dA = np.zeros((K, G), np.float32)
    dB = np.zeros((K, G), np.float32)
    dF = np.zeros((K, G), np.float32)
    n = np.zeros(K, int)
    Q, owner, pool = [], [], []
    srng = split_rng if split_rng is not None else rng
    for i, c in enumerate(cands):
        r = rows_of[c].copy()
        srng.shuffle(r)
        hh = r.size // 2
        XA, XB = densify(X[r[:hh]]), densify(X[r[hh:]])
        if ctrl_mean_override is not None:
            # reproduce inverse_gap.py exactly: both halves against the full control mean
            dA[i] = XA.mean(0) - mu
            dB[i] = XB.mean(0) - mu
        else:
            dA[i] = XA.mean(0) - muA
            dB[i] = XB.mean(0) - muB
        dF[i] = (XA.sum(0) + XB.sum(0)) / r.size - mu
        n[i] = r.size
        p = min(POOL, XB.shape[0])
        draws = N_DRAWS if XB.shape[0] > POOL else 1
        base = mu if ctrl_mean_override is not None else muB
        for _ in range(draws):
            pick = rng.choice(XB.shape[0], size=p, replace=False)
            Q.append(XB[pick].mean(0) - base)
            owner.append(i)
            pool.append(p)
    rho = np.sum(unit(dA) * unit(dB), axis=1)
    d = {
        "name": name, "genes": genes, "gidx": {g: j for j, g in enumerate(genes)},
        "cands": cands, "cidx": {c: i for i, c in enumerate(cands)}, "n": n,
        "dA": dA, "dB": dB, "dF": dF, "Q": np.vstack(Q).astype(np.float32),
        "owner": np.asarray(owner), "pool": np.asarray(pool), "rho": rho,
        "n_ctrl": int(ctrl_rows.size),
    }
    log(f"  {K} candidates (>= {MIN_CELLS} cells), {len(Q)} queries, "
        f"median pool {int(np.median(pool))}, median split-half cos {np.median(rho):.3f}")
    del adata, X
    return d


# ================================================================ scoring
_H = None


def harmonic(N):
    global _H
    if _H is None or _H.size < N + 2:
        _H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, N + 2))])
    return _H


def score(P, Q, owner, cand_ok=None):
    """Rank every candidate for every query by cosine. Tie-aware: a predictor that
    gives several candidates the same score gets the EXPECTED rank under random
    tie-breaking, so a constant predictor scores exactly chance.

    Macro-averaged: per candidate over its queries, then over candidates.
    cand_ok restricts the QUERIES to candidates the predictor covers; the
    candidate set itself always stays complete."""
    K = P.shape[0]
    S = np.round(unit(Q) @ unit(P).T, 6)
    t = S[np.arange(S.shape[0]), owner]
    g = (S > t[:, None]).sum(1)
    e = (S == t[:, None]).sum(1) - 1
    H = harmonic(K)
    rr = (H[g + e + 1] - H[g]) / (e + 1)
    rank = g + 1 + e / 2.0
    per = {f"top{k}": np.clip((k - g) / (e + 1), 0, 1) for k in TOPKS}
    per["mrr"] = rr
    per["rank"] = rank
    keep = np.ones(K, bool) if cand_ok is None else np.asarray(cand_ok, bool)
    out = {}
    for key, v in per.items():
        sums = np.bincount(owner, weights=v, minlength=K)
        cnt = np.bincount(owner, minlength=K)
        m = (cnt > 0) & keep
        per_c = sums[m] / cnt[m]
        out[key] = per_c
    res = {f"top{k}": float(out[f"top{k}"].mean()) for k in TOPKS}
    res["mrr"] = float(out["mrr"].mean())
    res["median_rank"] = float(np.median(out["rank"]))
    res["norm_rank"] = float(np.mean((out["rank"] - 1) / max(1, K - 1)))
    res["K"] = int(K)
    res["chance_top1"] = 1.0 / K
    res["chance_mrr"] = float(H[K] / K)
    res["n_query_candidates"] = int(out["mrr"].size)
    return res


def fidelity(P, dB, rho):
    """cos(prediction, held-out half-B centroid), against the noise ceiling for that
    comparison: a perfect predictor of the true mean scores sqrt(rho) against a
    half-sample centroid whose split-half cosine is rho."""
    c = np.sum(unit(P) * unit(dB), axis=1)
    ceil = np.sqrt(np.clip(rho, 0, None))
    ok = ceil > 0.2
    return {
        "median_cos_vs_heldout": float(np.median(c)),
        "median_noise_ceiling": float(np.median(ceil)),
        "median_frac_of_ceiling": float(np.median(c[ok] / ceil[ok])) if ok.any() else float("nan"),
    }


# ================================================================ predictors
def folds_of(K, rng, k=N_FOLDS):
    perm = rng.permutation(K)
    return [np.sort(perm[i::k]) for i in range(k)]


def target_sign(D, train_idx):
    """+1 for CRISPRa-like data, -1 for CRISPRi-like, learned from training rows."""
    v = []
    for i in train_idx:
        for g in parts(D["cands"][i]):
            j = D["gidx"].get(g)
            if j is not None:
                v.append(D["dF"][i, j])
    return float(np.sign(np.median(v))) if v else -1.0


def barcode_pred(D, idx, sign):
    P = np.zeros((len(idx), len(D["genes"])), np.float32)
    for r, i in enumerate(idx):
        for g in parts(D["cands"][i]):
            j = D["gidx"].get(g)
            if j is not None:
                P[r, j] = sign
    return P


def go_knn(D, tr, te, g2g, k=10):
    if g2g is None:
        return None
    tg = [D["cands"][i] for i in tr]
    tsets = [g2g.get(g, frozenset()) for g in tg]
    base = D["dF"][tr].mean(0)
    P = np.repeat(base[None], len(te), 0)
    for r, i in enumerate(te):
        s = g2g.get(D["cands"][i], frozenset())
        if not s:
            continue
        J = np.array([len(s & t) / len(s | t) if t else 0.0 for t in tsets])
        top = np.argsort(-J)[:k]
        top = top[J[top] > 0]
        if top.size:
            w = J[top] / J[top].sum()
            P[r] = w @ D["dF"][tr][top]
    return P


def _lin_prep(Y):
    b = Y.mean(1)
    Yc = Y - b[:, None]
    U = np.linalg.svd(Yc, full_matrices=False)[0]
    return b, Yc, U


def _lin_predict(prep, tgt_tr, tgt_te, r, lam):
    """Ahlmann-Eltze et al. (Nat Methods 2025) linear baseline:
    Y - b ~ E W P^T with E = top-r gene PCs of the TRAINING signatures and each
    perturbation embedded as the E-row of its target gene. Needs the target measured."""
    b, Yc, U = prep
    r = min(r, U.shape[1])
    E = U[:, :r]
    scale = np.sqrt(E.shape[0])
    okt = tgt_tr >= 0
    if okt.sum() < 3:
        return None
    Pm = E[tgt_tr[okt]] * scale
    W = (E.T @ Yc[:, okt] @ Pm) @ np.linalg.inv(Pm.T @ Pm + lam * np.eye(r)) / (1.0 + lam)
    out = np.repeat(b[None], len(tgt_te), 0)
    m = tgt_te >= 0
    if m.any():
        out[m] = b[None] + (E @ W @ (E[tgt_te[m]] * scale).T).T
    return out


def linear_pred(D, tr, te, mask, rng):
    tgt = np.array([D["gidx"].get(D["cands"][i], -1) for i in range(len(D["cands"]))])
    # inner split on the training fold only, selecting (r, lambda) by retrieval MRR
    inner = rng.permutation(len(tr))
    nv = max(5, len(tr) // 5)
    iv, it = tr[inner[:nv]], tr[inner[nv:]]
    prep = _lin_prep(D["dF"][it].T.astype(np.float64))
    q = D["dF"][iv][:, mask]
    best, best_cfg = -1.0, (10, 0.1)
    for r in (5, 10, 20, 50):
        for lam in (0.01, 0.1, 1.0, 10.0):
            p = _lin_predict(prep, tgt[it], tgt[iv], r, lam)
            if p is None:
                continue
            m = score(p[:, mask].astype(np.float32), q, np.arange(len(iv)))["mrr"]
            if m > best:
                best, best_cfg = m, (r, lam)
    p = _lin_predict(_lin_prep(D["dF"][tr].T.astype(np.float64)), tgt[tr], tgt[te], *best_cfg)
    return (None if p is None else p.astype(np.float32)), best_cfg, (tgt[te] >= 0)


def coresp_knn(D, tr, te, k=10):
    """Neighbours by how the TARGET GENE itself responds across training
    perturbations. Uses only training signatures; needs the target measured."""
    tgt = np.array([D["gidx"].get(D["cands"][i], -1) for i in range(len(D["cands"]))])
    Y = D["dF"][tr]                     # n_tr x G
    Yc = (Y - Y.mean(0)).T              # G x n_tr: each gene's response profile
    base = Y.mean(0)
    P = np.repeat(base[None], len(te), 0)
    tt = tgt[tr]
    okt = tt >= 0
    if okt.sum() < 3:
        return P
    prof_tr = unit(Yc[tt[okt]])
    Ytr_ok = Y[okt]
    for r, i in enumerate(te):
        g = tgt[i]
        if g < 0:
            continue
        sim = prof_tr @ unit(Yc[g][None])[0]
        top = np.argsort(-sim)[:k]
        top = top[sim[top] > 0]
        if top.size:
            w = sim[top] / sim[top].sum()
            P[r] = w @ Ytr_ok[top]
    return P


# ================================================================ axes
def masks_for(D, extra_targets=()):
    targets = {g for c in D["cands"] for g in parts(c)} | set(extra_targets)
    masked = np.array([g not in targets for g in D["genes"]])
    return {"full": np.ones(len(D["genes"]), bool), "masked": masked}


def emit(rows, base, space, pred, sc, fid, **extra):
    row = {**base, "space": space, "predictor": pred, **sc, **fid, **extra}
    rows.append(row)
    log(f"    {base.get('setting','')} {space:6s} {pred:14s} top1={sc['top1']:.3f} "
        f"mrr={sc['mrr']:.3f} (chance {sc['chance_top1']:.4f}) cos={fid['median_cos_vs_heldout']:.3f} "
        f"of ceil {fid['median_frac_of_ceiling']:.2f}")


def axis_gene(D, g2g, rng, rows, oof_store):
    """Unseen gene, same context: singles only, OOF over 5 folds of genes."""
    sing = [i for i, c in enumerate(D["cands"]) if len(parts(c)) == 1]
    if len(sing) < 20:
        return
    sub = {k: (v[sing] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(D["cands"]) else v)
           for k, v in D.items()}
    sub["cands"] = [D["cands"][i] for i in sing]
    sub["cidx"] = {c: i for i, c in enumerate(sub["cands"])}
    qm = np.isin(D["owner"], sing)
    remap = {old: new for new, old in enumerate(sing)}
    sub["Q"] = D["Q"][qm]
    sub["owner"] = np.array([remap[o] for o in D["owner"][qm]])
    S = sub
    K, G = len(S["cands"]), len(S["genes"])
    M = masks_for(S)
    log(f"[gene] {S['name']}: K={K} singles, masked space drops {int((~M['masked']).sum())} target genes")
    preds = {p: np.zeros((K, G), np.float32) for p in ("mean", "barcode", "go_knn", "coresp_knn", "linear")}
    covered = np.zeros(K, bool)
    cfgs = []
    for f, te in enumerate(folds_of(K, rng)):
        tr = np.setdiff1d(np.arange(K), te)
        preds["mean"][te] = S["dF"][tr].mean(0)
        preds["barcode"][te] = barcode_pred(S, te, target_sign(S, tr))
        go = go_knn(S, tr, te, g2g)
        if go is not None:
            preds["go_knn"][te] = go
        else:
            preds.pop("go_knn", None)
        preds["coresp_knn"][te] = coresp_knn(S, tr, te)
        lp, cfg, cov = linear_pred(S, tr, te, M["masked"], rng)
        cfgs.append(cfg)
        if lp is not None:
            preds["linear"][te] = lp
        covered[te] = cov
    base = {"axis": "gene", "setting": S["name"], "dataset": S["name"],
            "frac_targets_measured": float(covered.mean()), "linear_cfgs": [list(c) for c in cfgs]}
    for space, m in M.items():
        Qm = S["Q"][:, m]
        sc = score(S["dA"][:, m], Qm, S["owner"])
        emit(rows, base, space, "ceiling", sc, fidelity(S["dA"][:, m], S["dB"][:, m], S["rho"]))
        for p, P in preds.items():
            if p == "barcode" and space == "masked":
                continue  # identically zero there: chance by construction
            sc = score(P[:, m], Qm, S["owner"])
            emit(rows, base, space, p, sc, fidelity(P[:, m], S["dB"][:, m], S["rho"]))
            if p in ("linear", "coresp_knn"):
                sc2 = score(P[:, m], Qm, S["owner"], cand_ok=covered)
                emit(rows, {**base, "subset": "targets_measured"}, space, p, sc2,
                     fidelity(P[covered][:, m], S["dB"][covered][:, m], S["rho"][covered]))
    oof_store[S["name"]] = {"cands": S["cands"], "preds": preds}


def axis_combo(umi, oracle, rng, rows):
    """Unseen combination: both singles measured, the pair is not."""
    C = umi
    cidx = C["cidx"]
    combos = [i for i, c in enumerate(C["cands"])
              if len(parts(c)) == 2 and all(p in cidx for p in parts(c))]
    if len(combos) < 10:
        log("[combo] too few combos with both singles measured")
        return
    K = len(combos)
    G = len(C["genes"])
    oc = {str(c): i for i, c in enumerate(oracle["conditions"])}
    gears = np.stack([oracle["oracle_sig"][oc[C["cands"][i]]] for i in combos]).astype(np.float32)
    add = np.stack([C["dF"][cidx[a]] + C["dF"][cidx[b]]
                    for a, b in (parts(C["cands"][i]) for i in combos)]).astype(np.float32)
    dF = C["dF"][combos]
    preds = {"mean": np.zeros((K, G), np.float32), "additive": add,
             "additive_fit": np.zeros((K, G), np.float32), "gears_seen_all": gears,
             "barcode": np.zeros((K, G), np.float32)}
    alphas = []
    for te in folds_of(K, rng):
        tr = np.setdiff1d(np.arange(K), te)
        m = dF[tr].mean(0)
        preds["mean"][te] = m
        # 2-parameter fit on training combos: d_AB ~ a (d_A + d_B) + b * mean_combo
        A = np.stack([add[tr].ravel(), np.tile(m, len(tr))], 1).astype(np.float64)
        coef, *_ = np.linalg.lstsq(A, dF[tr].ravel().astype(np.float64), rcond=None)
        alphas.append([float(coef[0]), float(coef[1])])
        preds["additive_fit"][te] = coef[0] * add[te] + coef[1] * m
        sgn = target_sign(C, [combos[i] for i in tr])
        preds["barcode"][te] = barcode_pred(C, [combos[i] for i in te], sgn)
    qm = np.isin(C["owner"], combos)
    remap = {old: new for new, old in enumerate(combos)}
    Q = C["Q"][qm]
    owner = np.array([remap[o] for o in C["owner"][qm]])
    sub = {"cands": [C["cands"][i] for i in combos], "genes": C["genes"]}
    # mask every gene targeted by any Norman perturbation, singles included
    M = masks_for({"cands": C["cands"], "genes": C["genes"]})
    base = {"axis": "combo", "setting": "norman (GEARS gene space)", "dataset": "norman",
            "additive_fit_coefs": alphas,
            "note": "gears_seen_all = official checkpoint trained on ALL Norman combos "
                    "(split=no_test): an optimistic bound, not an unseen-combo number."}
    rho = C["rho"][combos]
    for space, mk in M.items():
        sc = score(C["dA"][combos][:, mk], Q[:, mk], owner)
        emit(rows, base, space, "ceiling", sc, fidelity(C["dA"][combos][:, mk], C["dB"][combos][:, mk], rho))
        for p, P in preds.items():
            if p == "barcode" and space == "masked":
                continue
            sc = score(P[:, mk], Q[:, mk], owner)
            emit(rows, base, space, p, sc, fidelity(P[:, mk], C["dB"][combos][:, mk], rho))
    # how much of each combo is NOT additive, beyond noise?
    c_add = np.sum(unit(add) * unit(C["dB"][combos]), axis=1)
    ceil = np.sqrt(np.clip(rho, 0, None))
    rows.append({"axis": "combo", "setting": "norman (GEARS gene space)", "kind": "per_combo",
                 "conditions": sub["cands"], "cos_additive_vs_heldout": c_add.tolist(),
                 "noise_ceiling": ceil.tolist()})


def ridge_map(Xs, Yt, Xte, rng, mask):
    """Source-line signature -> target-line signature, ridge in source PCA space.
    (r, lambda) chosen on an inner split of the training perturbations by MRR;
    lambda is relative to the mean eigenvalue so the grid is scale-free."""
    def prep(Xa, Ya):
        xm, ym = Xa.mean(0), Ya.mean(0)
        Xc = Xa - xm
        Vt = np.linalg.svd(Xc, full_matrices=False)[2]
        return xm, ym, Xc, Ya - ym, Vt

    def pred(pp, Xb, r, lam):
        xm, ym, Xc, Yc, Vt = pp
        r = max(1, min(r, Vt.shape[0] - 1))
        V = Vt[:r].T
        Z = Xc @ V
        ZZ = Z.T @ Z
        B = np.linalg.solve(ZZ + lam * np.trace(ZZ) / r * np.eye(r), Z.T @ Yc)
        return ym + (Xb - xm) @ V @ B

    n = Xs.shape[0]
    best, cfg = -1.0, (10, 0.1)
    if n >= 10:
        perm = rng.permutation(n)
        nv = max(3, n // 5)
        iv, it = perm[:nv], perm[nv:]
        pp = prep(Xs[it], Yt[it])
        for r in (5, 10, 20, 50, 100):
            if r >= len(it):
                continue
            for lam in (0.01, 0.1, 1.0):
                p = pred(pp, Xs[iv], r, lam)
                m = score(p[:, mask].astype(np.float32), Yt[iv][:, mask], np.arange(nv))["mrr"]
                if m > best:
                    best, cfg = m, (r, lam)
    return pred(prep(Xs, Yt), Xte, *cfg).astype(np.float32), cfg


def axis_context(src, tgt, rng, rows, oof_store):
    shared = sorted(set(c for c in src["cands"] if len(parts(c)) == 1)
                    & set(c for c in tgt["cands"] if len(parts(c)) == 1))
    genes = sorted(set(src["genes"]) & set(tgt["genes"]))
    if len(shared) < 15 or len(genes) < 200:
        log(f"[context] {src['name']} -> {tgt['name']}: {len(shared)} shared perts, {len(genes)} genes; skip")
        return
    si = np.array([src["cidx"][c] for c in shared])
    ti = np.array([tgt["cidx"][c] for c in shared])
    sg = np.array([src["gidx"][g] for g in genes])
    tg = np.array([tgt["gidx"][g] for g in genes])
    targets = {g for c in src["cands"] for g in parts(c)} | {g for c in tgt["cands"] for g in parts(c)}
    M = {"full": np.ones(len(genes), bool), "masked": np.array([g not in targets for g in genes])}
    K = len(shared)
    Xs = src["dF"][si][:, sg]
    Yt = tgt["dF"][ti][:, tg]
    qm = np.isin(tgt["owner"], ti)
    remap = {old: new for new, old in enumerate(ti)}
    Q = tgt["Q"][qm][:, tg]
    owner = np.array([remap[o] for o in tgt["owner"][qm]])
    dA, dB, rho = tgt["dA"][ti][:, tg], tgt["dB"][ti][:, tg], tgt["rho"][ti]
    setting = f"{src['name']} -> {tgt['name']}"
    log(f"[context] {setting}: K={K} shared perturbations, {len(genes)} shared genes "
        f"({int(M['masked'].sum())} after masking targets)")

    preds = {"naive": Xs.copy(), "mean_tgt": np.zeros_like(Yt), "shifted": np.zeros_like(Yt)}
    learn = K >= MIN_SHARED_FOR_LEARNING
    if learn:
        preds["ridge_map"] = np.zeros_like(Yt)
    cfgs = []
    F = folds_of(K, rng)
    for te in F:
        tr = np.setdiff1d(np.arange(K), te)
        preds["mean_tgt"][te] = Yt[tr].mean(0)
        preds["shifted"][te] = Xs[te] - Xs[tr].mean(0) + Yt[tr].mean(0)
        if learn:
            p, cfg = ridge_map(Xs[tr], Yt[tr], Xs[te], rng, M["masked"])
            preds["ridge_map"][te] = p
            cfgs.append(cfg)
    # within-target unseen-gene predictor on the same perturbations, for comparison:
    # is the other cell line more informative than the rest of this one?
    within = oof_store.get(tgt["name"])
    if within is not None:
        wi = {c: i for i, c in enumerate(within["cands"])}
        if all(c in wi for c in shared):
            for p in ("linear", "coresp_knn"):
                if p in within["preds"]:
                    preds[f"within_{p}"] = within["preds"][p][[wi[c] for c in shared]][:, tg]
    base = {"axis": "context", "setting": setting, "src": src["name"], "tgt": tgt["name"],
            "n_shared_perts": K, "n_shared_genes": len(genes), "ridge_cfgs": [list(c) for c in cfgs]}
    for space, mk in M.items():
        sc = score(dA[:, mk], Q[:, mk], owner)
        emit(rows, base, space, "ceiling", sc, fidelity(dA[:, mk], dB[:, mk], rho))
        for p, P in preds.items():
            sc = score(P[:, mk], Q[:, mk], owner)
            emit(rows, base, space, p, sc, fidelity(P[:, mk], dB[:, mk], rho))

    # disattenuated cross-context similarity per perturbation: the cosine two noisy
    # measurements of the SAME true effect would reach is sqrt(R_src R_tgt), with
    # R = 2 rho / (1 + rho) the full-sample reliability.
    for space, mk in M.items():
        def rel(D, idx, gi):
            r = np.sum(unit(D["dA"][idx][:, gi][:, mk]) * unit(D["dB"][idx][:, gi][:, mk]), axis=1)
            r = np.clip(r, 1e-6, None)
            return 2 * r / (1 + r)
        Rs, Rt = rel(src, si, sg), rel(tgt, ti, tg)
        cx = np.sum(unit(Xs[:, mk]) * unit(Yt[:, mk]), axis=1)
        ceil = np.sqrt(Rs * Rt)
        ok = ceil > 0.2
        rows.append({"axis": "context", "setting": setting, "kind": "disattenuated", "space": space,
                     "n": int(ok.sum()), "median_cross_cos": float(np.median(cx[ok])),
                     "median_ceiling": float(np.median(ceil[ok])),
                     "median_disattenuated": float(np.median(cx[ok] / ceil[ok])),
                     "values_disattenuated": (cx[ok] / ceil[ok]).round(4).tolist()})
        log(f"    {space:6s} cross-context cos {np.median(cx[ok]):.3f} vs same-effect ceiling "
            f"{np.median(ceil[ok]):.3f} -> disattenuated {np.median(cx[ok]/ceil[ok]):.3f}")

    # few-shot: how many perturbations must be measured in the NEW context?
    if learn:
        for ntr in FEWSHOT + (None,):
            for seed in range(FEWSHOT_SEEDS if ntr is not None else 1):
                frng = np.random.default_rng(1000 + seed)
                P = {"shifted": np.zeros_like(Yt), "ridge_map": np.zeros_like(Yt)}
                used = []
                for te in F:
                    tr = np.setdiff1d(np.arange(K), te)
                    if ntr is not None:
                        if ntr > tr.size:
                            break
                        tr = np.sort(frng.choice(tr, size=ntr, replace=False))
                    used.append(tr.size)
                    P["shifted"][te] = Xs[te] - Xs[tr].mean(0) + Yt[tr].mean(0)
                    P["ridge_map"][te], _ = ridge_map(Xs[tr], Yt[tr], Xs[te], frng, M["masked"])
                if len(used) < len(F):
                    continue
                mk = M["masked"]
                for p, PP in P.items():
                    sc = score(PP[:, mk], Q[:, mk], owner)
                    rows.append({**base, "kind": "fewshot", "space": "masked", "predictor": p,
                                 "n_train": int(np.mean(used)), "seed": seed, **sc})
                log(f"    fewshot n_train={int(np.mean(used))} seed={seed} done")


# ================================================================ main
def summarize(rows):
    """Headroom vs the best cheap predictor, per (axis, setting, space)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if "predictor" in r and "kind" not in r and r.get("subset") is None:
            groups[(r["axis"], r["setting"], r["space"])].append(r)
    out = []
    for (axis, setting, space), rs in groups.items():
        by = {r["predictor"]: r for r in rs}
        if "ceiling" not in by:
            continue
        ce = by["ceiling"]
        cheap = {p: r for p, r in by.items()
                 if p not in ("ceiling",) and not p.startswith("gears")}
        best_p = max(cheap, key=lambda p: cheap[p]["mrr"]) if cheap else None
        b = cheap[best_p] if best_p else None
        ch1, chm = ce["chance_top1"], ce["chance_mrr"]
        rec = lambda v, c, top: (v - c) / (top - c) if top > c else float("nan")
        out.append({
            "axis": axis, "setting": setting, "space": space, "K": ce["K"],
            "chance_top1": ch1, "ceiling_top1": ce["top1"], "ceiling_mrr": ce["mrr"],
            "best_cheap": best_p,
            "best_cheap_top1": b["top1"] if b else None,
            "best_cheap_mrr": b["mrr"] if b else None,
            "headroom_top1_pts": 100 * (ce["top1"] - b["top1"]) if b else None,
            "frac_recovered_top1": rec(b["top1"], ch1, ce["top1"]) if b else None,
            "frac_recovered_mrr": rec(b["mrr"], chm, ce["mrr"]) if b else None,
            "gears_top1": by.get("gears_seen_all", {}).get("top1"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--umi-dir", required=True, help="norman_umi_go dataloader dir (GEARS gene space)")
    ap.add_argument("--oracle-npz", required=True)
    ap.add_argument("--gene2go", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", nargs="*", default=list(SINGLE_DATASETS))
    args = ap.parse_args()

    od = os.path.join(args.out_dir, "generalization")
    os.makedirs(od, exist_ok=True)
    rows, status = [], {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "errors": []}

    def dump():
        with open(os.path.join(od, "results.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(od, "summary.json"), "w") as f:
            json.dump({**status, "headroom": summarize(rows),
                       "config": {"pool": POOL, "n_draws": N_DRAWS, "min_cells": MIN_CELLS,
                                  "n_folds": N_FOLDS, "seed": SEED}}, f, indent=2)

    g2g = load_gene2go(args.gene2go)
    data, oof = {}, {}
    pc_rows = []
    for name in args.datasets:
        h5 = os.path.join(args.gears_dir, name, "perturb_processed.h5ad")
        try:
            D = build_dataset(name, h5, np.random.default_rng(SEED))
            data[name] = D
            for i, c in enumerate(D["cands"]):
                pc_rows.append({"dataset": name, "condition": c, "n_cells": int(D["n"][i]),
                                "split_half_cos": float(D["rho"][i])})
            axis_gene(D, g2g, np.random.default_rng(SEED + 1), rows, oof)
            dump()
        except Exception as e:
            traceback.print_exc()
            status["errors"].append(f"{name}: {type(e).__name__}: {e}")

    try:
        oracle = np.load(args.oracle_npz, allow_pickle=False)
        conds = [str(c) for c in oracle["conditions"]]
        umi = build_dataset("norman_umi_go", os.path.join(args.umi_dir, "perturb_processed.h5ad"),
                            np.random.default_rng(SEED + 7),
                            ctrl_mean_override=oracle["ctrl_mean"], split_conditions=conds,
                            split_rng=np.random.default_rng(0))
        # sanity: we reproduce inverse_gap.py's split, so our half-A centroids must
        # equal the ones it saved. If not, the files are not the same data.
        oi = {c: i for i, c in enumerate(conds)}
        diff = max(float(np.abs(umi["dA"][umi["cidx"][c]] - oracle["empirical_sig"][oi[c]]).max())
                   for c in umi["cands"])
        status["umi_split_reproduction_max_abs_diff"] = diff
        log(f"[combo] split reproduction vs inverse_gap.py: max |diff| = {diff:.2e}")
        if list(map(str, oracle["gene_names"])) != list(map(str, umi["genes"])):
            raise RuntimeError("oracle npz gene order differs from norman_umi_go h5ad")
        axis_combo(umi, oracle, np.random.default_rng(SEED + 2), rows)
        dump()
    except Exception as e:
        traceback.print_exc()
        status["errors"].append(f"combo: {type(e).__name__}: {e}")

    for s, t in CONTEXT_PAIRS:
        if s in data and t in data:
            try:
                axis_context(data[s], data[t], np.random.default_rng(SEED + 3), rows, oof)
                dump()
            except Exception as e:
                traceback.print_exc()
                status["errors"].append(f"context {s}->{t}: {type(e).__name__}: {e}")

    with open(os.path.join(od, "per_condition.jsonl"), "w") as f:
        for r in pc_rows:
            f.write(json.dumps(r) + "\n")
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump()
    log("SUMMARY (headroom vs best cheap predictor):")
    for h in summarize(rows):
        log(f"  {h['axis']:8s} {h['setting'][:48]:48s} {h['space']:6s} K={h['K']:5d} "
            f"ceiling={h['ceiling_top1']:.3f} best={h['best_cheap']}:{(h['best_cheap_top1'] or 0):.3f} "
            f"chance={h['chance_top1']:.4f} recovered(mrr)={h['frac_recovered_mrr']:.2f}")
    log("errors:", status["errors"] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
