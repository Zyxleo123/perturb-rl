#!/usr/bin/env python
"""The combiner's input is an INNER PRODUCT, not a ranking. Estimate it directly.

`repr_select.py` showed the ridge combiner w* = (G_SS + lam I)^-1 b_S is worth 3.4-3.6x
given the true signature similarity (12.5 -> 44.4% top-1 on K562, 14.1 -> 48.3% on RPE1)
and only TRADES on any real representation (top-1 up 1.1-2.1x, median rank worse on 5 of
6 rows). `repr_curve.py --calibration-only` showed that standardising the oracle
similarity -- same ordering, different magnitudes -- costs that combiner 24% of its top-1
on K562 and 42% on RPE1 while kNN does not move. So the combiner reads MAGNITUDES, and
nothing in the chain so far ever estimated them.

The observation this script is built on. The combiner needs b_j = <y_j, y_g>. Every row in
GAPS plugs in

    b_j = s(g,j) * ||y_j||                                  [the plug-in]

where s is a cosine from some OTHER space (GO Jaccard, another cell line). For the
*oracle* source s is the true in-line cosine, so the plug-in equals

    b_j = cos(y_g,y_j) ||y_j|| = <y_j,y_g> / ||y_g||

-- the true b up to the per-row constant ||y_g||, which cancels in cos(y_hat, y_g). The
oracle row is therefore not "a better ranking": it is the combiner handed an EXACT b, and
the entire oracle-to-real gap is b-estimation error by construction. That reframes the
representation problem: its job is not to rank neighbours, it is to predict inner products.

Two things follow, and this script measures both.

PART 1 -- the decomposition that bounds the method, before trusting it.
Neighbourhood selection and b estimation are separable, so run the 2x2:

                        b = plug-in          b = true (cheats; diagnostic only)
    neighbourhood S      A = GAPS' row        B = how much is reachable WITHOUT
      from real repr                              a better ordering
    neighbourhood S      C                    D = the 44.4% / 48.3% oracle row
      from oracle

B is the quantity the hypothesis is about (cf. the blind-regime negative: log the
quantity, not only the metric). B ~ A means the neighbourhood binds and calibrating b
cannot help -- a refuted mechanism. B ~ D means b is the whole story and the method has
room. C == D exactly for the oracle source by the algebra above; that is asserted, not
assumed, as the reconciliation point for the whole script.

PART 2 -- the method: regress b instead of plugging it in.
For a training gene i the target is MEASURED -- cos(y_i^A, y_j^A) is a row of the atlas
Gram -- so the map (pair features) -> in-line cosine can be fitted on genes and applied
out of fold to a gene whose signature is unknown. Features use only quantities available
for an unseen gene: prior similarities, the other line's cosine and its coverage
indicator, each one's RANK within the row (so the model can use ordering as well as
value), the rank of g within j's column (a learned hubness correction rather than CSLS
imposed), and the atlas gene's norm and split-half reliability (cross-line cosines are
attenuated by the noise in whichever gene is measured). Training pairs are drawn from the
TOP of each gene's ordering plus a random tail, which is the top-k objective GAPS'
specification asked for instead of a fit to the whole similarity matrix.

Three nested variants, so a gain is attributable:
    plugin   b = s(g,j)||y_j||                      what GAPS reports
    iso      b = f(s(g,j))||y_j||, f isotonic       ORDERING FIXED, magnitudes only
    learned  b = t_hat(g,j)||y_j||, ridge on feats   ordering may change too
`iso` is the mirror image of the standardisation ablation, which removed calibration and
cost 24% / 42%. It is made STRICTLY monotone (plain isotonic regression has plateaus, and
those retie distinct similarities and change the neighbour set), so the genes it selects
are provably the source's own. Its kNN still moves a little, because kNN weights by the
similarity and the magnitudes are what this changes -- so the asserted invariant is the
neighbour set, not the kNN output.

Each variant is fitted twice -- on prior views only, and with the other line's measured
cosine added -- and reported on all genes, on the genes the other line actually measured,
and on the mean-imputed rest, because the aggregate mixes a measurement with a constant
(45% of RPE1) and that artefact is what made the cross-line row look bad on RPE1.

Same task, atlas, tie-aware metric and 5 gene folds as every number in GAPS.md; the
GO-kNN and learned-kNN rows are asserted against repr_select's saved per-gene ranks to
max |delta| = 0 before anything new is reported.

Writes <art>/repr_gram/{result_<dataset>.npz, summary.json, report.md}. Exits 0.

Usage: python -I repr_gram.py --art-dir D --gears-dir D --datasets A B [--knn 10]
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
M_GRID = (10, 25, 50, 100)
LAM_GRID = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)   # for the b-regression itself
TOP_TRAIN = 200        # training pairs from the top of each gene's ordering
TAIL_TRAIN = 200       # plus this many sampled from the rest of the row
BLOCK = 128            # query-gene rows per feature block


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ============================================================ the combiner
def gls_from_b(Snb, B, YA, G, m, lam):
    """y_hat_i = Y_S^T (G_SS + lam I)^-1 b_S, neighbourhood S = top m of Snb row i.

    Snb chooses WHICH atlas genes are in the neighbourhood; B supplies the inner
    products the solve is asked to match. Splitting them is the whole point: it is what
    makes `b` measurable separately from `ordering`. Identical to
    repr_select.predict_gls when B is the plug-in built from the same Snb."""
    K = Snb.shape[0]
    out = np.zeros_like(YA)
    mm = int(min(m, K - 1))
    top = np.argpartition(-Snb, mm - 1, axis=1)[:, :mm]
    ridge = lam * float(np.mean(np.diag(G)))
    eye = ridge * np.eye(mm)
    for i in range(K):
        idx = top[i]
        A = G[np.ix_(idx, idx)] + eye
        b = B[i, idx]
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        out[i] = w.astype(np.float32) @ YA[idx]
    return out


def neg_diag(S):
    """A float32 copy with the diagonal at -inf -- a gene is not its own neighbour.

    The dtype is part of the contract, not housekeeping: `np.where(eye, -np.inf, S)`
    promotes a float32 similarity to float64 because -np.inf is a Python float, and then
    knn_from_sim averages its neighbours in double precision. That shifted per-gene ranks
    by 0.25 against repr_select's saved result -- small, but it means the baseline row is
    not the same measurement GAPS.md reports. Mirror repr_select exactly instead."""
    out = np.array(S, dtype=np.float32, copy=True)
    np.fill_diagonal(out, -np.inf)
    return out


def plugin_b(S, nrm, clip=True):
    """b_j = s(g,j) * ||y_j||, negatives clipped to zero when clip -- what every GAPS row
    feeds the combiner (repr_probe.knn_from_sim and repr_select.predict_gls both clip,
    because a kNN is a weighted average and a negative weight is not one).

    The clip is a misspecification for the ridge solve, which is a regression and may
    legitimately want a negative coefficient. It also breaks an otherwise exact identity:
    UNCLIPPED, the oracle's plug-in b equals the measured b up to the per-row constant
    ||y_g||, which cancels in a cosine. So `clip=False` on the oracle is the exact-b row,
    and the gap between the two measures what clipping costs."""
    B = (np.maximum(S, 0.0) if clip else np.where(np.isfinite(S), S, 0.0)) * nrm[None, :]
    return np.where(np.isfinite(B), B, 0.0)


# ============================================================ tuning harness
def tune_pred(make_pred, grid, YA, Qu, own, K, folds, val_fold):
    """repr_select.tune, unchanged in protocol: one setting per fold, chosen on the genes
    of a DIFFERENT fold, so no gene is predicted by a setting selected on it."""
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
    for g in set(picked.values()):
        P = make_pred(g)
        for f, te in enumerate(folds):
            if picked[f] == g:
                out[te] = P[te]
        del P
    return out, picked


def evaluate(P, Qu, own, K):
    S = (Qu @ RH.unit(P).T).astype(np.float32)
    m, rank, rr = RP.score_matrix(S, own)
    gr, _ = RP.per_gene(rank, rr, own, K)
    return m, gr


# ============================================================ b regression
def ranks_in_rows(S):
    """1-based rank of each column within its row, descending. Ties broken by index,
    which is fine: this is a feature, not a reported metric."""
    order = np.argsort(-S, axis=1, kind="stable")
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(1, S.shape[1] + 1)[None, :].repeat(S.shape[0], 0), axis=1)
    return r.astype(np.float32)


class PairFeatures:
    """Pair features for (query gene g, atlas gene j), none of which touch y_g.

    `views` are the similarity matrices the arm is allowed to see. For each one the raw
    value, its rank within g's row, and its rank within j's column (the hubness term) go
    in; plus the atlas gene's norm and split-half reliability, and interactions of those
    with the strongest view, since a cross-line cosine is attenuated by the noise in the
    gene that was measured. A few hinges give the fit a shape rather than a slope."""

    HINGES = (0.05, 0.1, 0.2, 0.35, 0.5)

    def __init__(self, views, nrm, rel, cov, strong):
        self.names = list(views)
        self.V = {k: np.asarray(v, np.float32) for k, v in views.items()}
        self.R = {k: np.log1p(ranks_in_rows(v)) for k, v in self.V.items()}
        self.C = {k: np.log1p(ranks_in_rows(v.T).T) for k, v in self.V.items()}
        self.lognrm = np.log1p(nrm).astype(np.float32)
        self.rel = np.asarray(rel, np.float32)
        self.cov = np.asarray(cov, np.float32)
        self.strong = strong
        self.labels = (
            [f"{k}" for k in self.names]
            + [f"rank_{k}" for k in self.names]
            + [f"hub_{k}" for k in self.names]
            + ["lognrm_j", "rel_j", "cov_j", "cov_g", "cov_g_x_strong",
               "strong_x_rel_j", "strong_x_lognrm_j", "strong_sq"]
            + [f"hinge_{h}" for h in self.HINGES]
            + ["bias"])

    @property
    def dim(self):
        return len(self.labels)

    def block(self, rows, cols=None):
        """Design matrix for the cartesian product rows x cols (cols=None -> all)."""
        rows = np.asarray(rows, np.int64)
        K = self.lognrm.size
        cols = np.arange(K) if cols is None else np.asarray(cols, np.int64)
        n, k = rows.size, cols.size
        feats = []
        for nm in self.names:
            feats.append(self.V[nm][np.ix_(rows, cols)])
        for nm in self.names:
            feats.append(self.R[nm][np.ix_(rows, cols)])
        for nm in self.names:
            feats.append(self.C[nm][np.ix_(rows, cols)])
        st = self.V[self.strong][np.ix_(rows, cols)]
        feats += [np.broadcast_to(self.lognrm[cols][None, :], (n, k)),
                  np.broadcast_to(self.rel[cols][None, :], (n, k)),
                  np.broadcast_to(self.cov[cols][None, :], (n, k)),
                  np.broadcast_to(self.cov[rows][:, None], (n, k)),
                  self.cov[rows][:, None] * st,
                  st * self.rel[cols][None, :],
                  st * self.lognrm[cols][None, :],
                  st * st]
        for h in self.HINGES:
            feats.append(np.maximum(st - h, 0.0))
        feats.append(np.ones((n, k), np.float32))
        X = np.stack([np.asarray(f, np.float32).reshape(-1) for f in feats], 1)
        return X


def training_pairs(rows, S_strong, K, rng):
    """Pairs from the TOP of each training gene's ordering plus a random tail.

    A uniform sample over all K^2 pairs is dominated by far pairs the combiner never
    reads; GAPS' specification asked for a top-k objective, so the sampler implements it
    and keeps a tail so the fit still knows what "far" looks like."""
    a, b = [], []
    tt = int(min(TOP_TRAIN, K - 1))
    top = np.argpartition(-S_strong[rows], tt - 1, axis=1)[:, :tt]
    for n, i in enumerate(rows):
        j = top[n]
        j = j[j != i]
        tail = rng.choice(K, size=TAIL_TRAIN, replace=False)
        tail = tail[(tail != i) & ~np.isin(tail, j)]
        both = np.concatenate([j, tail])
        a.append(np.full(both.size, i, np.int64))
        b.append(both)
    return np.concatenate(a), np.concatenate(b)


def fit_ridge(X, y, lam):
    """Standardised ridge, closed form. The bias column is left unpenalised by centring
    the target instead of fitting through the origin."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd
    ym = float(y.mean())
    A = Z.T @ Z + lam * Z.shape[0] * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - ym))
    return {"w": w, "mu": mu, "sd": sd, "ym": ym}


def apply_ridge(mdl, X):
    return ((X - mdl["mu"]) / mdl["sd"]) @ mdl["w"] + mdl["ym"]


def learned_tmatrix(PF, Csig, folds, val_fold, K, S_strong, rng):
    """Out-of-fold estimate of the in-line cosine for every (g, j) pair.

    For fold f the model is fitted on pairs whose QUERY gene lies outside f, with the
    ridge strength chosen on a different fold again (val_fold[f]), and then applied to
    the genes of f. A gene is therefore never scored by a model that saw its own
    signature in the target, and never by a ridge strength tuned on it."""
    T = np.zeros((K, K), np.float32)
    chosen, r2 = {}, {}
    for f, te in enumerate(folds):
        va = folds[val_fold[f]]
        tr = np.setdiff1d(np.arange(K), np.union1d(te, va))
        ta, tb = training_pairs(tr, S_strong, K, rng)
        Xtr = pair_design(PF, ta, tb)
        ytr = Csig[ta, tb].astype(np.float64)
        va_a, va_b = training_pairs(va, S_strong, K, rng)
        Xva, yva = pair_design(PF, va_a, va_b), Csig[va_a, va_b].astype(np.float64)
        best, bl, bmdl = -np.inf, RIDGE_GRID[0], fit_ridge(Xtr, ytr, RIDGE_GRID[0])
        for lam in RIDGE_GRID:
            mdl = fit_ridge(Xtr, ytr, lam)
            p = apply_ridge(mdl, Xva)
            ss = 1.0 - float(((yva - p) ** 2).sum() / max(((yva - yva.mean()) ** 2).sum(), 1e-12))
            if ss > best:
                best, bl, bmdl = ss, lam, mdl
        chosen[f], r2[f] = bl, best
        for s in range(0, te.size, BLOCK):
            rows = te[s:s + BLOCK]
            T[rows] = apply_ridge(bmdl, PF.block(rows)).reshape(rows.size, K).astype(np.float32)
    return T, chosen, r2


def pair_design(PF, a, b):
    """Design matrix for explicit pair lists (not a cartesian product)."""
    feats = []
    for nm in PF.names:
        feats.append(PF.V[nm][a, b])
    for nm in PF.names:
        feats.append(PF.R[nm][a, b])
    for nm in PF.names:
        feats.append(PF.C[nm][a, b])
    st = PF.V[PF.strong][a, b]
    feats += [PF.lognrm[b], PF.rel[b], PF.cov[b], PF.cov[a], PF.cov[a] * st,
              st * PF.rel[b], st * PF.lognrm[b], st * st]
    for h in PF.HINGES:
        feats.append(np.maximum(st - h, 0.0))
    feats.append(np.ones(a.size, np.float32))
    return np.stack([np.asarray(f, np.float64) for f in feats], 1)


def isotonic_tmatrix(S, Csig, folds, val_fold, K):
    """Out-of-fold STRICTLY monotone recalibration of one similarity into a cosine.

    Isotonic regression is monotone NON-decreasing: it maps whole ranges of input onto
    plateaus. Used raw that is not an order-preserving recalibration at all -- every
    distinct similarity inside a plateau becomes tied, the top-k selection among the ties
    is then arbitrary, and the neighbour set changes wholesale (measured: it moved GO's
    kNN by 642 rank positions). So the plateaus are broken by the SOURCE similarity, with
    a step small enough never to cross a real gap between distinct isotonic outputs. The
    result is strictly increasing in S, hence the top-k neighbour SET is unchanged.

    Note what this does and does not fix. The neighbour set is preserved; the kNN
    predictor still moves a little, because kNN uses the similarities as WEIGHTS and this
    deliberately changes their magnitudes. There is no exact kNN identity to assert here
    -- repr_curve's "kNN unchanged to 0.05 points" under standardisation was an empirical
    near-invariance, not an identity. What is asserted downstream is the invariant that
    actually holds: the selected neighbours are the same genes."""
    from sklearn.isotonic import IsotonicRegression
    # float64, not float32. The plateau-breaking term is small by design, and in float32
    # two source values 5e-05 apart round to the same stored value and are tied again --
    # which is exactly how this failed on the finely spaced cross-line kernel after it
    # had passed on GO Jaccard, whose distinct values are ratios of small integers and so
    # are far apart. Storing the similarity in double costs 19 MB at K=1543.
    T = np.zeros((K, K), np.float64)
    for f, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(K), te)
        ii, jj = np.meshgrid(tr, tr, indexing="ij")
        keep = ii != jj
        x, y = S[ii[keep], jj[keep]], Csig[ii[keep], jj[keep]]
        sub = np.random.default_rng(SEED + f).choice(x.size, size=min(400_000, x.size),
                                                     replace=False)
        ir = IsotonicRegression(out_of_bounds="clip").fit(x[sub].astype(np.float64),
                                                          y[sub].astype(np.float64))
        blk = ir.predict(S[te].reshape(-1).astype(np.float64)).reshape(te.size, K)
        src = S[te].astype(np.float64)
        lo, hi = float(np.min(src)), float(np.max(src))
        # iso(s) + c*s is strictly increasing in s for ANY c > 0, so the coefficient
        # cannot reorder anything -- it only sets how much of the shape stays isotonic.
        # Size it to the OUTPUT range, not to the smallest gap between isotonic outputs:
        # that gap can be ~1e-7 over 400k points, and a step that small is erased by the
        # float32 store, which leaves the plateaus intact. GO Jaccard is a ratio of small
        # integers and is massively tied, so surviving plateaus retie distinct neighbours
        # and the selected set moves (measured: 0.26 in source similarity).
        rng_out = float(np.max(blk) - np.min(blk))
        step = 1e-3 * max(rng_out, 1e-6) / max(hi - lo, 1e-12)
        T[te] = blk + step * (src - lo)
    return T


def topk_source_values(S, key, k):
    """The SOURCE similarities of the k neighbours a given key selects, sorted.

    Comparing these instead of the index sets lets two genuinely tied source values swap
    places (argpartition's choice among equals is arbitrary) while still catching any
    change in WHICH similarities were selected."""
    kk = int(min(k, S.shape[1] - 1))
    idx = np.argpartition(-key, kk - 1, axis=1)[:, :kk]
    return np.sort(np.take_along_axis(S, idx, axis=1), axis=1)


# ============================================================ driver
def coverage_mask(priors_path, cands, meas_name):
    """Which genes the other line actually measured. `related_work.aligned_feat`
    mean-imputes a gene the source lacks, so every imputed row is identical; a row that
    appears more than once is imputed. Same rule as repr_figs.paired_stats, so the strata
    here and the ones already in GAPS.md are the same split."""
    z = np.load(priors_path, allow_pickle=False)
    pnames = [str(x) for x in z["names"]]
    if [str(x) for x in z["cands"]] != [str(x) for x in cands]:
        raise RuntimeError("prior kernels are not aligned to this cache's candidates")
    R = z["kernels"][pnames.index(meas_name)]
    _, inv, cnt = np.unique(np.round(R, 5), axis=0, return_inverse=True, return_counts=True)
    return ~(cnt[inv] > 1)


def strata_of(cov):
    return [("all", np.ones_like(cov, bool)), ("xline_measured", cov), ("mean_imputed", ~cov)]


def metrics_by_stratum(gr, cov):
    """Per-gene rank -> the reported numbers, on each stratum. Per-gene (a gene's ~1.8
    queries averaged first), which is the unit the paired tests use."""
    out = {}
    for nm, sel in strata_of(cov):
        if sel.sum() == 0:
            continue
        r = gr[sel]
        out[nm] = {"n_genes": int(sel.sum()), "top1": float((r <= 1).mean()),
                   "top10": float((r <= 10).mean()), "median_rank": float(np.median(r)),
                   # 1/(a gene's mean query rank), averaged over genes. NOT the per-query
                   # MRR the other tables report -- different unit, so differently named.
                   "mean_inverse_gene_rank": float(np.mean(1.0 / r))}
    return out


def paired_vs(gr, base, cov):
    from scipy.stats import wilcoxon
    out = {}
    for nm, sel in strata_of(cov):
        if sel.sum() == 0:
            continue
        d = base[sel] - gr[sel]
        nz = d[d != 0]
        try:
            _, pv = wilcoxon(nz) if nz.size else (float("nan"), 1.0)
        except Exception:
            pv = float("nan")
        out[nm] = {"n_genes": int(sel.size), "wins": int((d > 0).sum()),
                   "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                   "win_rate_excl_ties": (float((d > 0).sum() / (d != 0).sum())
                                          if (d != 0).sum() else float("nan")),
                   "wilcoxon_p": float(pv)}
    return out


def b_quality(T, B_true, S_nb, m=50):
    """How good is the b estimate itself, where the combiner reads it?

    The quantity the hypothesis is about. Correlation is computed WITHIN each gene's
    neighbourhood, because that is the only place the solve looks, and a global
    correlation is the statistic this project already showed to be misleading.

    Also reports the global R^2 and the R^2 after the row and column means are removed.
    The b-regression scores a healthy global R^2 (~0.33 on prior views alone) while
    moving retrieval not at all, and the split says why: a gene's overall similarity
    LEVEL is predictable -- some genes resemble everything, and an atlas gene's norm and
    reliability set the scale -- but the combiner is invariant to a per-row constant and
    reads only the CONTRAST between a gene's own neighbours. Quoting the global number
    beside a flat metric would be the same mistake as reading these curves on Spearman."""
    K = T.shape[0]
    mm = int(min(m, K - 1))
    top = np.argpartition(-S_nb, mm - 1, axis=1)[:, :mm]
    pear, spear = [], []
    for i in range(K):
        idx = top[i]
        a, b = T[i, idx].astype(np.float64), B_true[i, idx].astype(np.float64)
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        pear.append(float(np.corrcoef(a, b)[0, 1]))
        spear.append(RP.spearman(a, b))

    # The same correlation over a gene's WHOLE row, not just its neighbourhood. Per row,
    # so it is invariant to the per-row factor ||y_g|| that b carries and the estimate
    # does not -- which is also why a pair-level global R^2 is not the comparison to make
    # here, and why the regression's own validation R^2 must be read beside this rather
    # than instead of it.
    full = []
    for i in range(K):
        sel = np.isfinite(T[i]) & (np.arange(K) != i)
        a, b = T[i, sel].astype(np.float64), B_true[i, sel].astype(np.float64)
        if a.std() > 1e-12 and b.std() > 1e-12:
            full.append(float(np.corrcoef(a, b)[0, 1]))
    return {"n": len(pear), "pearson_in_neighbourhood": float(np.mean(pear)),
            "spearman_in_neighbourhood": float(np.nanmean(spear)), "m": mm,
            "pearson_full_row": float(np.mean(full)) if full else float("nan")}


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
    YB = D["dB"][:, mk].astype(np.float32)
    del D
    YAu, YBu = RH.unit(YA), RH.unit(YB)
    Csig = (YAu @ YAu.T).astype(np.float32)
    rel = np.sum(YAu * YBu, 1).astype(np.float32)        # split-half reliability per gene
    G = (YA.astype(np.float64) @ YA.astype(np.float64).T)
    nrm = np.sqrt(np.maximum(np.diag(G), 1e-24))
    B_true = G.copy()                                    # b_ij = <y_i, y_j>, measured
    np.fill_diagonal(B_true, 0.0)

    Qu = RH.unit(W.Q)
    own = W.owner.astype(np.int64)
    folds = RH.folds_of(K, np.random.default_rng(SEED), k=N_FOLDS)
    val_fold = {f: (f + 1) % N_FOLDS for f in range(N_FOLDS)}
    log(f"  reliability: median cos(dA,dB) = {float(np.median(rel)):.3f}")

    # ---------------------------------------------------- similarity sources
    Jg = W.JJ.copy()
    np.fill_diagonal(Jg, 0.0)
    views = {"go_jaccard": Jg.astype(np.float32)}
    rich = W.rich()
    views["go_idf"] = rich["idf"].astype(np.float32)
    views["go_emb"] = rich["emb"].astype(np.float32)
    z = np.load(priors_path, allow_pickle=False)
    pnames = [str(x) for x in z["names"]]
    if [str(x) for x in z["cands"]] != [str(x) for x in W.cands]:
        raise RuntimeError("prior kernels are not aligned to this cache's candidates")
    prior_views, meas_views = [], []
    for i, nm in enumerate(pnames):
        v = RD.cosnorm(z["kernels"][i].astype(np.float64))
        np.fill_diagonal(v, 0.0)
        views[nm] = v
        (meas_views if nm in RD.MEASUREMENT_KERNELS else prior_views).append(nm)
    if not meas_views:
        raise RuntimeError("no cross-line measurement kernel for this dataset")
    xname = meas_views[0]

    names = ["go_jaccard", "go_idf", "go_emb"] + prior_views
    Slearn, _ = RD.learned_similarity([views[n] for n in names], Csig, folds, K,
                                      np.random.default_rng(SEED))
    cov = coverage_mask(priors_path, W.cands, xname)
    log(f"  {xname}: the other line measured {int(cov.sum())}/{K} genes "
        f"({cov.mean()*100:.1f}%); the rest are mean-imputed")

    # ------------------------------------------------- reconcile at a known point
    # Before any new number, the two rows this script shares with repr_select must come
    # out bit-identical. Two silent table-changing bugs were caught this way before.
    recon = {}
    sel_path = os.path.join(os.path.dirname(out_dir), "repr_select", f"result_{ds}.npz")
    gr_go = evaluate(RP.knn_from_sim(neg_diag(Jg), YA, knn), Qu, own, K)[1]
    gr_le = evaluate(RP.knn_from_sim(neg_diag(Slearn), YA, knn), Qu, own, K)[1]
    if os.path.exists(sel_path):
        zs = np.load(sel_path)
        for key, got in (("go__knn_avg", gr_go), ("learned__knn_avg", gr_le)):
            want = zs[f"rank_{key}"]
            dv = np.abs(want - got)
            nd, d = int((dv > 0).sum()), float(dv.max())
            recon[key] = {"n_genes_differing": nd, "max_abs_delta": d,
                          "genes": [str(W.cands[i]) for i in np.where(dv > 0)[0][:8]]}
            # Not bit-equality. A gene's rank is tie-aware (g + 1 + e/2) and score_matrix
            # rounds the scores to 6 decimals, so a ~1e-7 float difference between two
            # PROCESSES flips one tie and moves that gene's mean rank by 0.5/nq. Measured
            # 2026-11-23: it is not the node or the thread count -- an 8-thread rerun on
            # the same node as the saved run reproduces the live value, not the saved one;
            # scripts/repr_recon_check.py shows the gene sits on a rounding boundary. Measured here: exactly one gene, CCT3, a chaperonin subunit
            # with near-duplicate complex partners in the atlas, which is the class this
            # project already reports as having a competing near-duplicate. The project's
            # own rule for this is to re-score in one environment, so the LIVE row is the
            # baseline for every comparison below and this check is provenance only. The
            # bound still catches a real bug: that would move many genes, or move the
            # reported metrics, which are asserted exactly just below.
            if nd > max(2, int(0.005 * K)) or d > 1.0:
                raise RuntimeError(
                    f"{key} differs from repr_select on {nd} genes (max|delta|={d}); "
                    "more than tie-boundary noise, so the task or predictor has changed")
        # The numbers GAPS.md actually prints must be identical, not merely close.
        for key, got in (("go__knn_avg", gr_go), ("learned__knn_avg", gr_le)):
            w = zs[f"rank_{key}"]
            for nm, f in (("top1", lambda r: float((r <= 1).mean())),
                          ("median_rank", lambda r: float(np.median(r)))):
                if abs(f(w) - f(got)) > 1e-9:
                    raise RuntimeError(f"{key} {nm} does not reproduce repr_select: "
                                       f"{f(w)} vs {f(got)}")
        log(f"  reconciled with repr_select (tie-boundary genes only): {recon}")
    else:
        recon["note"] = "repr_select result absent; nothing to reconcile against"
        log("  WARNING: no repr_select result to reconcile against")

    # ---------------------------------------------------------------- part 1
    # The 2x2. `oracle` with the plug-in b IS the true b up to a per-row constant, and
    # the constant cancels in a cosine -- so plug-in and true must agree there exactly.
    # That identity is the reconciliation point for the whole decomposition.
    sources = {"go": neg_diag(Jg), "learned": neg_diag(Slearn),
               "xline": neg_diag(views[xname]), "oracle": neg_diag(Csig)}

    res, granks, diag, warn = {}, {}, {}, []
    for sname, S in sources.items():
        Bp = plugin_b(S, nrm)
        for bname, B in (("plugin", Bp), ("plugin_noclip", plugin_b(S, nrm, clip=False)),
                         ("true_b", B_true)):
            grid = [(m, l) for m in M_GRID for l in LAM_GRID]
            t0 = time.time()
            P, picked = tune_pred(lambda p, S=S, B=B: gls_from_b(S, B, YA, G, p[0], p[1]),
                                  grid, YA, Qu, own, K, folds, val_fold)
            m, gr = evaluate(P, Qu, own, K)
            key = f"{sname}__gls_{bname}"
            granks[key] = gr
            res[key] = {"metrics": m, "by_stratum": metrics_by_stratum(gr, cov),
                        "chosen_per_fold": {str(k): str(v) for k, v in picked.items()},
                        "secs": round(time.time() - t0, 1)}
            log(f"  {key:26s} top1={m['top1']:.4f} top10={m['top10']:.4f} "
                f"medrank={m['median_rank']:.0f} [{res[key]['secs']}s]")
        diag[f"{sname}__plugin"] = b_quality(Bp, B_true, S)
    # The exact-b identity, asserted on the UNCLIPPED oracle row: b = cos(g,j)||y_j|| is
    # the measured <y_j,y_g> divided by ||y_g||, a per-row constant that cancels in a
    # cosine, so the two rows must agree to floating point. This is the reconciliation
    # point for the whole decomposition -- if it drifts, gls_from_b is wrong.
    d = abs(res["oracle__gls_plugin_noclip"]["metrics"]["top1"]
            - res["oracle__gls_true_b"]["metrics"]["top1"])
    dr = float(np.abs(granks["oracle__gls_plugin_noclip"] - granks["oracle__gls_true_b"]).max())
    if d > 1e-3:
        raise RuntimeError("the unclipped oracle plug-in must equal the measured b up to a "
                           f"per-row constant, which cancels in a cosine; top-1 differs by {d:.4f}")
    log(f"  exact-b identity holds: unclipped oracle == measured b "
        f"(top-1 differs {d:.6f}, max per-gene rank delta {dr:.3f})")
    recon["oracle_exact_b_identity_top1_delta"] = d
    recon["oracle_exact_b_identity_rank_delta"] = dr

    # ---------------------------------------------------------------- part 2
    # Two arms: prior views only, and with the other line's measured cosine added.
    arms = {"prior": (["go_jaccard", "go_idf", "go_emb"] + prior_views, "go_jaccard"),
            "xline": (["go_jaccard", "go_idf", "go_emb"] + prior_views + [xname], xname)}
    for arm, (vnames, strong) in arms.items():
        vv = {n: views[n] for n in vnames}
        # zero, not -inf: these are regression features, and a -inf would poison the
        # row-rank features for every column. The diagonal pair is never trained on.
        sl = np.array(Slearn, dtype=np.float32, copy=True)
        np.fill_diagonal(sl, 0.0)
        vv["learned"] = sl
        PF = PairFeatures(vv, nrm, rel, cov.astype(np.float32), strong)
        T, chosen, r2 = learned_tmatrix(PF, Csig, folds, val_fold, K, views[strong],
                                        np.random.default_rng(SEED + 7))
        np.fill_diagonal(T, -np.inf)
        log(f"  [{arm}] b-regression: {PF.dim} features, out-of-fold R^2 "
            f"{ {f: round(v, 4) for f, v in r2.items()} }, ridge {chosen}")

        Tiso = isotonic_tmatrix(views[strong], Csig, folds, val_fold, K)
        np.fill_diagonal(Tiso, -np.inf)

        for vname, Tm in (("iso", Tiso), ("learned", T)):
            Bh = plugin_b(Tm, nrm)
            # kNN on the same estimate: separates "the neighbours changed" from
            # "the magnitudes changed". For `iso` the neighbour SET is fixed by
            # construction, so its kNN move is purely the weights.
            Pk = RP.knn_from_sim(Tm, YA, knn)
            mk_, grk = evaluate(Pk, Qu, own, K)
            kk = f"{arm}_{vname}__knn_avg"
            granks[kk], res[kk] = grk, {"metrics": mk_, "by_stratum": metrics_by_stratum(grk, cov),
                                        "chosen_per_fold": {}, "secs": 0.0}
            t0 = time.time()
            P, picked = tune_pred(lambda p, S=Tm, B=Bh: gls_from_b(S, B, YA, G, p[0], p[1]),
                                  [(m, l) for m in M_GRID for l in LAM_GRID],
                                  YA, Qu, own, K, folds, val_fold)
            m, gr = evaluate(P, Qu, own, K)
            kg = f"{arm}_{vname}__gls"
            granks[kg] = gr
            res[kg] = {"metrics": m, "by_stratum": metrics_by_stratum(gr, cov),
                       "chosen_per_fold": {str(k): str(v) for k, v in picked.items()},
                       "secs": round(time.time() - t0, 1),
                       "b_ridge_per_fold": {str(k): v for k, v in chosen.items()},
                       "b_regression_val_r2": {str(k): v for k, v in r2.items()}}
            diag[f"{arm}_{vname}"] = b_quality(Bh, B_true, sources[strong_src(strong, xname)])
            log(f"  {kk:26s} top1={mk_['top1']:.4f} medrank={mk_['median_rank']:.0f}")
            log(f"  {kg:26s} top1={m['top1']:.4f} top10={m['top10']:.4f} "
                f"medrank={m['median_rank']:.0f} [{res[kg]['secs']}s]")
        # What a strictly monotone recalibration must preserve is the neighbour SET, not
        # the kNN output -- kNN weights by the similarity, and the magnitudes are exactly
        # what this changes. So assert the invariant that holds, and record how far the
        # kNN moved as a measurement rather than a check. If the SET moves, the `iso` row
        # cannot be read as "calibration only" and must not be reported that way.
        Ssrc = neg_diag(views[strong])
        dset = float(np.abs(topk_source_values(Ssrc, Tiso, knn)
                            - topk_source_values(Ssrc, Ssrc, knn)).max())
        base_knn = evaluate(RP.knn_from_sim(Ssrc, YA, knn), Qu, own, K)[1]
        dk = float(np.abs(granks[f"{arm}_iso__knn_avg"] - base_knn).max())
        ok = dset <= 1e-6
        for key in (f"{arm}_iso__knn_avg", f"{arm}_iso__gls"):
            res[key]["magnitudes_only"] = bool(ok)
            res[key]["max_abs_delta_topk_source_values"] = dset
        res[f"{arm}_iso__knn_avg"]["max_abs_delta_vs_source_knn"] = dk
        if not ok:
            # Record, do not raise. `iso` is a CONTROL; part 1 and the method rows are the
            # measurement, and they are already computed by this point. An earlier version
            # raised here and threw away a whole dataset's primary results because a
            # control's invariant was violated. The flag is what protects the write-up:
            # a row with magnitudes_only false must not be described as calibration-only.
            warn.append(f"{ds}/{arm}: monotone recalibration of {strong} moved the "
                        f"selected neighbours by {dset:.2e}; its iso rows are NOT a "
                        f"magnitudes-only ablation and must not be reported as one")
            log(f"  WARNING: iso on {strong} moved the neighbour set by {dset:.2e} -- "
                f"iso rows flagged magnitudes_only=False, method rows unaffected")
        else:
            log(f"  iso keeps the {strong} neighbour set (max|delta|={dset:.2e}); "
                f"its kNN moves {dk:.3f} rank, which is the weights changing, as expected")

    # ------------------------------------------------------------- references
    granks["go__knn_avg"] = gr_go
    res["go__knn_avg"] = {"metrics": evaluate(RP.knn_from_sim(neg_diag(Jg), YA, knn),
                                              Qu, own, K)[0],
                          "by_stratum": metrics_by_stratum(gr_go, cov),
                          "chosen_per_fold": {}, "secs": 0.0}
    Pg, _ = RP.greedy_reconstruct(G, YA, k=knn)
    for nm, P in (("ceiling__oracle_greedy", Pg), ("ceiling__measured", YA)):
        m, gr = evaluate(P, Qu, own, K)
        granks[nm] = gr
        res[nm] = {"metrics": m, "by_stratum": metrics_by_stratum(gr, cov),
                   "chosen_per_fold": {}, "secs": 0.0}

    # --------------------------------------------------------- paired tests
    base = granks["go__knn_avg"]
    paired = {k: paired_vs(v, base, cov) for k, v in granks.items() if k != "go__knn_avg"}

    np.savez_compressed(os.path.join(out_dir, f"result_{ds}.npz"),
                        cands=np.array([str(x) for x in W.cands]),
                        xline_covered=cov, reliability=rel,
                        **{f"rank_{k}": v.astype(np.float32) for k, v in granks.items()})
    return {"dataset": ds, "K": int(K), "knn": int(knn), "n_queries": int(Qu.shape[0]),
            "measurement_view": xname, "xline_covered": int(cov.sum()),
            "reconciliation_max_abs_delta": recon, "results": res,
            "b_quality": diag, "paired_vs_go_knn": paired,
            "interpretation_warnings": warn}


def write_report(summary, out_dir):
    """report.md from summary.json, so a presentation fix needs no recompute.

    TWO units of top-1, labelled, because they differ by up to 9 points on the same row:
    PER QUERY is the unit of every table in GAPS.md (a query is top-1 if its own gene
    ranks first); PER GENE averages a gene's ~1.8 queries first and counts the gene only
    if that mean rank is 1, which is stricter and is the unit the paired tests and the
    coverage strata use. An earlier version of this report printed only the per-gene
    number under "top-1 all", so its 35.3% for the oracle row read as a contradiction of
    GAPS.md's 44.4% for the same measurement."""
    L = ["# The combiner's input is an inner product: estimate it instead of plugging it in",
         "",
         "`gls_plugin` is the `gls` row GAPS.md reports, b_j = s(g,j)||y_j|| (its per-query",
         "top-1 reproduces repr_select's exactly). `gls_true_b` feeds the same neighbourhood",
         "the MEASURED inner product -- it cheats, and bounds what fixing b alone can buy.",
         "`iso` recalibrates the similarity strictly monotonically, so it keeps the",
         "neighbours and changes only their magnitudes; `learned` regresses the in-line",
         "cosine on pair features, out of fold over genes. `plugin_noclip` keeps negative",
         "similarities instead of clipping them.",
         "",
         "**Two units of top-1.** *Per query* is GAPS.md's unit. *Per gene* averages a",
         "gene's ~1.8 queries first and is stricter; it is the unit of the strata and the",
         "paired tests. Do not compare a per-gene number with a GAPS.md table.",
         ""]
    for ds, s in summary["datasets"].items():
        L += [f"## {ds} (K={s['K']}, {s['n_queries']} queries, "
              f"{s['xline_covered']} genes measured in the other line)", ""]
        L += ["| row | top-1 **per query** | top-1 per gene, all | per gene, xline-measured "
              "| per gene, imputed | median rank per gene |",
              "|---|---:|---:|---:|---:|---:|"]
        for k, v in s["results"].items():
            st = v["by_stratum"]
            def g(n, f):
                return f"{st[n][f]*100:.1f}%" if n in st else "—"
            L.append(f"| `{k}` | {v['metrics']['top1']*100:.1f}% | {g('all','top1')} "
                     f"| {g('xline_measured','top1')} | {g('mean_imputed','top1')} "
                     f"| {st['all']['median_rank']:.0f} |")
        L.append("")
        for w in s.get("interpretation_warnings", []):
            L += [f"> **Caveat.** {w}", ""]
        L += ["b-quality inside the neighbourhood the combiner reads (m=50):", "",
              "| b estimate | Pearson in nbhd | Spearman in nbhd | Pearson whole row |",
              "|---|---:|---:|---:|"]
        for k, v in s["b_quality"].items():
            L.append(f"| `{k}` | {v['pearson_in_neighbourhood']:.3f} "
                     f"| {v['spearman_in_neighbourhood']:.3f} "
                     f"| {v['pearson_full_row']:.3f} |")
        L.append("")
    if summary["errors"]:
        L += ["## Errors", ""] + [f"- {e['dataset']}: {e['error']}" for e in summary["errors"]]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def strong_src(strong, xname):
    """Which part-1 neighbourhood the b-quality diagnostic should be read inside."""
    return "xline" if strong == xname else "go"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild report.md from an existing summary.json; no computation")
    a = ap.parse_args()

    out_dir = os.path.join(a.art_dir, "repr_gram")
    os.makedirs(out_dir, exist_ok=True)
    if a.report_only:
        write_report(json.load(open(os.path.join(out_dir, "summary.json"))), out_dir)
        log("rebuilt", os.path.join(out_dir, "report.md"), "from summary.json")
        return 0
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
                                      "traceback": traceback.format_exc()[-3000:]})
            log(f"{ds}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc()

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_report(summary, out_dir)
    log("wrote", os.path.join(out_dir, "report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
