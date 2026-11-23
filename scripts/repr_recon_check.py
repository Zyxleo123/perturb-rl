#!/usr/bin/env python
"""Bisect why repr_gram's GO-kNN row misses repr_select's saved one by 0.25.

Builds the row BOTH ways in one process -- repr_select's exact statement sequence, and
repr_gram's -- and compares the intermediates in order (similarity, prediction, query
scores, per-query rank, per-gene rank), so the first stage that differs names the cause
instead of being guessed at. Also compares both against the saved artifact, since a
difference against disk and a difference between the two code paths have different
explanations. Read-only: writes nothing, exits 0.

Usage: python -I repr_recon_check.py --art-dir D --gears-dir D --dataset NAME
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_headroom as RH          # noqa: E402
import rl_acquire as RA           # noqa: E402
import repr_probe as RP           # noqa: E402
import repr_select as RS          # noqa: E402
import repr_gram as RG            # noqa: E402


def describe(tag, a, b):
    a, b = np.asarray(a), np.asarray(b)
    same = a.shape == b.shape and a.dtype == b.dtype
    fa, fb = a.astype(np.float64), b.astype(np.float64)
    ok = np.isfinite(fa) & np.isfinite(fb)          # -inf diagonals subtract to nan
    d = float(np.abs(fa[ok] - fb[ok]).max()) if a.shape == b.shape and ok.any() else float("nan")
    n = int((a != b).sum()) if a.shape == b.shape else -1
    print(f"  {tag:34s} dtype {str(a.dtype):8s}/{str(b.dtype):8s} "
          f"shape {str(a.shape):14s}/{str(b.shape):14s} max|d|={d:.3e} ndiff={n}"
          + ("" if same else "   <-- SHAPE/DTYPE DIFFER"))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--knn", type=int, default=10)
    a = ap.parse_args()
    ds, knn = a.dataset, a.knn

    import json
    cc = json.load(open(os.path.join(a.art_dir, "rl_acquire", "cache_check.json")))
    W = RA.World(cc["datasets"][ds]["path"])
    K = W.K
    D = RH.build(ds, os.path.join(a.gears_dir, ds, "perturb_processed.h5ad"),
                 np.random.default_rng(RH.SEED))
    mk = D["mask"]
    YA = D["dA"][:, mk].astype(np.float32)
    Qu = RH.unit(W.Q)
    own = W.owner.astype(np.int64)

    # ---- repr_select's sequence, statement for statement
    Jg_s = W.JJ.copy()
    np.fill_diagonal(Jg_s, 0.0)
    S_sel = Jg_s.astype(np.float32)
    np.fill_diagonal(S_sel, -np.inf)
    P_sel = RS.predict_knn(S_sel, YA, knn)
    Sq_sel = (Qu @ RH.unit(P_sel).T).astype(np.float32)
    m_sel, rank_sel, rr_sel = RP.score_matrix(Sq_sel, own)
    gr_sel = RP.per_gene(rank_sel, rr_sel, own, K)[0]

    # ---- repr_gram's sequence
    Jg_g = W.JJ.copy()
    np.fill_diagonal(Jg_g, 0.0)
    S_gram = RG.neg_diag(Jg_g)
    P_gram = RP.knn_from_sim(S_gram, YA, knn)
    Sq_gram = (Qu @ RH.unit(P_gram).T).astype(np.float32)
    m_gram, rank_gram, rr_gram = RP.score_matrix(Sq_gram, own)
    gr_gram = RP.per_gene(rank_gram, rr_gram, own, K)[0]

    print(f"\n{ds}: K={K}, knn={knn}, nq={Qu.shape[0]}")
    print("\nrepr_select path vs repr_gram path, stage by stage:")
    describe("similarity S", S_sel, S_gram)
    describe("prediction P", P_sel, P_gram)
    describe("query scores Qu@unit(P).T", Sq_sel, Sq_gram)
    describe("per-query rank", rank_sel, rank_gram)
    d_live = describe("per-gene rank", gr_sel, gr_gram)
    print(f"  top1 {m_sel['top1']:.6f} / {m_gram['top1']:.6f}   "
          f"medrank {m_sel['median_rank']:.1f} / {m_gram['median_rank']:.1f}")

    # ---- both against the saved artifact
    p = os.path.join(a.art_dir, "repr_select", f"result_{ds}.npz")
    saved = np.load(p)["rank_go__knn_avg"]
    print(f"\nagainst the saved artifact {os.path.basename(p)}:")
    d_sel = describe("saved vs repr_select path", saved, gr_sel)
    d_gram = describe("saved vs repr_gram path", saved, gr_gram)

    # Is each differing gene sitting on a rounding boundary? score_matrix rounds scores to
    # 6 decimals before ranking, so a query is fragile when some competitor's RAW score is
    # within the rounding resolution of the query's own-candidate score: a last-bit
    # difference between two processes can then create or break a tie (rank g+1+e/2).
    # This is measured in ONE process, so it does not need the flip to reproduce.
    #
    # The criterion is NOT "the two raw scores are within RES now" -- that asks whether
    # they share a rounding bin in this run, and two scores in different bins can still be
    # tied in another run. A perturbation delta can create a tie only if gap < RES + delta,
    # where delta is the float32 difference two processes can produce on that score: a few
    # ulps at its magnitude. The first version used gap < RES, and missed CCT3 query 128
    # (gap 1.01e-6 to CCT6A, i.e. 1e-8 outside a bin, under one ulp). The tolerance is
    # derived from float32 precision, not tuned to that case, and the margin is printed in
    # ulps so a reader can judge it.
    RES = 1e-6
    ULPS = 4
    diff_genes = np.where(saved != gr_sel)[0]
    fragile = {}
    for i in diff_genes:
        rows = []
        for q in np.where(own == i)[0]:
            raw = Sq_sel[q].astype(np.float64)
            gap = np.abs(raw - raw[i])
            gap[i] = np.inf
            j = int(np.argmin(gap))
            ulp = float(np.spacing(np.float32(abs(Sq_sel[q, i]))))
            rows.append((int(q), float(gap[j]), str(W.cands[j]), ulp))
        fragile[int(i)] = rows
    if fragile:
        print(f"\n  nearest competitor to the true candidate, per query (rounding resolution {RES:g}):")
        for i, rows in fragile.items():
            for q, g, nm, ulp in rows:
                lim = RES + ULPS * ulp
                print(f"    {str(W.cands[i]):12s} query {q:5d}: |own - {nm:10s}| = {g:.3e}   "
                      f"(bin {RES:g}, ulp {ulp:.1e}; {(g - RES) / ulp:+8.2f} ulp past the bin)"
                      + ("   <-- a last-bit difference can tie these" if g < lim else ""))
    all_fragile = bool(fragile) and all(any(g < RES + ULPS * u for _, g, _, u in rows)
                                         for rows in fragile.values())

    for nm, dd, gr in (("repr_select", d_sel, gr_sel), ("repr_gram", d_gram, gr_gram)):
        if dd > 0:
            w = np.argsort(-np.abs(saved - gr))[:6]
            print(f"\n  genes where {nm} differs from disk (worst 6):")
            for i in w:
                if saved[i] == gr[i]:
                    continue
                qs = np.where(own == i)[0]
                print(f"    gene {i:5d} {str(W.cands[i]):12s} nq={qs.size} "
                      f"saved={saved[i]:.4f} live={gr[i]:.4f} "
                      f"per-query saved?/live={rank_sel[qs]}/{rank_gram[qs]}")
    # The two live paths run in ONE process, so their agreement says nothing about
    # reproducibility ACROSS processes. An earlier version of this verdict missed that and
    # concluded "different code or inputs" for CCT3 -- measured wrong: an 8-thread rerun on
    # the same node gives the same live value, and no dependency changed after the artifact
    # was written. Distinguish the cases by the measurement above instead.
    small = d_sel <= 0.5 and diff_genes.size <= max(2, int(0.005 * K))
    print("\nverdict: "
          + ("the two code paths disagree; the stage above with max|d|>0 is the cause"
             if d_live > 0 else
             "everything agrees" if d_sel == 0 else
             f"tie-boundary noise: {diff_genes.size} gene(s), max|d|={d_sel:g}, and every one "
             "has a competitor inside the rounding resolution, so a last-bit float difference "
             "between two processes flips a tie. Not a code or input change; aggregates are "
             "unaffected. Reconcile per-gene ranks with a tie-boundary tolerance."
             if small and all_fragile else
             "the saved artifact differs from both live paths by MORE than tie-boundary noise "
             "(or on genes not at a rounding boundary): it was written by different code or "
             "inputs. Find the change before trusting either."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
