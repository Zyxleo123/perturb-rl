#!/usr/bin/env python
"""Case studies for the gap / improvement story: named genes, not method tables.

Two subcommands, so figures redraw without recomputation:

  compute  reads the RL cache, the saved chosen sets (artifacts/rl_acquire/{baselines,
           runs}) and, for leak-free 'measured' signatures, rebuilds each Replogle
           dataset through rl_headroom.build itself (same seed, checked against the
           cache). Writes artifacts/story/story_<dataset>.npz + story.json.
  draw     reads ONLY artifacts/story/ and writes fig18-fig21.

What each case is (dataset split 4 = GAPS part 4's split throughout):
  gap       one gene's held-out query against every candidate twice: measured
            signatures (half A of the cells; the query is half B) and GO-kNN
            predictions from each candidate's 10 closest GO neighbours (leave-one-out,
            also built from half A). The rank of the right answer under each.
  flow      every report gene's rank under a typical random 100-gene atlas (median
            over the same 100 draws as random_dist) and under REINFORCE's atlas
            (median over its 3 seeds). Reconciled against records.jsonl.
  anatomy   the most-rescued report genes: the 10 GO neighbours each atlas lends
            the kNN, their GO Jaccard, weight, and signature agreement with the gene.
  noise     greedy (every aux gene as reward, every candidate scored, seed 0), at
            every step: each available candidate's gain on the visible reward genes
            (what greedy maximises) and on the report genes (what it is judged on).
  coherence a diagnostic, not a figure: each candidate's `coherence` (rl_acquire's
            peek feature, PPO's strongest: do its GO neighbours respond like it?)
            against what is visible before measuring it. Measured 2026-11-18: R^2
            0.65 (K562) / 0.50 (RPE1), so coherence is NOT invisible to a blind
            policy and cannot by itself explain the blind regime's ceiling.

Usage:
  python -I story_cases.py compute --art-dir A --gears-dir G [--force]
  python -I story_cases.py draw    --art-dir A --fig-dir F
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own scripts/ only
import numpy as np  # noqa: E402

import rl_acquire as RA  # noqa: E402
import rl_headroom as RH  # noqa: E402

SPLIT = 4
DATASETS = ("replogle_k562_essential", "replogle_rpe1_essential")
SHORT = {"replogle_k562_essential": "K562", "replogle_rpe1_essential": "RPE1"}
RL_METHOD = "pg_tab"          # REINFORCE, per-gene logits: GAPS's headline peek method
N_ANATOMY = 3                 # most-rescued genes kept per dataset (draw uses the first)


def log(*a):
    RH.log(*a)


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


# ===================================================================== scoring
def tie_rank(S, own):
    """rl_headroom.score's tie-aware rank and reciprocal rank of each row's owner."""
    S = np.round(S, 6)
    t = S[np.arange(S.shape[0]), own]
    g = (S > t[:, None]).sum(1)
    e = (S == t[:, None]).sum(1) - 1
    H = RH.harmonic(S.shape[1])
    return g + 1 + e / 2.0, (H[g + e + 1] - H[g]) / (e + 1)


class RepScorer:
    """Reporter (rl_acquire) per gene instead of averaged: identical predictor and
    metric, so the mean of per_gene()'s rr is the run's reported MRR."""

    def __init__(self, W, rep):
        self.W, self.rep = W, np.asarray(rep)
        m = np.isin(W.owner, rep)
        remap = {int(o): i for i, o in enumerate(rep)}
        self.Qu = RH.unit(W.Q[m])
        self.o = np.array([remap[int(o)] for o in W.owner[m]])
        self.cnt = np.bincount(self.o, minlength=len(rep))

    def predict(self, chosen):
        c = np.asarray(chosen, np.int64)
        J = RH.jaccard(self.W.F[self.rep], self.W.F[c])
        return RH.go_knn_predict(J, self.W.Y[c], self.W.Y[c].mean(0))

    def per_gene(self, chosen):
        P = self.predict(chosen)
        rank, rr = tie_rank(self.Qu @ RH.unit(P).T, self.o)
        n = len(self.rep)
        return (np.bincount(self.o, weights=rank, minlength=n) / self.cnt,
                np.bincount(self.o, weights=rr, minlength=n) / self.cnt)


# ===================================================================== compute
def case_gap(W, YA):
    """Every gene's first held-out query vs all K measured (half A) signatures, and vs
    all K leave-one-out GO-kNN predictions built from half A. Half A never contains a
    query cell, so neither row is fitted to the query."""
    K = W.K
    qi = np.array([W.q_of[i][0] for i in range(K)])
    Qu = RH.unit(W.Q[qi])
    J = W.JJ.copy()
    np.fill_diagonal(J, 0.0)                           # a gene never predicts itself
    P = RH.go_knn_predict(J, YA, YA.mean(0))
    S_meas = Qu @ RH.unit(YA).T
    S_pred = Qu @ RH.unit(P).T
    own = np.arange(K)
    r_meas, _ = tie_rank(S_meas, own)
    r_pred, _ = tie_rank(S_pred, own)
    fid = np.einsum("ij,ij->i", RH.unit(P), RH.unit(YA))          # cos(prediction, measured)
    spread_pred = S_pred.std(1)
    spread_meas = S_meas.std(1)
    return {"gap_S_meas": S_meas.astype(np.float16), "gap_S_pred": S_pred.astype(np.float16),
            "gap_rank_meas": r_meas.astype(np.float32), "gap_rank_pred": r_pred.astype(np.float32),
            "gap_fidelity": fid.astype(np.float32),
            "gap_spread_meas": spread_meas.astype(np.float32),
            "gap_spread_pred": spread_pred.astype(np.float32)}


def case_flow(W, dep, rl_runs, recs):
    sc = RepScorer(W, dep["rep"])
    act = dep["act"]
    srng = np.random.default_rng(9100)                 # random_dist's own stream
    draws = [act[srng.choice(act.size, size=RA.B_DEPLOY, replace=False)] for _ in range(RA.N_RAND)]
    R_rand = np.zeros((len(draws), len(dep["rep"])), np.float32)
    RR_rand = np.zeros_like(R_rand)
    for k, ch in enumerate(draws):
        R_rand[k], RR_rand[k] = sc.per_gene(ch)
    rl_sets = [r["chosen"] for r in sorted(rl_runs, key=lambda r: r["seed"])]
    R_rl = np.zeros((len(rl_sets), len(dep["rep"])), np.float32)
    RR_rl = np.zeros_like(R_rl)
    for k, ch in enumerate(rl_sets):
        R_rl[k], RR_rl[k] = sc.per_gene(ch)
    # reconcile with what the report table says, or nothing here is the same measurement
    rand_mrr = RR_rand.mean(1)
    rec_rand = [r["ref_med"] for r in recs if r["method"] == RL_METHOD][0]
    rec_rl = {r["seed"]: r["mrr"] for r in recs if r["method"] == RL_METHOD}
    chk = {"random_median_here": float(np.median(rand_mrr)), "random_median_records": rec_rand,
           "rl_mrr_here": [float(x) for x in RR_rl.mean(1)],
           "rl_mrr_records": [rec_rl.get(r["seed"]) for r in sorted(rl_runs, key=lambda r: r["seed"])]}
    chk["max_abs_diff"] = float(max([abs(chk["random_median_here"] - rec_rand)] +
                                    [abs(a - b) for a, b in zip(chk["rl_mrr_here"], chk["rl_mrr_records"])
                                     if b is not None]))
    return sc, draws, rl_sets, {"flow_R_rand": R_rand, "flow_RR_rand": RR_rand,
                                "flow_R_rl": R_rl, "flow_RR_rl": RR_rl,
                                "flow_rep": dep["rep"], "flow_rl_sets": np.array(rl_sets)}, chk


def case_anatomy(W, dep, sc, draws, rl_sets, F, genes):
    """For the most-rescued report genes: the neighbourhood each atlas lends the kNN."""
    rep = dep["rep"]
    med_rand = np.median(F["flow_R_rand"], 0)
    med_rl = np.median(F["flow_R_rl"], 0)
    gain = np.log2(med_rand / med_rl)
    elig = np.flatnonzero(med_rl <= 1.5)               # RL atlas puts it (about) first
    order = elig[np.argsort(-gain[elig], kind="stable")][:N_ANATOMY]
    out = []
    for j in order:
        g = int(rep[j])
        atl = {}
        # the specific atlases whose rank for this gene is the median of their family
        kr = int(np.argmin(np.abs(F["flow_R_rand"][:, j] - med_rand[j])))
        kl = int(np.argmin(np.abs(F["flow_R_rl"][:, j] - med_rl[j])))
        for name, ch, k in (("random", draws[kr], kr), ("rl", rl_sets[kl], kl)):
            c = np.asarray(ch, np.int64)
            Jg = RH.jaccard(W.F[[g]], W.F[c])
            P = RH.go_knn_predict(Jg, W.Y[c], W.Y[c].mean(0))[0]
            kk = min(RA.KNN, c.size)
            top = np.argpartition(-Jg, kk - 1, axis=1)[0, :kk]
            jv = Jg[0, top]
            w = jv / jv.sum() if jv.sum() > 0 else np.full(kk, 1.0 / kk)
            o = np.argsort(-w, kind="stable")
            nb = c[top][o]
            # what beat it: the report gene whose prediction the query matched best
            Pall = sc.predict(c)
            qs = W.q_of[g]
            S = RH.unit(W.Q[qs]) @ RH.unit(Pall).T
            winners = [str(W.cands[rep[int(np.argmax(s))]]) for s in S]
            atl[name] = {"index": k, "rank": float(F["flow_R_rand" if name == "random" else "flow_R_rl"][k, j]),
                         "neighbours": [str(W.cands[x]) for x in nb],
                         "jaccard": [float(x) for x in jv[o]], "weight": [float(x) for x in w[o]],
                         "agree": [float(W.CC[g, x]) for x in nb],
                         "pred_cos": float(RH.unit(P[None])[0] @ RH.unit(W.Y[[g]])[0]),
                         "top_match": winners, "pred": P}
        top_genes = np.argsort(-np.abs(W.Y[g]), kind="stable")[:40]
        out.append({"gene": str(W.cands[g]), "rep_pos": int(j), "rank_random_median": float(med_rand[j]),
                    "rank_rl_median": float(med_rl[j]), "log2_gain": float(gain[j]),
                    "strip_genes": [str(genes[x]) for x in top_genes],
                    "strip_truth": [float(W.Y[g, x]) for x in top_genes],
                    "strip_random": [float(atl["random"]["pred"][x]) for x in top_genes],
                    "strip_rl": [float(atl["rl"]["pred"][x]) for x in top_genes],
                    "atlases": {k: {kk: v for kk, v in a.items() if kk != "pred"} for k, a in atl.items()}})
    return out


def case_noise(W, dep, chosen):
    """Greedy's actual path; at each step, every available candidate's gain on the
    visible reward genes (aux, greedy's own objective) and on the report genes."""
    rs_vis, rs_hid = RA.RewardSet(W, dep["aux"]), RA.RewardSet(W, dep["rep"])
    act = dep["act"]
    steps, cand, dv, dh, picked = [], [], [], [], []
    rs = []
    for t in range(1, len(chosen)):
        base = list(chosen[:t])
        av = np.setdiff1d(act, base)
        v0, h0 = rs_vis.mrr(base), rs_hid.mrr(base)
        v = np.array([rs_vis.mrr(base + [int(c)]) for c in av]) - v0
        h = np.array([rs_hid.mrr(base + [int(c)]) for c in av]) - h0
        steps.append(np.full(av.size, t, np.int16))
        cand.append(av.astype(np.int32))
        dv.append(v.astype(np.float32))
        dh.append(h.astype(np.float32))
        picked.append(int(chosen[t]))
        rs.append(float(np.corrcoef(v, h)[0, 1]) if v.std() > 0 and h.std() > 0 else np.nan)
        if t % 10 == 0:
            log(f"    noise step {t}: r(vis, hid) = {rs[-1]:.3f}")
    return {"noise_step": np.concatenate(steps), "noise_cand": np.concatenate(cand),
            "noise_dvis": np.concatenate(dv), "noise_dhid": np.concatenate(dh),
            "noise_picked": np.array(picked, np.int32), "noise_r": np.array(rs, np.float32),
            "noise_chosen": np.array(chosen, np.int32)}


def case_coherence(W, dep, base_rows, rl_rows):
    A, aux = dep["act"], dep["aux"]
    # peek feature (needs the candidate's own signature)
    fs = RA.FeatureState(W, A, aux, "peek", RA.B_DEPLOY)
    coh = fs.static["coherence"]
    # what is visible before measuring it: blind FeatureState's features at the start
    fb = RA.FeatureState(W, A, aux, "blind", RA.B_DEPLOY)
    fb.begin_round(0)
    g, av = fb.raw_gain()
    vis = {"go_degree": W.deg[A], "go_centrality": fb.static["centrality"],
           "facility_gain": g / max(1, fb.T.size),
           "neighbour_agreement": fb.nb["nb_agree"], "neighbour_signal": fb.nb["nb_lognorm"],
           "neighbour_go_mass": fb.nb["nb_mass"]}

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    rho = {k: spearman(coh, v) for k, v in vis.items()}
    # how well ALL the visible features together predict coherence (5-fold ridge, R^2)
    X = np.stack([vis[k] for k in vis], 1).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = coh.astype(np.float64)
    pred = np.zeros_like(y)
    perm = np.random.default_rng(0).permutation(y.size)
    for f in range(5):
        te = perm[f::5]
        tr = np.setdiff1d(perm, te)
        Xt = np.c_[np.ones(tr.size), X[tr]]
        beta = np.linalg.solve(Xt.T @ Xt + 1.0 * np.eye(Xt.shape[1]), Xt.T @ y[tr])
        pred[te] = np.c_[np.ones(te.size), X[te]] @ beta
    r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    picks = {}
    for m in ("pg_tab", "ppo_peek", "critic_shrink"):
        sets = [set(r["chosen"]) for r in rl_rows if r["method"] == m]
        picks[m] = np.array([sum(int(a) in s for s in sets) for a in A], np.int8)
    fac = set(next(r for r in base_rows if r["strategy"] == "go_facility")["chosen"])
    picks["go_facility"] = np.array([int(int(a) in fac) for a in A], np.int8)
    out = {"coh_act": A, "coh_value": coh, "coh_pred_visible": pred.astype(np.float32)}
    for k, v in vis.items():
        out[f"coh_vis_{k}"] = np.asarray(v, np.float32)
    for k, v in picks.items():
        out[f"coh_pick_{k}"] = v
    return out, {"spearman_coherence_vs_visible": rho, "r2_visible_to_coherence": r2,
                 "n_methods_seeds": {m: int(sum(1 for r in rl_rows if r["method"] == m))
                                                 for m in ("pg_tab", "ppo_peek", "critic_shrink")}}


def compute(a):
    od = os.path.join(a.art_dir, "story")
    os.makedirs(od, exist_ok=True)
    jpath = os.path.join(od, "story.json")
    summary = json.load(open(jpath)) if os.path.exists(jpath) else {}
    summary.setdefault("errors", [])
    root = os.path.join(a.art_dir, "rl_acquire")
    check = json.load(open(os.path.join(root, "cache_check.json")))
    recs_all = load_jsonl(os.path.join(root, "records.jsonl"))
    g2g = None
    for ds in a.datasets:
        out_npz = os.path.join(od, f"story_{ds}.npz")
        if os.path.exists(out_npz) and ds in summary and not a.force:
            log(f"{ds}: {out_npz} exists, skipping (pass --force to redo)")
            continue
        try:
            t0 = time.time()
            W = RA.World(check["datasets"][ds]["path"])
            dep = RA.split_of(W.K, SPLIT)
            log(f"{ds}: K={W.K} rep={dep['rep'].size} act={dep['act'].size} aux={dep['aux'].size}")
            # leak-free measured signatures + readout gene names, via the same build
            if g2g is None:
                g2g = RH.load_gene2go(os.path.join(a.gears_dir, "gene2go_all.pkl"))
            D = RH.build(ds, os.path.join(a.gears_dir, ds, "perturb_processed.h5ad"),
                         np.random.default_rng(RH.SEED))
            mk = D["mask"]
            same = (list(D["cands"]) == [str(x) for x in W.cands]
                    and np.allclose(D["dF"][:, mk], W.Y, atol=1e-6))
            log(f"  rebuild matches cache: {same}")
            if not same:
                raise RuntimeError("rl_headroom.build no longer reproduces the cache; refusing to mix them")
            YA = D["dA"][:, mk].astype(np.float32)
            genes = np.asarray(D["genes"])[mk]
            del D
            res = {"cands": np.array([str(x) for x in W.cands]), "readout_genes": genes.astype(str)}
            res.update(case_gap(W, YA))
            log(f"  gap: median rank measured {np.median(res['gap_rank_meas']):.1f}, "
                f"predicted {np.median(res['gap_rank_pred']):.1f}")
            brows = load_jsonl(os.path.join(root, "baselines", f"{ds}__s{SPLIT}.jsonl"))
            runs = []
            for p in sorted(glob.glob(os.path.join(root, "runs", f"*__{ds}__s{SPLIT}__seed*.json"))):
                if not p.endswith(".error.json"):
                    runs += json.load(open(p))
            recs = [r for r in recs_all if r["dataset"] == ds and int(r["split"]) == SPLIT]
            rl_runs = [r for r in runs if r["method"] == RL_METHOD]
            sc, draws, rl_sets, F, chk = case_flow(W, dep, rl_runs, recs)
            res.update(F)
            log(f"  flow reconciliation: {chk}")
            anat = case_anatomy(W, dep, sc, draws, rl_sets, F, genes)
            log(f"  anatomy: {[(x['gene'], x['rank_random_median'], x['rank_rl_median']) for x in anat]}")
            greedy0 = next(r for r in brows if r["strategy"] == "greedy_aux_all" and int(r["seed"]) == 0)
            res.update(case_noise(W, dep, greedy0["chosen"]))
            cres, cinfo = case_coherence(W, dep, brows, runs)
            res.update(cres)
            log(f"  coherence: {cinfo}")
            np.savez_compressed(out_npz, **res)
            summary[ds] = {"split": SPLIT, "rl_method": RL_METHOD, "flow_check": chk, "anatomy": anat,
                           "coherence": cinfo,
                           "gap": {"median_rank_measured": float(np.median(res["gap_rank_meas"])),
                                   "median_rank_predicted": float(np.median(res["gap_rank_pred"])),
                                   "top1_measured": float((res["gap_rank_meas"] <= 1).mean()),
                                   "top1_predicted": float((res["gap_rank_pred"] <= 1).mean()),
                                   "median_fidelity": float(np.median(res["gap_fidelity"]))},
                           "noise": {"median_r": float(np.nanmedian(res["noise_r"])),
                                     "greedy_seed": 0, "strategy": "greedy_aux_all"},
                           "secs": round(time.time() - t0, 1)}
            log(f"  {ds} done in {time.time() - t0:.0f}s")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            summary["errors"].append(f"{ds}: {type(e).__name__}: {e}")
        with open(jpath, "w") as f:
            json.dump(summary, f, indent=1)
    log("errors:", summary["errors"] or "none")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--art-dir", required=True)
    c.add_argument("--gears-dir", required=True)
    c.add_argument("--datasets", nargs="*", default=list(DATASETS))
    c.add_argument("--force", action="store_true")
    d = sub.add_parser("draw")
    d.add_argument("--art-dir", required=True)
    d.add_argument("--fig-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "compute":
        return compute(a)
    import story_draw
    return story_draw.draw(a)


if __name__ == "__main__":
    sys.exit(main())
