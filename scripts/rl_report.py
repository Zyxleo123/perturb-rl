#!/usr/bin/env python
"""Aggregate the RL-acquisition runs into a table, a summary and two figures.

Reads only saved artifacts (<art>/rl_acquire/{baselines,runs}/), so it redraws in
seconds without recomputing anything:
  baselines/<dataset>__s<split>.jsonl          one row per baseline strategy
  runs/<method>__<dataset>__s<split>__seed<k>.json  final rows of one RL run
  runs/*.curve.jsonl                           PPO learning curves
Every RL number is expressed against ITS OWN split's random-x100 distribution and
cheating bound, because a split changes both.

Writes <art>/rl_acquire/{summary.json,report.md} and
<fig>/fig15_rl_acquisition.png, <fig>/fig16_rl_curves.png.
Usage: python -I rl_report.py --art-dir D --fig-dir D [--smoke]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own scripts/ only
import numpy as np  # noqa: E402

import rl_acquire as RA  # noqa: E402  (numpy + rl_headroom only; torch is imported lazily)

BASE_ORDER = ["go_coverage", "go_facility", "go_facility_idf", "go_facility_emb",
              "greedy_rew400_c120", "greedy_aux_all"]
BLIND_RL = ["ppo_blind", "ppo_blind_match", "ppo_blind_rich", "ppo_blind_xline",
            "ppo_blind_xline_rich"]
RL_ORDER = ["pg_tab", "pg_tab_boot", "critic_only", "critic_shrink", "ppo_peek"] + BLIND_RL
CHEATS = ["cheat_rep_c120", "cheat_rep_all"]
LABEL = {
    "go_coverage": "GO coverage (static, blind)",
    "go_facility": "GO facility location (static, blind)",
    "go_facility_idf": "facility location, IDF-weighted GO (static, blind)",
    "go_facility_emb": "facility location, GO embedding (static, blind)",
    "greedy_rew400_c120": "greedy, GAPS protocol (peek)",
    "greedy_aux_all": "greedy, all aux genes (peek)",
    "pg_tab": "REINFORCE, per-gene logits",
    "pg_tab_boot": "REINFORCE, resampled reward",
    "critic_only": "learned critic, greedy",
    "critic_shrink": "critic-shrunk greedy",
    "ppo_peek": "PPO feature policy (peek)",
    "ppo_blind": "PPO blind (as before: shrunk sub-problems, GO only)",
    "ppo_blind_match": "PPO blind, size-matched sub-problems",
    "ppo_blind_rich": "PPO blind, richer gene priors",
    "ppo_blind_xline": "PPO blind, trained on the other cell line",
    "ppo_blind_xline_rich": "PPO blind, other cell line + richer priors",
    "cheat_rep_c120": "cheat, GAPS protocol",
    "cheat_rep_all": "cheat, all candidates",
}
# published acquisition methods on the identical task (scripts/related_work.py)
REL_BLIND = ["iterpert", "iterpert_go", "iterpert_cold", "kc_go", "kc_iterpert_priors", "fl_iterpert_priors",
             "al_coreset", "al_bald", "al_batchbald",
             "al_badge", "al_lcmd", "al_acsfw", "al_kmeans", "al_typiclust"]
REL_PEEK = ["dv_shapley", "dv_banzhaf", "dv_datamodels", "rep_facility_sig", "rep_coreset_sig"]
RELATED = REL_BLIND + REL_PEEK
LABEL.update({
    "iterpert": "IterPert (its 8 released priors)",
    "iterpert_go": "IterPert, GO prior only",
    "iterpert_cold": "IterPert, no random first batch",
    "fl_iterpert_priors": "facility location on IterPert's priors (static)",
    "kc_iterpert_priors": "IterPert's k-centre on its priors (static)",
    "kc_go": "IterPert's k-centre on GO (static)",
    "al_coreset": "Core-Set",
    "al_bald": "BALD",
    "al_batchbald": "BatchBALD",
    "al_badge": "BADGE",
    "al_lcmd": "LCMD",
    "al_acsfw": "ACS-FW",
    "al_kmeans": "k-means sampling (GeneDisco)",
    "al_typiclust": "TypiClust",
    "dv_shapley": "TMC Data Shapley",
    "dv_banzhaf": "Data Banzhaf (MSR)",
    "dv_datamodels": "datamodels (l1 surrogate)",
    "rep_facility_sig": "exemplar facility location, signatures",
    "rep_coreset_sig": "Core-Set on true signatures",
})
BLIND = {"go_coverage", "go_facility", "go_facility_idf", "go_facility_emb", *BLIND_RL, *REL_BLIND}
# fig16 line styles, so the five blind variants are told apart
CURVE_STYLE = {"ppo_blind": "-", "ppo_blind_match": (0, (4, 2)), "ppo_blind_rich": (0, (1, 1.5)),
               "ppo_blind_xline": (0, (6, 2, 1, 2)), "ppo_blind_xline_rich": (0, (3, 1, 1, 1, 1, 1))}
SHORT = {"replogle_k562_essential": "Replogle K562", "replogle_rpe1_essential": "Replogle RPE1"}

# reference palette (dataviz skill, references/palette.md), light mode
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
C_PEEK, C_BLIND = "#2a78d6", "#eb6834"


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root = os.path.join(a.art_dir, "rl_acquire")
    runs_dir = os.path.join(a.art_dir, "rl_acquire_smoke" if a.smoke else "rl_acquire", "runs")
    os.makedirs(a.fig_dir, exist_ok=True)

    # ------------------------------------------------------------ inputs
    brows = []
    for p in sorted(glob.glob(os.path.join(root, "baselines", "*.jsonl"))):
        brows += load_jsonl(p)
    runs, errors = [], []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        if not p.endswith(".error.json"):
            runs += json.load(open(p))
    for p in sorted(glob.glob(os.path.join(runs_dir, "*.error.json"))):
        errors += [f"{os.path.basename(p)}: {r.get('error')}" for r in json.load(open(p))]
    # related work (scripts/related_work.py): same row format, re-scored below like every run
    rel_dir = os.path.join(a.art_dir, "rl_acquire", "related_smoke" if a.smoke else "related")
    for p in sorted(glob.glob(os.path.join(rel_dir, "runs", "*.json"))):
        if not p.endswith(".error.json"):
            runs += json.load(open(p))
    for p in sorted(glob.glob(os.path.join(rel_dir, "runs", "*.error.json"))):
        errors += [f"related/{os.path.basename(p)}: {r.get('error')}" for r in json.load(open(p))]

    # ------------------------------------------------------------ re-score in GAPS's env
    # The kNN picks neighbours with argpartition, and numpy 2.x (this env, where GAPS
    # ran) breaks ties among equal GO Jaccards differently from numpy 1.26 (the torch
    # env the runs trained in). So every chosen set, and the random x100, is scored
    # HERE, with rl_headroom's own functions: one measurement, the same as GAPS's.
    check = json.load(open(os.path.join(root, "cache_check.json")))
    worlds, reps, rand = {}, {}, {}

    class _W:
        pass

    def reporter(ds, sp):
        if (ds, sp) not in reps:
            if ds not in worlds:
                z = np.load(check["datasets"][ds]["path"], allow_pickle=False)
                w = _W()
                w.Y, w.Q, w.owner = z["Y"], z["Q"], z["owner"].astype(np.int64)
                w.F, w.K = z["F"].astype(np.float32), z["Y"].shape[0]
                worlds[ds] = w
            dep = RA.split_of(worlds[ds].K, sp)
            reps[(ds, sp)] = (RA.Reporter(worlds[ds], dep["rep"], dep["act"]), dep,
                              RA.Reporter(worlds[ds], dep["aux"], dep["act"]))
        return reps[(ds, sp)][:2]

    coh_cache = {}

    def coherence(ds, sp):
        """rl_acquire's `coherence` feature (PPO peek's strongest, +2.7) for every
        candidate: GO-Jaccard-weighted mean signature cosine to its 10 GO neighbours among
        act + aux. Do the CHOSEN reference genes have signatures their GO neighbourhood
        agrees with? A diagnostic only: it reads candidate signatures."""
        if (ds, sp) not in coh_cache:
            w = worlds[ds]
            dep = RA.split_of(w.K, sp)
            A, V = dep["act"], np.concatenate([dep["act"], dep["aux"]])
            J = RA.RH.jaccard(w.F[A], w.F[V])
            J[A[:, None] == V[None, :]] = 0.0
            top = np.argpartition(-J, RA.KNN - 1, axis=1)[:, :RA.KNN]
            wg = np.take_along_axis(J, top, 1).astype(np.float64)
            U = RA.RH.unit(np.asarray(w.Y, np.float64))
            cs = np.einsum("ij,ikj->ik", U[A], U[V[top]])
            coh = (wg * cs).sum(1) / np.maximum(wg.sum(1), 1e-9)
            coh_cache[(ds, sp)] = {int(g): float(c) for g, c in zip(A, coh)}
        return coh_cache[(ds, sp)]

    def rescore(row):
        fn, _ = reporter(row["dataset"], row["split"])
        r = fn(row["chosen"])
        row["mrr_train_env"], row["top1_train_env"] = row["rep_mrr"], row["rep_top1"]
        row["rep_mrr"], row["rep_top1"] = r["mrr"], r["top1"]
        # the same set scored on the aux genes: what a peek method's own reward sees.
        # High own + low rep = fitting the reward's noise; the gap is the overfit.
        row["aux_mrr"] = reps[(row["dataset"], row["split"])][2](row["chosen"])["mrr"]
        # What the GO-kNN predictor has to work with on the report genes: the summed GO
        # Jaccard of each report gene's 10 nearest CHOSEN genes, and the share of report
        # genes with no GO overlap to any chosen gene (they fall back to the mean). A
        # diagnostic of the coverage mechanism only -- it reads report-gene GO, never
        # their signatures, and no method sees it.
        c = np.array([fn.pos[int(g)] for g in row["chosen"]])
        J = fn.J[:, c]
        row["go_mass"] = float(np.sort(J, 1)[:, -RA.KNN:].sum(1).mean())
        row["go_orphan"] = float((J.max(1) <= 0).mean())
        ch = coherence(row["dataset"], row["split"])
        row["ref_coh"] = float(np.mean([ch[int(g)] for g in row["chosen"]]))

    keys = sorted({(r["dataset"], r["split"]) for r in brows if r["strategy"] == "random"})
    for ds, sp in keys:
        fn, dep = reporter(ds, sp)
        rand[(ds, sp)] = RA.random_dist(fn, dep["act"], RA.B_DEPLOY)
    for r in brows + runs:
        if r.get("chosen") and "rep_mrr" in r:
            rescore(r)
    env_diff = [abs(r["rep_mrr"] - r["mrr_train_env"]) for r in brows + runs if "mrr_train_env" in r]
    print(f"re-scored {len(env_diff)} chosen sets in the GAPS env; |MRR change| "
          f"mean {np.mean(env_diff) if env_diff else float('nan'):.5f} max {max(env_diff) if env_diff else float('nan'):.5f}")

    base = {}            # (ds, split) -> {strategy: [rows]}
    for r in brows:
        base.setdefault((r["dataset"], r["split"]), {}).setdefault(r["strategy"], []).append(r)

    def ref(ds, sp):
        b, rnd = base.get((ds, sp), {}), rand.get((ds, sp))
        if rnd is None:
            return None
        mean = lambda s: float(np.mean([x["rep_mrr"] for x in b[s]])) if s in b else np.nan  # noqa: E731
        cheats = [x["rep_mrr"] for s in CHEATS for x in b.get(s, [])]
        # the cheat bound is "a subset this good exists": the best cheat run found
        # the floor for a policy that gets the richer views as features is the BEST
        # static use of any of those views, not the Jaccard one alone
        facs = [mean(s) for s in ("go_facility", "go_facility_idf", "go_facility_emb") if s in b]
        return {"med": rnd["rep_mrr"], "p95": rnd["rep_mrr_p95"], "p05": rnd["rep_mrr_p05"],
                "max": rnd["rep_mrr_max"], "cheat": max(cheats) if cheats else np.nan,
                "greedy": mean("greedy_aux_all"), "facility": mean("go_facility"),
                "facility_best": max(facs) if facs else np.nan}

    # one record per (method, dataset, split, seed), everything relative to its split
    recs = []
    for (ds, sp), b in base.items():
        R = ref(ds, sp)
        if R is None:
            continue
        for s in BASE_ORDER + CHEATS:
            for x in b.get(s, []):
                recs.append({"method": s, "dataset": ds, "split": sp, "seed": x.get("seed"),
                             "mrr": x["rep_mrr"], "top1": x["rep_top1"], "aux": x.get("aux_mrr", np.nan),
                             "go_mass": x.get("go_mass", np.nan), "go_orphan": x.get("go_orphan", np.nan),
                             "ref_coh": x.get("ref_coh", np.nan),
                             **{f"ref_{k}": v for k, v in R.items()}})
    for r in runs:
        if "rep_mrr" not in r:
            continue
        R = ref(r["dataset"], r["split"])
        if R is None:
            continue
        recs.append({"method": r["method"], "dataset": r["dataset"], "split": r["split"], "seed": r["seed"],
                     "mrr": r["rep_mrr"], "top1": r["rep_top1"], "aux": r.get("aux_mrr", np.nan),
                     "go_mass": r.get("go_mass", np.nan), "go_orphan": r.get("go_orphan", np.nan),
                             "ref_coh": r.get("ref_coh", np.nan),
                     "diag": r.get("diag"), "train_scale": r.get("train_scale"),
                     "best_val": r.get("best_val"), "val_ref": r.get("val_ref"),
                     "best_iter": r.get("best_iter"), **{f"ref_{k}": v for k, v in R.items()}})
    for x in recs:
        x["ratio"] = x["mrr"] / x["ref_med"]
        x["above_p95"] = bool(x["mrr"] > x["ref_p95"])
        x["above_max"] = bool(x["mrr"] > x["ref_max"])
        x["cheat_frac"] = (x["mrr"] - x["ref_med"]) / (x["ref_cheat"] - x["ref_med"])
        x["vs_greedy"] = x["mrr"] - x["ref_greedy"]
        x["vs_facility"] = x["mrr"] - x["ref_facility"]
        x["vs_facility_best"] = x["mrr"] - x["ref_facility_best"]

    datasets = [d for d in SHORT if any(x["dataset"] == d for x in recs)]
    methods = [m for m in BASE_ORDER + RL_ORDER + RELATED + CHEATS if any(x["method"] == m for x in recs)]
    table = []
    for ds in datasets:
        for m in methods:
            xs = [x for x in recs if x["dataset"] == ds and x["method"] == m]
            if not xs:
                continue
            f = lambda k: np.array([x[k] for x in xs], float)  # noqa: E731
            sc = next((x["train_scale"] for x in xs if x.get("train_scale")), None)
            # how far above facility location the policy got on its OWN validation
            # sub-problems, i.e. at training scale. The blind question is whether that
            # advantage survives the move to deployment scale, so both are reported.
            vof = [x["best_val"] / x["val_ref"]["facility"] for x in xs
                   if x.get("best_val") and (x.get("val_ref") or {}).get("facility")]
            # paired comparison against greedy and facility on the SAME split
            table.append({
                "dataset": ds, "method": m, "n": len(xs), "splits": sorted({x["split"] for x in xs}),
                "train_scale": sc, "val_over_facility": float(np.mean(vof)) if vof else None,
                "mrr_mean": float(f("mrr").mean()), "mrr_sd": float(f("mrr").std()),
                "top1_mean": float(f("top1").mean()), "aux_mean": float(np.nanmean(f("aux"))),
                "ratio_mean": float(f("ratio").mean()), "ratio_min": float(f("ratio").min()),
                "ratio_max": float(f("ratio").max()),
                "frac_above_p95": float(f("above_p95").mean()), "frac_above_max": float(f("above_max").mean()),
                "cheat_frac_mean": float(f("cheat_frac").mean()),
                "vs_greedy_mean": float(f("vs_greedy").mean()),
                "frac_beats_greedy": float((f("vs_greedy") > 0).mean()),
                "vs_facility_mean": float(f("vs_facility").mean()),
                "frac_beats_facility": float((f("vs_facility") > 0).mean()),
                "vs_facility_best_mean": float(f("vs_facility_best").mean()),
                "frac_beats_facility_best": float((f("vs_facility_best") > 0).mean())})
    crit = [x["diag"] for x in recs if x.get("diag") and x["method"] == "critic_only"]
    diag = {}
    if crit:
        for k in ("corr_emp_vs_heldout_gain", "corr_critic_vs_heldout_gain", "corr_shrink_vs_heldout_gain",
                  "shrink_coef_critic", "shrink_coef_emp"):
            v = np.array([d[k] for d in crit], float)
            diag[k] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}
    # What did the PPO policies learn? Mean input-gradient of the logit per feature, over
    # inputs drawn like the real ones (per-candidate features are z-scored per decision;
    # context features uniform in [0, 1]). Signed: + means "prefer genes high on this".
    attrib = {}
    for pth in sorted(glob.glob(os.path.join(runs_dir, "ppo_*.policy.npz"))):
        z = np.load(pth)
        names = [str(x) for x in z["features"]]
        W1, b1, W2, b2, W3 = z["0.weight"], z["0.bias"], z["2.weight"], z["2.bias"], z["4.weight"]
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4000, len(names))).astype(np.float32)
        nctx = len(RA.CTX_FEATS)
        X[:, -nctx:] = rng.uniform(0, 1, (4000, nctx))
        h1 = np.tanh(X @ W1.T + b1)
        h2 = np.tanh(h1 @ W2.T + b2)
        g = ((W3 * (1 - h2 ** 2)) @ W2 * (1 - h1 ** 2)) @ W1        # 4000 x d
        me = os.path.basename(pth).split("__")[0]
        ds = os.path.basename(pth).split("__")[1]
        attrib.setdefault(f"{me}__{ds}", []).append(dict(zip(names, g.mean(0).tolist())))
    attrib_mean = {k: {f: float(np.mean([r[f] for r in v])) for f in v[0]} for k, v in attrib.items()}
    attrib_sd = {k: {f: float(np.std([r[f] for r in v])) for f in v[0]} for k, v in attrib.items()}
    # ------------------------------------------------------------ related work vs ours
    # Every published method against the references a reader needs, PAIRED BY SPLIT: a
    # run minus that reference's mean on the same split, since a split moves everything.
    # blind methods face the best static floor and PPO blind; peek methods face greedy,
    # REINFORCE and critic-shrunk greedy, the methods with the same information.
    smean = {}
    for x in recs:
        smean.setdefault((x["dataset"], x["split"], x["method"]), []).append(x["mrr"])
    smean = {k: float(np.mean(v)) for k, v in smean.items()}
    REFS = {"blind": [("facility_best", "best static floor"), ("ppo_blind", "PPO blind")],
            "peek": [("greedy_aux_all", "greedy, all aux"), ("pg_tab", "REINFORCE"),
                     ("critic_shrink", "critic-shrunk greedy")]}
    OWN = {"blind": ["go_facility", "go_facility_idf", "ppo_blind", "ppo_blind_xline_rich"],
           "peek": ["greedy_aux_all", "pg_tab", "critic_shrink", "ppo_peek"]}
    CITE = {r["method"]: r.get("cite", "") for r in runs if r.get("method") in RELATED}

    def versus(xs, ref):
        d = []
        for x in xs:
            r = x["ref_facility_best"] if ref == "facility_best" else smean.get((x["dataset"], x["split"], ref))
            if r is not None and np.isfinite(r):
                d.append(x["mrr"] - r)
        d = np.array(d)
        if d.size == 0:
            return None
        sd = d.std(ddof=1) if d.size > 1 else 0.0
        return {"diff": float(d.mean()), "wins": float((d > 0).mean()), "n": int(d.size),
                "t": float(d.mean() / (sd / np.sqrt(d.size))) if sd > 0 else float("nan")}

    related_cmp = []
    if any(x["method"] in RELATED for x in recs):
        for ds in datasets:
            for regime in ("blind", "peek"):
                rel = REL_BLIND if regime == "blind" else REL_PEEK
                for m in OWN[regime] + rel:
                    xs = [x for x in recs if x["dataset"] == ds and x["method"] == m]
                    t = next((t for t in table if t["dataset"] == ds and t["method"] == m), None)
                    if not xs or t is None:
                        continue
                    related_cmp.append({
                        "dataset": ds, "regime": regime, "method": m, "ours": m not in rel, "n": len(xs),
                        "mrr_mean": t["mrr_mean"], "mrr_sd": t["mrr_sd"], "aux_mean": t["aux_mean"],
                        "ratio_mean": t["ratio_mean"], "cheat_frac_mean": t["cheat_frac_mean"],
                        "vs": {k: versus(xs, k) for k, _ in REFS[regime] if k != m}})
    rel_diag = {}
    for m in RELATED:
        rr = [r for r in runs if r.get("method") == m and isinstance(r.get("diag"), dict)]
        if not rr:
            continue
        rounds = [q for r in rr for q in r["diag"].get("rounds", [])]
        ov = [q["replica_overlap"] for q in rounds if "replica_overlap" in q]
        st = [q["selection_status"] for q in rounds if q.get("selection_status")]
        dd = {"n_runs": len(rr), "n_rounds_with_status": len(st), "statuses": sorted(set(st))}
        if ov:
            dd.update(replica_overlap_mean=float(np.mean(ov)), replica_overlap_min=float(np.min(ov)),
                      n_rounds_checked=len(ov))
        for k in ("cv_r2", "frac_truncated", "n_perm", "min_samples_per_gene", "own_reward", "n_nonzero",
                  "valuation_secs"):
            v = [r["diag"][k] for r in rr if k in r["diag"]]
            if v:
                dd[k] = float(np.mean(v))
        rel_diag[m] = dd
    # The coverage mechanism, tested on every BLIND record (ours, published, floors):
    # does the report genes' GO-neighbour mass among the chosen genes track the score?
    coverage = {}
    for ds in datasets:
        bl = [x for x in recs if x["dataset"] == ds and x["method"] in BLIND and np.isfinite(x["go_mass"])]
        if len(bl) < 5:
            continue
        gm, ra = np.array([x["go_mass"] for x in bl]), np.array([x["ratio"] for x in bl])
        co = np.array([x["ref_coh"] for x in bl])
        rk = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
        per = {}
        for m in sorted({x["method"] for x in bl}):
            xs = [x for x in bl if x["method"] == m]
            per[m] = {"go_mass": float(np.mean([x["go_mass"] for x in xs])),
                      "go_orphan": float(np.mean([x["go_orphan"] for x in xs])),
                      "ref_coh": float(np.mean([x["ref_coh"] for x in xs])),
                      "ratio": float(np.mean([x["ratio"] for x in xs])), "n": len(xs)}
        # across METHOD means too: within-method seed noise does not inflate it
        mm = np.array([[v["go_mass"], v["ratio"], v["ref_coh"]] for v in per.values()])
        coverage[ds] = {"n_records": len(bl), "pearson_records": float(np.corrcoef(gm, ra)[0, 1]),
                        "spearman_records": float(np.corrcoef(rk(gm), rk(ra))[0, 1]),
                        "pearson_method_means": float(np.corrcoef(mm[:, 0], mm[:, 1])[0, 1]),
                        "coh_pearson_records": float(np.corrcoef(co, ra)[0, 1]),
                        "coh_spearman_records": float(np.corrcoef(rk(co), rk(ra))[0, 1]),
                        "coh_pearson_method_means": float(np.corrcoef(mm[:, 2], mm[:, 1])[0, 1]),
                        "n_methods": len(per), "per_method": per}
    pcp = os.path.join(a.art_dir, "rl_acquire", "related", "priors_check.json")
    priors_check = json.load(open(pcp)) if os.path.exists(pcp) else None

    chk = {f"{ds}__s{sp}": b["scorer_check"][0] for (ds, sp), b in base.items() if "scorer_check" in b}
    # the generalised facility-location path must reproduce go_facility's own result
    # under the Jaccard view, or the two richer views are not measuring the same thing
    fpc = {f"{ds}__s{sp}": bool(b["facility_path_check"][0].get("jaccard_view_matches_go_facility"))
           for (ds, sp), b in base.items() if "facility_path_check" in b}
    summary = {"table": table, "critic_diag": diag, "scorer_check": chk,
               "facility_path_check": fpc, "errors": errors,
               "ppo_feature_gradient_mean": attrib_mean, "ppo_feature_gradient_sd": attrib_sd,
               "random_gaps_env": {f"{d}__s{s}": v for (d, s), v in rand.items()},
               "train_env_vs_gaps_env_abs_mrr_diff": {"mean": float(np.mean(env_diff)) if env_diff else None,
                                                      "max": float(max(env_diff)) if env_diff else None,
                                                      "n": len(env_diff)},
               "related_work": related_cmp, "related_diag": rel_diag, "blind_go_coverage": coverage,
               "related_priors": (priors_check or {}).get("datasets"),
               "n_runs": len(runs), "baseline_splits": sorted(f"{d}__s{s}" for d, s in base),
               "smoke": a.smoke}
    out_dir = os.path.dirname(runs_dir)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # every record, already re-scored in GAPS's env, so any number in the write-up can
    # be re-read without recomputing
    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for x in recs:
            f.write(json.dumps({k: v for k, v in x.items() if k != "diag"}, default=float) + "\n")

    # ------------------------------------------------------------ report.md
    L = ["# RL on the acquisition gap: results", "",
         "Report-set MRR, budget 100 of 437, pooled over splits x seeds. `x rand` is the multiple "
         "of that split's random-x100 median; `>p95` the share of runs beating that split's random "
         "p95; `cheat %` the share of (cheat - random median) captured, cheat = greedy on the report "
         "genes with every candidate scored; `vs greedy` the paired difference from greedy on all "
         "aux genes on the same split. `own (aux) MRR` scores the same chosen set on the aux genes, "
         "the reward a peek method optimises: high own with low rep is fitting that reward's noise.", ""]
    for ds in datasets:
        L += [f"## {SHORT[ds]}", "",
              "| method | n | rep MRR | top-1 | own (aux) MRR | x rand | >p95 | cheat % | vs greedy (wins) | vs facility (wins) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for t in [t for t in table if t["dataset"] == ds]:
            L.append(f"| {LABEL.get(t['method'], t['method'])} | {t['n']} | {t['mrr_mean']:.4f} ± {t['mrr_sd']:.4f} | "
                     f"{t['top1_mean']:.4f} | {t['aux_mean']:.4f} | {t['ratio_mean']:.2f}× | {t['frac_above_p95']:.0%} | "
                     f"{t['cheat_frac_mean']:.0%} | {t['vs_greedy_mean']:+.4f} ({t['frac_beats_greedy']:.0%}) | "
                     f"{t['vs_facility_mean']:+.4f} ({t['frac_beats_facility']:.0%}) |")
        L.append("")
    BLIND_FLOORS = ["go_facility", "go_facility_idf", "go_facility_emb"]
    blind_rows = [t for t in table if t["method"] in BLIND_RL or t["method"] in BLIND_FLOORS]
    if any(t["method"] in BLIND_RL[1:] for t in blind_rows):
        L += ["## The blind regime: does the train/deploy gap close?", "",
              "Blind = a candidate's signature is revealed only once measured, which is genuine "
              "prospective experimental design. GO facility location is the floor to beat (it uses "
              "no signature at all). `train scale` is one training sub-problem's candidate count / "
              "budget and the atlas it is drawn from; deployment is always 437 candidates, budget "
              "100 on this line. `val ÷ fac` is how far above facility location the selected "
              "checkpoint got on its own validation sub-problems, i.e. AT TRAINING SCALE -- the "
              "question is whether that advantage survives to the deployment column. "
              "`vs best floor` is the paired difference from the strongest static heuristic on the "
              "same split, over all three prior views, which is the floor a richer-prior policy "
              "has to clear.", ""]
        for ds in datasets:
            L += [f"### {SHORT[ds]}", "",
                  "| method | n | train scale | val ÷ fac | rep MRR | x rand | vs GO facility "
                  "(wins) | vs best floor (wins) |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"]
            for t in [t for t in blind_rows if t["dataset"] == ds]:
                s = t["train_scale"]
                sc = (f"{s['n_cand']}/{s['budget']} on {SHORT.get(s['train_dataset'], s['train_dataset'])}"
                      + (" + rich" if s.get("rich") else "")) if s else "static"
                v = f"{t['val_over_facility']:.3f}×" if t["val_over_facility"] else "—"
                L.append(f"| {LABEL.get(t['method'], t['method'])} | {t['n']} | {sc} | {v} | "
                         f"{t['mrr_mean']:.4f} ± {t['mrr_sd']:.4f} | {t['ratio_mean']:.2f}× | "
                         f"{t['vs_facility_mean']:+.4f} ({t['frac_beats_facility']:.0%}) | "
                         f"{t['vs_facility_best_mean']:+.4f} ({t['frac_beats_facility_best']:.0%}) |")
            L.append("")
    if diag:
        L += ["## Is the reward noisy? (critic diagnostics, held-out sub-problems)", "",
              "| correlation with the held-out one-step gain | mean | min | max |", "|---|---:|---:|---:|"]
        for k, nm in (("corr_emp_vs_heldout_gain", "greedy's empirical gain (visible genes)"),
                      ("corr_critic_vs_heldout_gain", "learned critic"),
                      ("corr_shrink_vs_heldout_gain", "shrinkage of the two")):
            L.append(f"| {nm} | {diag[k]['mean']:.3f} | {diag[k]['min']:.3f} | {diag[k]['max']:.3f} |")
        L.append("")
    if attrib_mean:
        L += ["## What the PPO policies learned", "",
              "Mean d(logit)/d(feature) over inputs drawn like the real ones, averaged over runs "
              "(± sd across runs); + means the policy prefers genes high on the feature. Features "
              "are z-scored per decision, so magnitudes are comparable.", ""]
        for k in sorted(attrib_mean):
            top = sorted(attrib_mean[k].items(), key=lambda kv: -abs(kv[1]))[:8]
            L.append(f"- **{k}** (n={len(attrib[k])}): " + ", ".join(
                f"{f} {v:+.2f}±{attrib_sd[k][f]:.2f}" for f, v in top))
        L.append("")
    if related_cmp:
        L += ["## Related work on the same task", "",
              "Published acquisition methods run by scripts/related_work.py on the identical splits, "
              "budget (100 of 437 in 4 rounds of 25), GO-kNN predictor and report genes, re-scored here "
              "with every other chosen set. Our own rows are in bold for reference. Comparisons are "
              "PAIRED BY SPLIT: a run minus the reference's mean on the same split, with the share of runs "
              "above it and a one-sample t over runs. `own (aux) MRR` is the reward the peek methods read.",
              ""]
        for ds in datasets:
            for regime in ("blind", "peek"):
                rows = [r for r in related_cmp if r["dataset"] == ds and r["regime"] == regime]
                if not any(not r["ours"] for r in rows):
                    continue
                refs = REFS[regime]
                L += [f"### {SHORT[ds]}, {regime}", "",
                      "| method | published as | n | rep MRR | own (aux) MRR | x rand | "
                      + " | ".join(f"vs {nm}" for _, nm in refs) + " |",
                      "|---|---|---:|---:|---:|---:|" + "---:|" * len(refs)]
                for r in rows:
                    cells = []
                    for k, _ in refs:
                        v = r["vs"].get(k)
                        cells.append("—" if v is None else f"{v['diff']:+.4f} ({v['wins']:.0%}, t={v['t']:+.1f})")
                    nm = LABEL.get(r["method"], r["method"])
                    L.append(f"| {'**' + nm + '**' if r['ours'] else nm} | {'ours' if r['ours'] else CITE.get(r['method'], '')} "
                             f"| {r['n']} | {r['mrr_mean']:.4f} ± {r['mrr_sd']:.4f} | {r['aux_mean']:.4f} | "
                             f"{r['ratio_mean']:.2f}× | " + " | ".join(cells) + " |")
                L.append("")
        if coverage:
            L += ["### Why the published blind methods lose: two candidate mechanisms", "",
                  "`GO mass` = mean over report genes of the summed GO Jaccard to their 10 nearest chosen "
                  "genes, the only input the GO-kNN predictor has; `orphans` = share of report genes with "
                  "no GO overlap to any chosen gene (predicted as the mean). A diagnostic, never seen by a "
                  "method. `ref coherence` = mean over the CHOSEN genes of rl_acquire's `coherence` feature "
                  "(GO-weighted signature cosine to their 10 GO neighbours in act + aux; PPO peek's strongest "
                  "feature): are the references genes whose GO neighbourhood predicts them? Correlations with "
                  "x rand are over every blind record (ours, published, floors).", ""]
            for ds, cv in coverage.items():
                L += [f"**{SHORT[ds]}**, over {cv['n_records']} blind records / {cv['n_methods']} method means: "
                      f"GO mass r = {cv['pearson_records']:+.2f} (Spearman {cv['spearman_records']:+.2f}) / "
                      f"{cv['pearson_method_means']:+.2f}; ref coherence r = {cv['coh_pearson_records']:+.2f} "
                      f"(Spearman {cv['coh_spearman_records']:+.2f}) / {cv['coh_pearson_method_means']:+.2f}.", "",
                      "| method | n | GO mass | orphans | ref coherence | x rand |", "|---|---:|---:|---:|---:|---:|"]
                for m, v in sorted(cv["per_method"].items(), key=lambda kv: -kv[1]["ratio"]):
                    L.append(f"| {LABEL.get(m, m)} | {v['n']} | {v['go_mass']:.3f} | {v['go_orphan']:.1%} | "
                             f"{v['ref_coh']:.3f} | {v['ratio']:.2f}× |")
                L.append("")
        L += ["### Implementation checks", ""]
        for m, d in rel_diag.items():
            bits = []
            if "replica_overlap_mean" in d:
                bits.append(f"numpy replica of IterPert's MaxDist picks {d['replica_overlap_mean']:.1%} of the "
                            f"same genes as IterPert's own code (min {d['replica_overlap_min']:.0%}, "
                            f"{d['n_rounds_checked']} rounds)")
            if d["n_rounds_with_status"]:
                bits.append(f"{d['n_rounds_with_status']} rounds reported a selection status: {d['statuses']}")
            for k, nm in (("cv_r2", "datamodel 5-fold CV R²"), ("frac_truncated", "share of TMC steps truncated"),
                          ("n_perm", "permutations"), ("min_samples_per_gene", "min marginal samples per gene"),
                          ("valuation_secs", "valuation seconds")):
                if k in d:
                    bits.append(f"{nm} {d[k]:.3g}")
            if bits:
                L.append(f"- **{LABEL.get(m, m)}** (n={d['n_runs']}): " + "; ".join(bits))
        if priors_check:
            for ds, v in (priors_check.get("datasets") or {}).items():
                cov = ", ".join(f"{k.replace('_kernel', '')} {c:.0%}" for k, c in v.get("coverage", {}).items())
                L.append(f"- IterPert priors on {SHORT.get(ds, ds)} (share of candidates the source covers, rest "
                         f"mean-imputed as in IterPert's own preprocessing): {cov}")
        L.append("")
    L += ["## Checks", "", "Fast scorer vs rl_headroom's own path (max |diff| over 30 random subsets):", ""]
    for k, v in sorted(chk.items()):
        L.append(f"- {k}: MRR {v['max_abs_diff_mrr']:.2e}, top-1 {v['max_abs_diff_top1']:.2e}")
    if fpc:
        n_ok = sum(fpc.values())
        L += ["", f"Generalised facility location reproduces `go_facility` exactly under the "
              f"Jaccard view: {n_ok}/{len(fpc)} splits" +
              ("." if n_ok == len(fpc) else
               " -- MISMATCH on " + ", ".join(k for k, v in fpc.items() if not v) + ".")]
    if env_diff:
        L += ["", f"Every chosen set is re-scored here in GAPS's env (numpy 2.x argpartition tie-breaking); "
              f"|MRR change| vs the training env: mean {np.mean(env_diff):.4f}, max {max(env_diff):.4f} "
              f"over {len(env_diff)} sets."]
    L += ["", f"Runs read: {len(runs)}. Errors: {len(errors)}."] + [f"- {e}" for e in errors]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    # ------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "figure.facecolor": SURF, "axes.facecolor": SURF})
    if datasets:
        rows = [m for m in BASE_ORDER + RL_ORDER if m in methods]
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets) + 1.6, 0.36 * len(rows) + 1.6),
                                 sharey=True, squeeze=False)
        for ax, ds in zip(axes[0], datasets):
            xs_ds = [x for x in recs if x["dataset"] == ds]
            p05 = np.mean([x["ref_p05"] / x["ref_med"] for x in xs_ds])
            p95 = np.mean([x["ref_p95"] / x["ref_med"] for x in xs_ds])
            ch = np.mean([x["ref_cheat"] / x["ref_med"] for x in xs_ds])
            ax.axvspan(p05, p95, color=GRID, zorder=0, lw=0)
            ax.axvline(1.0, color=MUTED, lw=1, zorder=1)
            ax.axvline(ch, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(ch, -0.9, f" best cheat run {ch:.2f}×", color=INK2, fontsize=8, va="center")
            ax.text((p05 + p95) / 2, -0.9, "random p05–p95", color=MUTED, fontsize=8, ha="center", va="center")
            for i, m in enumerate(rows):
                v = [x["ratio"] for x in xs_ds if x["method"] == m]
                if not v:
                    continue
                col = INK2 if m in BASE_ORDER else (C_BLIND if m in BLIND else C_PEEK)
                jit = np.linspace(-0.18, 0.18, len(v)) if len(v) > 1 else [0.0]
                ax.scatter(v, i + np.asarray(jit), s=12, color=col, alpha=0.55, lw=0, zorder=2)
                ax.scatter([np.mean(v)], [i], s=70, color=col, edgecolor=SURF, lw=1.5, zorder=3,
                           marker="D" if m in BASE_ORDER else "o")
                ax.text(max(np.max(v), np.mean(v)) + 0.02, i, f"{np.mean(v):.2f}×", va="center",
                        fontsize=8, color=INK2)
            ax.set_title(SHORT[ds], loc="left", fontsize=10, color=INK)
            ax.set_xlabel("report MRR ÷ random-x100 median (same split)")
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(len(rows) - 0.4, -1.4)
            if len(BASE_ORDER) < len(rows):
                ax.axhline(len([r for r in rows if r in BASE_ORDER]) - 0.5, color=GRID, lw=1)
        axes[0][0].set_yticks(range(len(rows)))
        axes[0][0].set_yticklabels([LABEL[m] for m in rows])
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="D", ls="", color=INK2, label="baseline"),
             Line2D([], [], marker="o", ls="", color=C_PEEK, label="RL, peek (same information as greedy)"),
             Line2D([], [], marker="o", ls="", color=C_BLIND, label="RL, blind (signature seen only once measured)")]
        fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Acquisition, budget 100 of 437: small dots = split × seed, large = mean",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        p = os.path.join(a.fig_dir, "fig15_rl_acquisition.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print("wrote", p)

    # fig17: published methods beside ours. Colour = information regime (as in fig15),
    # marker = who: diamond = static/greedy baseline, circle = our RL, square = published.
    if datasets and any(m in methods for m in RELATED):
        groups = [("Blind: a candidate's signature is seen only once it is measured",
                   [m for m in OWN["blind"] + REL_BLIND if m in methods]),
                  ("Peek: every non-report signature is visible (greedy's information)",
                   [m for m in OWN["peek"] + REL_PEEK if m in methods])]
        rows = []
        for title, ms in groups:
            rows.append(("hdr", title))
            rows += [("m", m) for m in ms]
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets) + 2.4, 0.30 * len(rows) + 1.5),
                                 sharey=True, squeeze=False)
        for ax, ds in zip(axes[0], datasets):
            xs_ds = [x for x in recs if x["dataset"] == ds]
            p05 = np.mean([x["ref_p05"] / x["ref_med"] for x in xs_ds])
            p95 = np.mean([x["ref_p95"] / x["ref_med"] for x in xs_ds])
            ax.axvspan(p05, p95, color=GRID, zorder=0, lw=0)
            ax.axvline(1.0, color=MUTED, lw=1, zorder=1)
            fb = np.mean([x["ref_facility_best"] / x["ref_med"] for x in xs_ds])
            ax.axvline(fb, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(fb, -0.9, f" best static floor {fb:.2f}×", color=INK2, fontsize=8, va="center")
            ax.text((p05 + p95) / 2, -0.9, "random p05–p95", color=MUTED, fontsize=8, ha="center",
                    va="center")
            for i, (kind, m) in enumerate(rows):
                if kind == "hdr":
                    continue
                v = [x["ratio"] for x in xs_ds if x["method"] == m]
                if not v:
                    continue
                base_like = m in BASE_ORDER
                col = INK2 if base_like else (C_BLIND if m in BLIND else C_PEEK)
                mk = "D" if base_like else ("s" if m in RELATED else "o")
                jit = np.linspace(-0.16, 0.16, len(v)) if len(v) > 1 else [0.0]
                ax.scatter(v, i + np.asarray(jit), s=10, color=col, alpha=0.5, lw=0, zorder=2, marker=mk)
                ax.scatter([np.mean(v)], [i], s=62, color=col, edgecolor=SURF, lw=1.5, zorder=3, marker=mk)
                ax.text(max(np.max(v), np.mean(v)) + 0.02, i, f"{np.mean(v):.2f}×", va="center",
                        fontsize=7.5, color=INK2)
            for i, (kind, _) in enumerate(rows):
                if kind == "hdr" and i > 0:
                    ax.axhline(i - 0.5, color=GRID, lw=1)
            ax.set_title(SHORT[ds], loc="left", fontsize=10, color=INK)
            ax.set_xlabel("report MRR ÷ random-x100 median (same split)")
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(len(rows) - 0.4, -1.4)
        axes[0][0].set_yticks(range(len(rows)))
        axes[0][0].set_yticklabels([t if k == "hdr" else LABEL.get(t, t) for k, t in rows])
        for lab, (k, _) in zip(axes[0][0].get_yticklabels(), rows):
            if k == "hdr":
                lab.set_fontweight("bold")
                lab.set_color(INK)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="D", ls="", color=INK2, label="static / greedy baseline"),
             Line2D([], [], marker="o", ls="", color=INK2, label="our RL"),
             Line2D([], [], marker="s", ls="", color=INK2, label="published method"),
             Line2D([], [], marker="o", ls="", color=C_BLIND, label="blind regime"),
             Line2D([], [], marker="o", ls="", color=C_PEEK, label="peek regime")]
        fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Published acquisition methods on the identical task: small marks = split × seed, large = mean",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        p = os.path.join(a.fig_dir, "fig17_related_work.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print("wrote", p)

    curves = sorted(glob.glob(os.path.join(runs_dir, "ppo_*.curve.jsonl")))
    if curves:
        fig, axes = plt.subplots(2, max(1, len(datasets)), figsize=(5.2 * max(1, len(datasets)), 6.4),
                                 squeeze=False, sharex=True)
        for j, ds in enumerate(datasets):
            for p in curves:
                nm = os.path.basename(p)
                if f"__{ds}__" not in nm:
                    continue
                me = nm.split("__")[0]
                sp = int(nm.split("__s")[1].split("__")[0])
                R = ref(ds, sp)
                c = load_jsonl(p)
                if not c or R is None:
                    continue
                it = [x["iter"] for x in c]
                col = C_BLIND if me in BLIND else C_PEEK
                st = CURVE_STYLE.get(me, "-")
                axes[0][j].plot(it, [x["val_mrr"] / x["facility"] for x in c], color=col, lw=1.2,
                                alpha=0.6, ls=st)
                axes[1][j].plot(it, [x["rep_mrr"] / R["med"] for x in c], color=col, lw=1.2,
                                alpha=0.6, ls=st)
            axes[0][j].axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
            axes[0][j].set_title(SHORT[ds], loc="left", fontsize=10)
            axes[0][j].set_ylabel("validation MRR ÷ facility location")
            axes[1][j].axhline(1.0, color=MUTED, lw=1)
            axes[1][j].set_ylabel("report MRR ÷ random median\n(diagnostic only, never selects)")
            axes[1][j].set_xlabel("PPO iteration")
            for ax in (axes[0][j], axes[1][j]):
                ax.grid(color=GRID, lw=0.6)
                ax.spines[["top", "right"]].set_visible(False)
        from matplotlib.lines import Line2D
        seen = {os.path.basename(p).split("__")[0] for p in curves}
        h = [Line2D([], [], color=C_PEEK, label="PPO peek")] + [
            Line2D([], [], color=C_BLIND, ls=CURVE_STYLE[m], label=LABEL[m])
            for m in BLIND_RL if m in seen]
        fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        p = os.path.join(a.fig_dir, "fig16_rl_curves.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
