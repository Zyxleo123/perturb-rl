#!/usr/bin/env python
"""Is REINFORCE just a black-box optimiser? Peek regime, from saved artifacts only.

REINFORCE on per-gene logits beats greedy on greedy's own reward (GAPS.md "RL on the
acquisition gap"). The obvious objection: greedy is a weak optimiser of a
non-submodular set objective, so compare against classical optimisers given the same
reward and the same number of reward evaluations (rl_acquire.anneal, greedy_swap).

For every chosen set this script recomputes, in the analysis env, both
  own     the reward the optimiser maximised (GO-kNN retrieval MRR on the split's aux
          genes -- what train_pg_tab, anneal and greedy_swap all optimise), and
  report  MRR on the split's report genes, which no optimiser sees, / the split's
          random x100 median (recomputed here, as rl_report does).
"Better optimiser" is read on own; "better acquisition" on report.

Writes <art>/rl_acquire/opt/{report,summary}_s<first>-<last split>.{md,json}.
Usage: python -I opt_report.py --art-dir D
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own scripts/ only
import numpy as np  # noqa: E402

import rl_acquire as RA  # noqa: E402

SHORT = {"replogle_k562_essential": "K562", "replogle_rpe1_essential": "RPE1"}
BASE = ["go_facility", "greedy_aux_all"]
RUNS = ["pg_tab", "pg_tab_boot", "critic_shrink", "sa", "sa_cold", "sa_hot", "greedy_swap",
        "cem", "cem_slow", "cem_fast", "pg_tab_nobase", "ppo_peek_xline",
        "pg_ent01", "pg_ent03", "pg_ent10", "pg_wide", "pg_wide_ent03", "cem_wide", "cem_wide_slow", "cem_wide_fast"]
# published methods (scripts/related_work.py, artifacts/rl_acquire/related/runs)
RELATED = ["dv_shapley", "dv_banzhaf", "dv_datamodels", "rep_facility_sig", "rep_coreset_sig", "iterpert"]
# derived: greedy over every candidate, from greedy_swap's own starting point (splits with
# no rl_acquire baselines file), and the best SA schedule per split CHOSEN ON THE REPORT
# GENES -- an oracle choice that favours annealing, never RL
LABEL = {"go_facility": "GO facility location (static, blind floor)",
         "greedy_aux_all": "greedy, every candidate (~40k evaluations)",
         "pg_tab": "**REINFORCE, per-gene logits (19.2k evaluations)**",
         "pg_tab_boot": "REINFORCE, resampled reward (19.2k)",
         "critic_shrink": "critic-shrunk greedy",
         "sa": "simulated annealing, swap moves (19.2k)",
         "greedy_swap": "greedy + swap local search (~40k + 19.2k)",
         "sa_cold": "simulated annealing, T0 x 0.1 (19.2k)",
         "sa_hot": "simulated annealing, T0 x 10 (19.2k)",
         "greedy_from_swap": "greedy, every candidate (greedy_swap's start)",
         "sa_best_oracle": "simulated annealing, best of 3 schedules per split (oracle on report)",
         "cem": "cross-entropy method, alpha 0.1 (19.2k)",
         "cem_slow": "cross-entropy method, alpha 0.03 (19.2k)",
         "cem_fast": "cross-entropy method, alpha 0.3 (19.2k)",
         "cem_best_oracle": "cross-entropy method, best of 3 rates per split (oracle on report)",
         "pg_tab_nobase": "REINFORCE without its baseline (ablation, 19.2k)",
         "ppo_peek_xline": "**PPO trained on the OTHER line, deployed zero-shot (0 evaluations here)**",
         "dv_shapley": "TMC Data Shapley (Ghorbani & Zou 2019; 19.2k)",
         "dv_banzhaf": "Data Banzhaf (Wang & Jia 2023; 19.2k)",
         "dv_datamodels": "datamodels (Ilyas et al. 2022; 19.2k)",
         "dv_best_oracle": "data valuation, best of 3 per split (oracle on report)",
         "rep_facility_sig": "exemplar facility location on signatures (static)",
         "rep_coreset_sig": "k-centre coreset on signatures (static)",
         "iterpert": "IterPert (Huang et al. 2024; blind, context only)",
         "pg_ent01": "max-entropy REINFORCE, c 0.1", "pg_ent03": "max-entropy REINFORCE, c 0.3",
         "pg_ent10": "max-entropy REINFORCE, c 1.0", "pg_wide": "REINFORCE, wide reward (aux + unchosen candidates)",
         "pg_wide_ent03": "max-entropy REINFORCE c 0.3, wide reward",
         "cem_wide": "CEM alpha 0.1, wide reward", "cem_wide_slow": "CEM alpha 0.03, wide reward",
         "cem_wide_fast": "CEM alpha 0.3, wide reward",
         "cem_wide_best_oracle": "CEM, wide reward, best of 3 rates per split (oracle on report)"}
DERIVED = ["greedy_from_swap", "sa_best_oracle", "cem_best_oracle", "dv_best_oracle", "cem_wide_best_oracle"]
ORACLES = {"sa_best_oracle": ("sa", "sa_cold", "sa_hot"), "cem_best_oracle": ("cem", "cem_slow", "cem_fast"),
           "dv_best_oracle": ("dv_shapley", "dv_banzhaf", "dv_datamodels"),
           "cem_wide_best_oracle": ("cem_wide", "cem_wide_slow", "cem_wide_fast")}
# (reference, the methods it is compared against): reference minus each, paired by (split, seed)
COMPARE = [("pg_tab", ["greedy_aux_all", "greedy_from_swap", "sa", "sa_best_oracle", "greedy_swap",
                       "cem_best_oracle", "pg_tab_nobase", "dv_best_oracle", "dv_shapley", "dv_banzhaf",
                       "dv_datamodels", "rep_facility_sig", "rep_coreset_sig", "iterpert"]),
           # the RPE1 work: each REINFORCE variant against the strongest sampling baselines
           *[(v, ["pg_tab", "cem_best_oracle", "cem_wide_best_oracle", "dv_best_oracle", "sa_best_oracle"])
             for v in ("pg_ent01", "pg_ent03", "pg_ent10", "pg_wide", "pg_wide_ent03")],
           ("ppo_peek_xline", ["greedy_aux_all", "greedy_from_swap", "sa_best_oracle", "greedy_swap",
                               "cem_best_oracle", "pg_tab", "dv_best_oracle", "iterpert"])]


def tstat(d):
    d = np.asarray(d, float)
    if d.size < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    a = ap.parse_args()
    root = os.path.join(a.art_dir, "rl_acquire")
    od = os.path.join(root, "opt")
    os.makedirs(od, exist_ok=True)
    check = json.load(open(os.path.join(root, "cache_check.json")))
    splits = [int(x) for x in os.environ.get("PJ_RLA_SPLITS", "4 104 204").split()]
    raw = []
    for ds in RA.DATASETS:
        for sp in splits:
            p = os.path.join(root, "baselines", f"{ds}__s{sp}.jsonl")
            if os.path.exists(p):
                raw += [dict(json.loads(x), method=json.loads(x)["strategy"]) for x in open(p) if x.strip()]
    errors = []
    for p in sorted(glob.glob(os.path.join(root, "runs", "*.json"))):
        if p.endswith(".error.json"):
            errors += [os.path.basename(p)]
            continue
        raw += [r for r in json.load(open(p)) if r.get("method") in RUNS and r.get("split") in splits]
    for p in sorted(glob.glob(os.path.join(root, "related", "runs", "*.json"))):
        x = json.load(open(p))
        for r in (x if isinstance(x, list) else [x]):
            if r.get("method") in RELATED and r.get("split") in splits and not r.get("smoke"):
                raw.append(r)
    raw += [dict(r, method="greedy_from_swap", chosen=r["greedy_chosen"]) for r in raw
            if r.get("method") == "greedy_swap" and r.get("greedy_chosen")]
    raw = [r for r in raw if r.get("method") in BASE + RUNS + RELATED + ["greedy_from_swap"] and r.get("chosen")]
    have = sorted({r["method"] for r in raw})
    if not set(have) & {"sa", "greedy_swap", "cem", "ppo_peek_xline", "pg_tab_nobase"} | set(RELATED):
        print("GATE NOT MET: no optimiser-comparison runs on disk yet (NFS can hide fresh files ~10 min)")
        return 0

    recs, rand = [], {}
    for ds in RA.DATASETS:
        W = RA.World(check["datasets"][ds]["path"])
        for sp in splits:
            dep = RA.split_of(W.K, sp)
            rep = RA.Reporter(W, dep["rep"], dep["act"])
            own = RA.RewardSet(W, dep["aux"])
            rand[(ds, sp)] = RA.random_dist(rep, dep["act"], RA.B_DEPLOY)["rep_mrr"]
            for r in raw:
                if r["dataset"] != ds or r["split"] != sp:
                    continue
                ch = np.asarray(r["chosen"], np.int64)
                recs.append({"method": r["method"], "dataset": ds, "split": sp, "seed": r.get("seed", 0),
                             "own": float(own.mrr(ch)), "mrr": float(rep(ch)["mrr"]),
                             "ratio": float(rep(ch)["mrr"]) / rand[(ds, sp)],
                             "own_logged": r.get("own_reward")})
    drift = [abs(x["own"] - x["own_logged"]) for x in recs if x["own_logged"] is not None]
    # oracles: per (dataset, split), the setting whose seed-mean REPORT MRR is highest --
    # a choice made on the held-out genes, so it can only flatter the baseline
    for om, members in ORACLES.items():
        for ds in RA.DATASETS:
            for sp in splits:
                best = None
                for m in members:
                    xs = [x for x in recs if x["dataset"] == ds and x["split"] == sp and x["method"] == m]
                    if xs and (best is None or np.mean([x["mrr"] for x in xs]) > best[0]):
                        best = (np.mean([x["mrr"] for x in xs]), xs)
                if best:
                    recs += [dict(x, method=om, own_logged=None) for x in best[1]]

    table, tests = [], []
    for ds in RA.DATASETS:
        for m in BASE + RUNS + RELATED + DERIVED:
            xs = [x for x in recs if x["dataset"] == ds and x["method"] == m]
            if not xs:
                continue
            table.append({"dataset": ds, "method": m, "n": len(xs), "own_mean": float(np.mean([x["own"] for x in xs])),
                          "ratio_mean": float(np.mean([x["ratio"] for x in xs])),
                          "mrr_mean": float(np.mean([x["mrr"] for x in xs]))})
        for refm, others in COMPARE:
            ref = {(x["split"], x["seed"]): x for x in recs if x["dataset"] == ds and x["method"] == refm}
            for m in others:
                for key in ("own", "mrr"):
                    d = np.array([ref[(x["split"], x["seed"])][key] - x[key] for x in recs
                                  if x["dataset"] == ds and x["method"] == m and (x["split"], x["seed"]) in ref])
                    if not d.size:
                        continue
                    # split level: average seeds first (3 seeds share one split's report genes)
                    sp_d = []
                    for sp in splits:
                        a_ = [ref[(sp, x["seed"])][key] - x[key] for x in recs if x["dataset"] == ds and
                              x["method"] == m and x["split"] == sp and (sp, x["seed"]) in ref]
                        if a_:
                            sp_d.append(float(np.mean(a_)))
                    tests.append({"dataset": ds, "ref": refm, "vs": m, "metric": key, "n_runs": int(d.size),
                                  "t_splits": tstat(sp_d), "mean_diff": float(d.mean()), "wins": int((d > 0).sum()),
                                  "t_runs": tstat(d), "split_diffs": sp_d, "split_wins": int(sum(v > 0 for v in sp_d))})

    L = ["# RL vs classical and distribution-based optimisers (peek regime)", "",
         f"Splits {splits}, seeds 0-2. Every set re-scored here; own-reward drift vs the training "
         f"env: mean {np.mean(drift):.5f}, max {np.max(drift):.5f}." if drift else "", "",
         "| method | K562 own reward | K562 report ÷ random | RPE1 own reward | RPE1 report ÷ random |",
         "|---|---:|---:|---:|---:|"]
    for m in BASE + RUNS + RELATED + DERIVED:
        cells = []
        for ds in RA.DATASETS:
            t = next((t for t in table if t["dataset"] == ds and t["method"] == m), None)
            cells += [f"{t['own_mean']:.4f}" if t else "—", f"{t['ratio_mean']:.3f}×" if t else "—"]
        L.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    for refm, others in COMPARE:
        if not any(t["ref"] == refm for t in tests):
            continue
        L += ["", f"{LABEL[refm].replace('**', '')} minus each method, paired by (split, seed): mean difference "
              "(runs won / runs, run-level t; splits won / splits after averaging seeds, split-level t).", "",
              "| vs | metric | K562 | RPE1 |", "|---|---|---|---|"]
        for m in others:
            for key in ("own", "mrr"):
                cells = []
                for ds in RA.DATASETS:
                    t = next((t for t in tests if t["dataset"] == ds and t["ref"] == refm and t["vs"] == m
                              and t["metric"] == key), None)
                    cells.append(f"{t['mean_diff']:+.4f} ({t['wins']}/{t['n_runs']}, t={t['t_runs']:+.1f}; "
                                 f"{t['split_wins']}/{len(t['split_diffs'])} splits, split t={t['t_splits']:+.1f})"
                                 if t else "—")
                if any(c != "—" for c in cells):
                    L.append(f"| {LABEL[m]} | {'own reward' if key == 'own' else 'report MRR'} | " + " | ".join(cells) + " |")
    if errors:
        L += ["", f"Error files on disk: {errors}"]
    tag = f"_s{splits[0]}-{splits[-1]}"      # one file per split set, so reports never overwrite
    with open(os.path.join(od, f"summary{tag}.json"), "w") as f:
        json.dump({"splits": splits, "table": table, "tests": tests, "random": {f"{k[0]}__s{k[1]}": v for k, v in rand.items()},
                   "records": recs, "errors": errors}, f, indent=1)
    with open(os.path.join(od, f"report{tag}.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
