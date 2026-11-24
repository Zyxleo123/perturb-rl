#!/usr/bin/env python
"""Report for the sim-to-real acquisition runs (scripts/rl_sim.py), from saved
artifacts only:
  <art>/rl_acquire/sim/baselines/<ds>__s<split>.jsonl   sim baselines + probes
  <art>/rl_acquire/sim/runs/*.json                      sim RL runs
  <art>/rl_acquire/{baselines,runs}/                    the existing floors and the
                                                        peek methods, for reference
Every chosen set is re-scored HERE with rl_headroom's own functions (the analysis env,
where GAPS ran), and the random x100 median is recomputed here too -- the same
measurement rl_report.py makes, so every ratio sits on the scale of GAPS.md's tables.
The script asserts that its go_facility row reproduces rl_report's summary.json on
every split before it prints a table (a reconciliation point).

Writes <art>/rl_acquire/sim/{report.md,summary.json}.
Usage: python -I rl_sim_report.py --art-dir D
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
OLD_BASE = ["go_facility", "go_facility_idf", "go_facility_emb", "greedy_aux_all", "cheat_rep_all"]
OLD_RUNS = ["pg_tab", "critic_shrink", "ppo_peek", "ppo_blind"]
_SB = ["sim_go_facility", "sim_sig_facility", "sim_greedy_aux_all"]
_SR = ["sim_pg_tab", "sim_pg_tab_boot", "sim_critic_only", "sim_critic_shrink", "sim_ppo_peek",
       "sim_greedy_combo", "sim_pg_combo", "sim_greedy_hybrid2", "sim_pg_hybrid2"]
# sim_*_hybrid (v1) is superseded (an empty reference set scored the random-set mean; see
# rl_sim.py) and deliberately absent from every table; its runs stay on disk.
# two simulators (rl_sim.POPS): `all` source genes, and `_tpop` = the deployment's population
SUFFIXES = ("_tpop", "_tpred", "_timp")
SIM_BASE = ["go_facility_cov"] + _SB + [m + x for x in SUFFIXES for m in _SB]
SIM_RL = _SR + [m + x for x in SUFFIXES for m in _SR]


def pop_of(m):
    return next((x for x in SUFFIXES if m.endswith(x)), "")
LABEL = {
    "go_facility": "GO facility location (static, blind; the floor)",
    "go_facility_idf": "facility, IDF-weighted GO (static, blind)",
    "go_facility_emb": "facility, GO embedding (static, blind)",
    "go_facility_cov": "GO facility, only candidates S measured",
    "sim_go_facility": "facility on S's GO view, covered candidates",
    "sim_sig_facility": "facility on S signatures (static, uses S)",
    "sim_greedy_aux_all": "greedy on the S reward (uses S)",
    "sim_pg_tab": "**REINFORCE in S** (sim-to-real)",
    "sim_pg_tab_boot": "REINFORCE in S, resampled reward",
    "sim_critic_only": "critic in S, greedy",
    "sim_critic_shrink": "critic-shrunk greedy in S",
    "sim_ppo_peek": "PPO feature policy trained in S",
    "sim_greedy_combo": "greedy on simulator MRR + T's GO facility objective",
    "sim_pg_combo": "**REINFORCE on simulator MRR + T's GO facility objective**",
    "sim_greedy_hybrid2": "greedy, combined reward over ALL candidates",
    "sim_pg_hybrid2": "**REINFORCE, combined reward over ALL candidates**",
    "ppo_blind": "PPO blind (T's GO only)",
    "greedy_aux_all": "greedy on T's own signatures (peek)",
    "pg_tab": "REINFORCE on T's own signatures (peek)",
    "critic_shrink": "critic-shrunk greedy (peek)",
    "ppo_peek": "PPO feature policy (peek)",
    "cheat_rep_all": "cheat: greedy on T's report genes",
}
# one label per (method, simulator): the simulator is named in brackets, never inferred
SIM_TAG = {"": "sim: S", "_tpop": "sim: S on T's genes", "_tpred": "sim: S on T's genes, T's GO",
           "_timp": "sim: T atlas + imputed candidates"}
for _x in SUFFIXES:
    for _m in _SB + _SR:
        LABEL[_m + _x] = LABEL[_m].replace("**", "") + f" [{SIM_TAG[_x]}]"


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


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
    sim = os.path.join(root, "sim")
    check = json.load(open(os.path.join(root, "cache_check.json")))

    sim_base, runs, errors = [], [], []
    for p in sorted(glob.glob(os.path.join(sim, "baselines", "*.jsonl"))):
        sim_base += load_jsonl(p)
    for p in sorted(glob.glob(os.path.join(sim, "runs", "*.json"))):
        if p.endswith(".error.json"):
            errors += [f"{os.path.basename(p)}: {r.get('error')}" for r in json.load(open(p))]
        else:
            runs += json.load(open(p))
    keys = sorted({(r["dataset"], r["split"]) for r in sim_base if r["strategy"] == "bridge"})
    # PJ_SIM_REPORT_SPLITS restricts the report to some splits (e.g. the held-out
    # confirmation splits), written under PJ_SIM_REPORT_TAG so the two never mix
    # default: the RL grid's own splits (PJ_RLA_SPLITS, 4 104 204 in env/common.sh)
    want = [int(x) for x in (os.environ.get("PJ_SIM_REPORT_SPLITS") or os.environ.get("PJ_RLA_SPLITS", "")).split()]
    tag = os.environ.get("PJ_SIM_REPORT_TAG", "")
    if want:
        keys = [k for k in keys if k[1] in want]
        sim_base = [r for r in sim_base if r["split"] in want]
        runs = [r for r in runs if r["split"] in want]
    if not keys:
        print("GATE NOT MET: no sim baselines on disk (did jobs/60_sim_base run? NFS can hide fresh files ~10 min)")
        return 0
    old_base, fresh = [], set()
    for ds, sp in keys:
        p = os.path.join(root, "baselines", f"{ds}__s{sp}.jsonl")
        if os.path.exists(p):
            old_base += load_jsonl(p)
        else:                    # a confirmation split: its floors come from the sim baselines
            fresh.add((ds, sp))
    old_runs = []
    for p in sorted(glob.glob(os.path.join(root, "runs", "*.json"))):
        if not p.endswith(".error.json"):
            old_runs += [r for r in json.load(open(p)) if r.get("method") in OLD_RUNS]

    # ---------------------------------------------------- re-score in GAPS's env
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
            reps[(ds, sp)] = (RA.Reporter(worlds[ds], dep["rep"], dep["act"]), dep)
        return reps[(ds, sp)]

    recs, drift = [], []

    def add(r, method):
        if (r["dataset"], r["split"]) not in keys or not r.get("chosen"):
            return
        fn, _ = reporter(r["dataset"], r["split"])
        s = fn(r["chosen"])
        if "rep_mrr" in r:
            drift.append(abs(s["mrr"] - r["rep_mrr"]))
        recs.append({"method": method, "dataset": r["dataset"], "split": r["split"], "seed": r.get("seed"),
                     "mrr": s["mrr"], "top1": s["top1"], "s_rep": r.get("s_rep_mrr"), "s_own": r.get("s_own_mrr"),
                     "n_chosen": len(r["chosen"]), "diag": r.get("diag")})

    for r in old_base:
        if r["strategy"] in OLD_BASE:
            add(r, r["strategy"])
    for r in old_runs:
        add(r, r["method"])
    for r in sim_base:
        if r["strategy"] in SIM_BASE:
            add(r, r["strategy"])
        elif r["strategy"] in OLD_BASE[:3] and (r["dataset"], r["split"]) in fresh:
            add(r, r["strategy"])
    for r in runs:
        add(r, r["method"])
    for ds, sp in keys:
        fn, dep = reporter(ds, sp)
        rand[(ds, sp)] = RA.random_dist(fn, dep["act"], RA.B_DEPLOY)
    print(f"re-scored {len(recs)} chosen sets; |MRR change vs train env| mean {np.mean(drift):.5f} "
          f"max {np.max(drift):.5f}")

    # reconciliation points against rl_report's summary.json (the numbers GAPS.md prints):
    # each split's random median, and each line's GO facility mean over its splits
    recon = {}
    sj = os.path.join(root, "summary.json")
    if os.path.exists(sj):
        S = json.load(open(sj))
        for ds, sp in keys:
            old = (S.get("random_gaps_env") or {}).get(f"{ds}__s{sp}")
            if old:
                recon[f"random__{ds}__s{sp}"] = abs(rand[(ds, sp)]["rep_mrr"] - old["rep_mrr"])
        for ds in {k[0] for k in keys}:
            sps = sorted(k[1] for k in keys if k[0] == ds)
            old = [t for t in S.get("table", []) if t["dataset"] == ds and t["method"] == "go_facility"]
            if old and sorted(old[0].get("splits", [])) == sps:
                mine = np.mean([x["mrr"] for x in recs if x["dataset"] == ds and x["method"] == "go_facility"])
                recon[f"go_facility__{ds}"] = abs(float(mine) - old[0]["mrr_mean"])
    recon_ok = bool(recon) and max(recon.values()) < 1e-9
    print(f"reconciliation vs rl_report summary.json (go_facility): {recon} -> {'OK' if recon_ok else 'NOT CHECKED/FAILED'}")

    # ---------------------------------------------------- relative to each split
    def split_mean(ds, sp, m):
        v = [x["mrr"] for x in recs if x["dataset"] == ds and x["split"] == sp and x["method"] == m]
        return float(np.mean(v)) if v else np.nan

    refs = {}
    for ds, sp in keys:
        floors = [split_mean(ds, sp, m) for m in ("go_facility", "go_facility_idf", "go_facility_emb")]
        refs[(ds, sp)] = {"med": rand[(ds, sp)]["rep_mrr"], "p95": rand[(ds, sp)]["rep_mrr_p95"],
                          "floor": float(np.nanmax(floors)),
                          "fac_cov": split_mean(ds, sp, "go_facility_cov"),
                          "sim_greedy": split_mean(ds, sp, "sim_greedy_aux_all"),
                          "sim_greedy_tpop": split_mean(ds, sp, "sim_greedy_aux_all_tpop"),
                          "sim_greedy_tpred": split_mean(ds, sp, "sim_greedy_aux_all_tpred"),
                          "sim_greedy_timp": split_mean(ds, sp, "sim_greedy_aux_all_timp"),
                          **{"greedy_combo" + x: split_mean(ds, sp, "sim_greedy_combo" + x) for x in ("",) + SUFFIXES},
                          "greedy_hybrid2": split_mean(ds, sp, "sim_greedy_hybrid2"),
                          "sig_fac": split_mean(ds, sp, "sim_sig_facility"),
                          "sig_fac_tpop": split_mean(ds, sp, "sim_sig_facility_tpop"),
                          "sig_fac_tpred": split_mean(ds, sp, "sim_sig_facility_tpred"),
                          "sig_fac_timp": split_mean(ds, sp, "sim_sig_facility_timp"),
                          "peek_pg": split_mean(ds, sp, "pg_tab"),
                          "cheat": split_mean(ds, sp, "cheat_rep_all")}
    for x in recs:
        R = refs[(x["dataset"], x["split"])]
        x["ratio"] = x["mrr"] / R["med"]
        x["above_p95"] = bool(x["mrr"] > R["p95"])
        # a simulator's methods face THAT simulator's greedy and S-signature facility
        for k in ("floor", "fac_cov", "sim_greedy", "sig_fac", "peek_pg"):
            kk = k + pop_of(x["method"]) if k in ("sim_greedy", "sig_fac") else k
            if k == "sim_greedy" and "_combo" in x["method"]:   # combo faces greedy on the SAME combined reward
                kk = "greedy_combo" + pop_of(x["method"])
            if k == "sim_greedy" and "_hybrid2" in x["method"]:
                kk = "greedy_hybrid2"
            x[f"vs_{k}"] = x["mrr"] - R[kk]
        x["cheat_frac"] = (x["mrr"] - R["med"]) / (R["cheat"] - R["med"])

    order = (OLD_BASE[:3] + ["go_facility_cov", "ppo_blind"] + _SB + _SR
             + [m + x for x in SUFFIXES for m in _SB + _SR]
             + ["greedy_aux_all", "pg_tab", "critic_shrink", "ppo_peek", "cheat_rep_all"])
    table = []
    for ds in [d for d in SHORT if any(k[0] == d for k in keys)]:
        for m in order:
            xs = [x for x in recs if x["dataset"] == ds and x["method"] == m]
            if not xs:
                continue
            f = lambda k: np.array([x[k] for x in xs], float)  # noqa: E731
            row = {"dataset": ds, "method": m, "n": len(xs), "ratio": float(f("ratio").mean()),
                   "mrr": float(f("mrr").mean()), "top1": float(f("top1").mean()),
                   "frac_above_p95": float(f("above_p95").mean()), "cheat_frac": float(f("cheat_frac").mean())}
            for k in ("floor", "fac_cov", "sim_greedy", "sig_fac", "peek_pg"):
                d = f(f"vs_{k}")
                row[f"vs_{k}"] = {"mean": float(np.nanmean(d)), "wins": int((d > 0).sum()), "n": int(d.size),
                                  "t": tstat(d)}
            s = [x["s_rep"] for x in xs if x.get("s_rep") is not None]
            row["s_rep_mean"] = float(np.mean(s)) if s else None
            table.append(row)

    # transfer probes, and how much of the S-side gain over greedy survives in T
    probes = {f"{r['dataset']}__s{r['split']}{pop_of(r['strategy'])}": {
        k: r[k] for k in ("spearman_s_own_vs_t_rep", "spearman_s_rep_vs_t_rep", "spearman_s_own_vs_s_rep")}
        for r in sim_base if r["strategy"] in ("transfer_probe",) + tuple("transfer_probe" + x for x in SUFFIXES)}
    transfer = {}
    for ds in SHORT:
        for m in SIM_RL:
            xs = [x for x in recs if x["dataset"] == ds and x["method"] == m and x.get("s_rep") is not None]
            if not xs:
                continue
            g_s = {sp: np.mean([x["s_rep"] for x in recs if x["dataset"] == ds and x["split"] == sp
                                and x["method"] == "sim_greedy_aux_all" + pop_of(m)])
                   for sp in {x["split"] for x in xs}}
            ds_ = np.mean([x["s_rep"] - g_s[x["split"]] for x in xs])
            dt_ = np.mean([x["vs_sim_greedy"] for x in xs])
            transfer[f"{ds}__{m}"] = {"s_side_gain_over_greedy": float(ds_), "t_side_gain_over_greedy": float(dt_)}
    reconc = [r for r in sim_base if r["strategy"] in ("random", "go_facility", "facility_helper_check",
                                                        "scorer_check_s", "done")]
    bridges = {f"{r['dataset']}__s{r['split']}__{r.get('pop', 'all')}": {
        k: r.get(k) for k in ("act_covered", "rep_covered_excluded_from_S", "aux_s", "n_shared_genes",
                              "atlas_median_cos_recentred")}   # timp bridges carry a different set
        for r in sim_base if r["strategy"] == "bridge"}
    expected = len(keys) * len(SIM_RL) * 3  # 3 seeds, both simulators
    have = sum(1 for x in recs if x["method"] in SIM_RL)
    summary = {"table": table, "probes": probes, "transfer": transfer, "bridges": bridges,
               "reconciliation_vs_rl_report": recon, "reconciliation_ok": recon_ok,
               "baseline_reconciliation": [{k: r.get(k) for k in ("dataset", "split", "strategy", "reconciled",
                                                                   "matches_go_facility", "ok",
                                                                   "reconciliation_failures")}
                                           for r in reconc],
               "rescore_drift_max": float(np.max(drift)), "errors": errors,
               "sim_rl_runs_found": have, "sim_rl_runs_expected_at_3_seeds": expected,
               "records": recs, "random": {f"{k[0]}__s{k[1]}": v for k, v in rand.items()}}
    with open(os.path.join(sim, f"summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    # ---------------------------------------------------- report.md
    L = ["# Sim-to-real acquisition: the other cell line as the simulator", "",
         f"Splits: {sorted({k[1] for k in keys})}. Sim RL rows found: {have} (expected {expected} at 3 seeds). "
         f"Errors: {len(errors)}. Re-score drift max {np.max(drift):.4f}. "
         f"go_facility reconciles with rl_report: {'yes' if recon_ok else 'NO / not checked'}.", ""]
    for ds in SHORT:
        rows = [r for r in table if r["dataset"] == ds]
        if not rows:
            continue
        L += [f"## {SHORT[ds]}", "",
              "| method | n | × random | top-1 | > p95 | vs best static floor (Δ, wins, t) "
              "| vs greedy on the same simulator's reward (Δ, wins, t) | vs GO facility, covered only (Δ, wins, t) |",
              "|---|---:|---:|---:|---:|---|---|---|"]
        for r in rows:
            def c(k):
                v = r[f"vs_{k}"]
                return f"{v['mean']:+.4f} ({v['wins']}/{v['n']}, {v['t']:+.1f})" if np.isfinite(v["mean"]) else "—"
            L.append(f"| {LABEL.get(r['method'], r['method'])} | {r['n']} | {r['ratio']:.2f}× | "
                     f"{100 * r['top1']:.1f}% | {100 * r['frac_above_p95']:.0f}% | {c('floor')} | "
                     f"{c('sim_greedy')} | {c('fac_cov')} |")
        L.append("")
    L += ["## Does the simulator rank sets the way the target does?", "",
          "Spearman over 100 random covered subsets of 100 genes.", "",
          "| split | S own reward vs T report | S report vs T report | S own vs S report |", "|---|---:|---:|---:|"]
    for k, v in probes.items():
        fm = lambda x: "—" if x is None else f"{x:+.3f}"  # noqa: E731  (timp has no S-side report)
        L.append(f"| {k} | {fm(v['spearman_s_own_vs_t_rep'])} | {fm(v['spearman_s_rep_vs_t_rep'])} | "
                 f"{fm(v['spearman_s_own_vs_s_rep'])} |")
    L += ["", "## How much of the simulator's gain over greedy survives in the target", "",
          "| line · method | S-side report gain over greedy | T-side report gain over greedy |", "|---|---:|---:|"]
    for k, v in transfer.items():
        L.append(f"| {k} | {v['s_side_gain_over_greedy']:+.4f} | {v['t_side_gain_over_greedy']:+.4f} |")
    L += ["", "## Bridges (candidates S measured; S rows of T's report genes, excluded)", ""]
    L += [f"- {k}: {v}" for k, v in bridges.items()]
    if errors:
        L += ["", "## Errors", ""] + [f"- {e}" for e in errors]
    with open(os.path.join(sim, f"report{tag}.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
