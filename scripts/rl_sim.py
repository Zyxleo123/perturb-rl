#!/usr/bin/env python
"""Sim-to-real gene acquisition: the OTHER cell line's screen is the simulator.

The question (GAPS.md, after "The combiner reads inner products"). Every blind
acquisition method, ours and published, ties a static GO facility-location
heuristic (1.23x / 1.30x random), and the only input that ever moved the unseen-gene
metric was a MEASUREMENT from another cell line. So: when a screen already exists in
one line (the source S) and we must choose which 100 of 437 genes to measure in a new
line (the target T), can RL use S as a simulator and choose better than prior
knowledge does?

The deployment task is unchanged, number for number: GAPS part 4's splits of T, the
GO-kNN predictor built from the chosen genes' REAL T signatures, retrieval MRR on T's
report genes, divided by T's own random-x100 median. Only what the chooser may look at
changes:
  visible  T's GO annotations and T's measured atlas (aux), as in the blind regime;
           S's whole screen, every gene S measured.
  hidden   every T signature of a candidate, and everything about T's report genes.

The simulator. A candidate is mapped to the same gene in S. The RL methods are the
peek-regime methods of rl_acquire.py run INSIDE S, on the S signatures of exactly the
deployment candidates:
  act_S  the candidates S measured (a candidate S never measured cannot be simulated
         and is not offered; baselines restricted the same way measure that cost)
  rep_S  the S rows of T's report genes -- EXCLUDED from training, reward and
         validation alike, so nothing about a report gene is read, not even its
         other-line signature. Used only for the S-side diagnostic below.
  aux_S  every other gene S measured: S's reward / visible atlas.
The chosen S genes map back to T, and the set is scored in T by rl_headroom's own
functions (rl_acquire.Reporter), exactly like every RL number in GAPS.md.

  sim_pg_tab        REINFORCE on per-gene Plackett-Luce logits, reward = S-side MRR
  sim_pg_tab_boot   ... resampling half the S reward genes every episode
  sim_critic        fitted one-step critic, used greedily / as greedy's control variate
  sim_ppo_peek      PPO feature policy trained on S sub-problems
Baselines with the same information (`baselines` mode): greedy on the identical S
reward (multi-seed: a greedy run is one draw), facility location on S signatures, and
GO facility location over all candidates and over the covered ones only.

Reconciliation points, asserted in the baselines job: T's random-x100 median and T's GO
facility set must reproduce artifacts/rl_acquire/baselines exactly; the facility
helper must reproduce rl_acquire.go_facility under GO Jaccard; the fast scorer must
agree with Reporter inside S.

Mechanism diagnostic (the quantity the sim-to-real hypothesis is about): over 100
random covered subsets, how well does the S-side score of a set rank its T-side score?
If set quality does not transfer across lines, no simulator-trained policy can win, and
this column says so independently of any policy.

Usage:
  python -I rl_sim.py baselines --cache-dir D --out-dir D --task-index I
  python -I rl_sim.py train     --cache-dir D --out-dir D --task-index I --threads N
  python -I rl_sim.py grid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own scripts/ only
import numpy as np  # noqa: E402

import rl_acquire as RA  # noqa: E402

BASE_METHODS = ("sim_pg_tab", "sim_pg_tab_boot", "sim_critic", "sim_ppo_peek")
# Two simulators, differing only in WHICH source genes the reward is computed on:
#   all   every gene S measured outside the candidates and T's report genes (the first
#         run). On K562 that is 1,065 RPE1 genes, 695 of which K562 never screened, so
#         the simulator mostly rewards predicting genes outside the deployment task.
#   tpop  only S's rows of T's own measured atlas (aux): the deployment's gene
#         population, as close as S can get to "genes T will be asked to predict".
#         Still excludes T's report genes, so nothing about them is read.
#   tpred tpop, and the simulator's kNN uses T's OWN GO Jaccard (the deployment
#         predictor) instead of S's: GO terms are filtered per line, so S's Jaccard is a
#         slightly different predictor. Every gene tpop touches is a T gene, so T's GO
#         row exists for all of them. Only the SIGNATURES then come from S.
# Suffix "_tpop" / "_tpred" on a method or baseline name selects the second / third.
POPS = ("all", "tpop", "tpred")
METHODS = BASE_METHODS + tuple(m + "_" + p for p in POPS[1:] for m in BASE_METHODS)


def split_pop(name):
    """'sim_pg_tab_tpred' -> ('sim_pg_tab', 'tpred'); no suffix -> 'all'."""
    for p in POPS[1:]:
        if name.endswith("_" + p):
            return name[:-len(p) - 1], p
    return name, "all"


def hybrid_world(WT, WS):
    """S's signatures and queries under T's GO view: each shared gene's GO row is T's,
    so the GO-kNN inside the simulator is the deployment predictor. Genes S measured
    and T did not get an all-zero GO row; under `tpred` none of them is ever a
    candidate, a reward gene or a visible gene."""
    import copy
    WH = copy.copy(WS)
    tpos = {str(g): i for i, g in enumerate(WT.cands)}
    F = np.zeros((WS.K, WT.F.shape[1]), np.float32)
    for s, g in enumerate(WS.cands):
        t = tpos.get(str(g))
        if t is not None:
            F[s] = WT.F[t]
    WH.F = F
    WH.JJ = RA.RH.jaccard(F, F)
    WH.deg = np.log1p(F.sum(1)).astype(np.float32)
    WH._rich = None
    WH.name = f"{WS.name}[signatures]+{WT.name}[GO]"
    return WH
N_PROBE = 100


def log(*a):
    RA.log(*a)


class Bridge:
    """One deployment split of T and its image in the source line S."""

    def __init__(self, WT, WS, split_seed, pop="all"):
        if pop == "tpred":
            WS = hybrid_world(WT, WS)
        self.WT, self.WS, self.sp, self.pop = WT, WS, split_seed, pop
        self.dep_t = dt = RA.split_of(WT.K, split_seed)
        spos = {str(g): i for i, g in enumerate(WS.cands)}
        self.t2s = np.array([spos.get(str(g), -1) for g in WT.cands], np.int64)
        self.s2t = {int(s): int(t) for t, s in enumerate(self.t2s) if s >= 0}
        cov = self.t2s[dt["act"]] >= 0
        self.act_t_cov = dt["act"][cov]
        rs = self.t2s[dt["rep"]]
        act_s = np.sort(self.t2s[self.act_t_cov])
        rep_s = np.sort(rs[rs >= 0])
        if pop == "all":
            aux_s = np.setdiff1d(np.arange(WS.K), np.concatenate([act_s, rep_s]))
        elif pop in ("tpop", "tpred"):
            xs = self.t2s[dt["aux"]]
            aux_s = np.sort(xs[xs >= 0])
        else:
            raise ValueError(pop)
        self.dep_s = {"act": act_s, "aux": aux_s, "rep": rep_s}
        self.rep_t = RA.Reporter(WT, dt["rep"], dt["act"])
        # S-side report: the same predictor and metric, scored on S's rows of T's report
        # genes. A DIAGNOSTIC of transfer only; no method or selection ever reads it.
        self.rep_s = RA.Reporter(WS, rep_s, act_s)
        self.own_s = RA.RewardSet(WS, aux_s)

    def to_t(self, chosen_s):
        return [self.s2t[int(g)] for g in chosen_s]

    def report_t(self, chosen_s):
        return self.rep_t(self.to_t(chosen_s))

    def check(self):
        WT, WS, d = self.WT, self.WS, self.dep_s
        names_ok = all(str(WT.cands[t]) == str(WS.cands[s]) for s, t in self.s2t.items())
        sets = [set(d["act"].tolist()), set(d["aux"].tolist()), set(d["rep"].tolist())]
        disjoint = not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        if self.pop == "all":      # every S gene is a candidate, a reward gene or excluded
            complete = len(sets[0] | sets[1] | sets[2]) == WS.K
        else:                      # reward genes are exactly S's rows of T's aux
            xs = self.t2s[self.dep_t["aux"]]
            complete = sets[1] == set(xs[xs >= 0].tolist())
        info = {"source": WS.name, "pop": self.pop, "n_shared_genes": len(self.s2t), "act_t": int(self.dep_t["act"].size),
                "act_covered": int(d["act"].size), "rep_t": int(self.dep_t["rep"].size),
                "rep_covered_excluded_from_S": int(d["rep"].size), "aux_s": int(d["aux"].size),
                "names_match": bool(names_ok), "disjoint": bool(disjoint), "partition_complete": bool(complete)}
        if not (names_ok and disjoint and complete):
            raise RuntimeError(f"bridge check failed: {info}")
        return info

    def diag(self, chosen_s):
        """S-side numbers for a set of S genes: its own reward on aux_S and the
        S-side report score. Both diagnostics; the reported number is T-side."""
        return {"s_own_mrr": float(self.own_s.mrr(np.asarray(chosen_s, np.int64))),
                "s_rep_mrr": float(self.rep_s(chosen_s)["mrr"])}


def facility_on(M, A, T, B):
    """Greedy facility location on similarity M over targets T, choosing from A.
    rl_acquire.facility with the matrix passed in (checked against go_facility)."""
    A, T = np.asarray(A), np.asarray(T)
    S = M[np.ix_(T, A)].astype(np.float32).copy()
    S[T[:, None] == A[None, :]] = 0.0
    s = np.zeros(T.size, np.float32)
    tm = np.ones(T.size, np.float32)
    avail = np.ones(A.size, bool)
    chosen = []
    for _ in range(B):
        av = np.flatnonzero(avail)
        g = (tm[:, None] * np.maximum(S[:, av] - s[:, None], 0)).sum(0)
        k = int(av[int(np.argmax(g))])
        avail[k] = False
        chosen.append(int(A[k]))
        tm[T == A[k]] = 0.0
        s = np.maximum(s, S[:, k])
    return chosen


def run_baselines(br, old_rows, out):
    WT, WS, dt, ds_ = br.WT, br.WS, br.dep_t, br.dep_s
    B = RA.B_DEPLOY
    base = {"dataset": WT.name, "source": WS.name, "split": br.sp, "budget": B,
            "act": int(dt["act"].size), "act_covered": int(ds_["act"].size), "K_report": int(dt["rep"].size)}
    rows = []

    def put(strategy, chosen_t=None, chosen_s=None, extra=None, t0=None):
        r = dict(base, strategy=strategy)
        if chosen_t is not None:
            r.update({f"rep_{k}": v for k, v in br.rep_t(chosen_t).items()})
            r["chosen"] = [int(x) for x in chosen_t]
        if chosen_s is not None:
            r["chosen_s"] = [int(x) for x in chosen_s]
            r.update(br.diag(chosen_s))
        r.update(extra or {})
        if t0 is not None:
            r["secs"] = round(time.time() - t0, 1)
        rows.append(r)
        with open(out + ".tmp", "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        os.replace(out + ".tmp", out)
        log(f"  [{WT.name} s{br.sp}] {strategy:22s} rep mrr={r.get('rep_mrr', float('nan')):.4f} "
            f"top1={r.get('rep_top1', float('nan')):.4f}  S rep={r.get('s_rep_mrr', float('nan')):.4f}")

    put("bridge", extra=br.check())
    old = {}
    for x in old_rows:
        old.setdefault(x["strategy"], x)
    sfx = "" if br.pop == "all" else "_" + br.pop
    fails = []
    if br.pop == "all":
        fails += t_side_rows(br, old, put)
    chk = RA.scorer_check(WS, ds_, br.rep_s)
    # The fast scorer takes GO Jaccards from the full K x K matrix, Reporter recomputes
    # them on the sub-matrix; float32 last bits differ, and among tied Jaccards that moves
    # which neighbour argpartition keeps. Measured 2026-11-23 inside K562 (the RPE1
    # target's source): up to 9e-4 MRR on 30 random subsets, vs 1e-6 in rl_acquire's own
    # check. It moves only the TRAINING reward; every reported number goes through
    # Reporter and is re-scored by rl_sim_report. A wrong scorer is off by >> 1e-2.
    ok_s = chk["max_abs_diff_mrr"] < 1e-2
    put("scorer_check_s" + sfx, extra=dict(chk, ok=bool(ok_s)))
    fails += [] if ok_s else ["S-side scorer"]
    s_side_rows(br, put, sfx)
    put("done", extra={"reconciliation_failures": fails})
    if fails:
        raise RuntimeError(f"reconciliation failed: {fails}")
    return rows


def t_side_rows(br, old, put):
    """Rows that do not depend on the simulator: the reconciliation points and the
    cost of offering only covered candidates. Written once, in the `all` file."""
    WT, dt, B = br.WT, br.dep_t, RA.B_DEPLOY
    fails = []
    t0 = time.time()
    rnd = RA.random_dist(br.rep_t, dt["act"], B)
    ok_r = "random" in old and abs(rnd["rep_mrr"] - old["random"]["rep_mrr"]) < 1e-12
    put("random", extra=dict(rnd, reconciled=bool(ok_r), old=old.get("random", {}).get("rep_mrr")), t0=t0)
    fails += [] if ok_r else ["random median"]
    t0 = time.time()
    ch_fac = RA.go_facility(WT, dt, B)
    ok_f = "go_facility" in old and [int(x) for x in ch_fac] == old["go_facility"]["chosen"]
    put("go_facility", ch_fac, extra={"reconciled": bool(ok_f)}, t0=t0)
    fails += [] if ok_f else ["go_facility set"]
    T_all = np.concatenate([dt["act"], dt["aux"]])
    ok_h = facility_on(WT.JJ, dt["act"], T_all, B) == [int(x) for x in ch_fac]
    put("facility_helper_check", extra={"matches_go_facility": bool(ok_h)})
    fails += [] if ok_h else ["facility helper"]
    # ---- the cost of offering only the candidates S measured
    t0 = time.time()
    put("random_cov", extra=RA.random_dist(br.rep_t, br.act_t_cov, B), t0=t0)
    t0 = time.time()
    put("go_facility_cov", facility_on(WT.JJ, br.act_t_cov, T_all, B), t0=t0)
    return fails


def s_side_rows(br, put, sfx):
    """Baselines that read the simulator: static uses of S, the transfer probe, and
    greedy on the very S reward the RL methods optimise."""
    WS, ds_, B = br.WS, br.dep_s, RA.B_DEPLOY
    # ---- static use of the S signatures: pick representative S signatures
    t0 = time.time()
    T_s = np.concatenate([ds_["act"], ds_["aux"]])
    ch = facility_on(WS.CC, ds_["act"], T_s, B)
    put("sim_sig_facility" + sfx, br.to_t(ch), ch, t0=t0)
    # ... and on S's GO view of the same genes, so "S's GO" is not confused with "S's signatures"
    t0 = time.time()
    ch = facility_on(WS.JJ, ds_["act"], T_s, B)
    put("sim_go_facility" + sfx, br.to_t(ch), ch, t0=t0)

    # ---- the mechanism diagnostic: does the simulator rank sets the way T does?
    t0 = time.time()
    prng = np.random.default_rng(9400)
    tr, sr, so = [], [], []
    for _ in range(N_PROBE):
        c = ds_["act"][prng.choice(ds_["act"].size, size=B, replace=False)]
        tr.append(br.report_t(c)["mrr"])
        d = br.diag(c)
        sr.append(d["s_rep_mrr"])
        so.append(d["s_own_mrr"])

    def spear(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    put("transfer_probe" + sfx, extra={"n": N_PROBE, "spearman_s_own_vs_t_rep": spear(so, tr),
                                 "spearman_s_rep_vs_t_rep": spear(sr, tr),
                                 "spearman_s_own_vs_s_rep": spear(so, sr),
                                 "t_rep": tr, "s_rep": sr, "s_own": so}, t0=t0)

    # ---- greedy on the IDENTICAL S reward the RL methods optimise (a run is one draw)
    for k in range(RA.N_GREEDY_ALL_SEEDS):
        t0 = time.time()
        ch = RA.greedy(WS, br.own_s, ds_["act"], B, np.random.default_rng(9201 + 1000 * k))
        put("sim_greedy_aux_all" + sfx, br.to_t(ch), ch,
            extra={"seed": k, "n_reward_genes": int(ds_["aux"].size)}, t0=t0)


def grid():
    ds = RA.env_list("PJ_RLA_DATASETS", RA.DATASETS)
    sp = RA.env_list("PJ_RLA_SPLITS", (4, 104, 204), int)
    me = RA.env_list("PJ_SIM_METHODS", METHODS)
    se = RA.env_list("PJ_RLA_SEEDS", (0, 1, 2), int)
    po = RA.env_list("PJ_SIM_POPS", POPS)
    return ([(d, s, m, z) for d in ds for s in sp for m in me for z in se],
            [(d, s, p) for p in po for d in ds for s in sp])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("baselines", "train", "grid"))
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()
    train_grid, base_grid = grid()
    if a.mode == "grid":
        print(json.dumps({"n_train": len(train_grid), "n_baselines": len(base_grid),
                          "train": train_grid, "baselines": base_grid}))
        return 0
    od = os.path.join(a.out_dir, "rl_acquire", "sim_smoke" if RA.SMOKE else "sim")
    for sub in ("runs", "baselines"):
        os.makedirs(os.path.join(od, sub), exist_ok=True)
    force = bool(os.environ.get("PJ_SIM_FORCE"))
    check = json.load(open(os.path.join(a.cache_dir, "cache_check.json")))

    def world(name):
        info = check["datasets"].get(name)
        if not info or not info.get("ok"):
            print(f"GATE NOT MET: cache for {name} missing or did not reproduce GAPS's random median: {info}")
            return None
        return RA.World(info["path"])

    if a.mode == "baselines":
        ds, sp, pop = base_grid[a.task_index]
        out = os.path.join(od, "baselines", f"{ds}__s{sp}" + ("" if pop == "all" else f"__{pop}") + ".jsonl")
        if os.path.exists(out) and not force:
            if any(json.loads(x)["strategy"] == "done" for x in open(out)):
                log(f"sim baselines already complete, skipping: {out}")
                return 0
        old_p = os.path.join(a.cache_dir, "baselines", f"{ds}__s{sp}.jsonl")
        if not os.path.exists(old_p):
            print(f"GATE NOT MET: {old_p} missing -- the reconciliation point does not exist")
            return 0
        old_rows = [json.loads(x) for x in open(old_p) if x.strip()]
        WT, WS = world(ds), world(RA.OTHER[ds])
        if WT is None or WS is None:
            return 0
        log(f"sim baselines: {ds} (source {WS.name}) split {sp} pop {pop} -> {out}")
        run_baselines(Bridge(WT, WS, sp, pop), old_rows, out)
        return 0

    ds, sp, me, seed = train_grid[a.task_index]
    tag = f"{me}__{ds}__s{sp}__seed{seed}"
    out = os.path.join(od, "runs", tag + ".json")
    curve = os.path.join(od, "runs", tag + ".curve.jsonl")
    if os.path.exists(out) and not force:
        log(f"already done, skipping (PJ_SIM_FORCE=1 to redo): {out}")
        return 0
    WT, WS = world(ds), world(RA.OTHER[ds])
    if WT is None or WS is None:
        return 0
    bm, pop = split_pop(me)
    sfx = "" if pop == "all" else "_" + pop
    br = Bridge(WT, WS, sp, pop)
    WS = br.WS                # the simulator: S itself, or S's signatures under T's GO (tpred)
    info = br.check()
    log(f"train {tag}: {info} cfg={RA.CFG}")
    base = {"dataset": ds, "source": WS.name, "split": sp, "seed": seed, "budget": RA.B_DEPLOY,
            "K_report": int(br.dep_t["rep"].size), "act_covered": info["act_covered"], "cfg": RA.CFG,
            "smoke": RA.SMOKE, "regime": "xline_sim", "pop": pop}
    t0 = time.time()
    try:
        if bm in ("sim_pg_tab", "sim_pg_tab_boot"):
            r = RA.train_pg_tab(WS, br.dep_s, seed, bm == "sim_pg_tab_boot", curve, rep_fn=br.report_t)
            outs = [(me, r)]
        elif bm == "sim_ppo_peek":
            r = RA.train_ppo(WS, br.dep_s, "peek", seed, a.threads, curve, rep_fn=br.report_t)
            outs = [(me, r)]
        elif bm == "sim_critic":
            r = RA.train_critic(WS, br.dep_s, seed, a.threads, curve, rep_fn=br.report_t)
            outs = [("sim_critic_only" + sfx, dict(r["critic_only"], diag=r["diag"])),
                    ("sim_critic_shrink" + sfx, dict(r["critic_shrink"], diag=r["diag"]))]
        else:
            raise ValueError(me)
        res = []
        for name, r in outs:
            ch_s = [int(c) for c in r.pop("chosen")]
            rep = r.pop("rep")
            row = dict(base, method=name, **{f"rep_{k}": v for k, v in rep.items()}, **r)
            row["chosen_s"] = ch_s
            row["chosen"] = br.to_t(ch_s)
            row.update(br.diag(ch_s))
            res.append(row)
    except Exception as e:
        traceback.print_exc()
        res = [dict(base, method=me, error=f"{type(e).__name__}: {e}")]
    for x in res:
        x["secs"] = round(time.time() - t0, 1)
        log(f"RESULT {x['method']:18s} T rep mrr={x.get('rep_mrr', float('nan')):.4f} "
            f"top1={x.get('rep_top1', float('nan')):.4f}  S rep={x.get('s_rep_mrr', float('nan')):.4f}")
    if any("error" in x for x in res):
        with open(os.path.join(od, "runs", tag + ".error.json"), "w") as f:
            json.dump(res, f)
        return 1
    with open(out + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(out + ".tmp", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
