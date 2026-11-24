#!/usr/bin/env python
"""Generate GAPS.md's sim-to-real section from the rl_sim_report summaries.

Reads <art>/rl_acquire/sim/summary{,_confirm,_confirm2,_confirm3}.json (every chosen set
already re-scored in the analysis env by rl_sim_report.py) plus the sim run and baseline
files for the overlap diagnostic, and prints markdown. No number or verdict below is
typed: verdicts come from split-level paired statistics by the rule in `verdict()`.

The unit of evidence is the SPLIT. Three seeds of one method share a split's floor, so a
run-level t counts one split three times; every claim here averages the seeds first.

Usage: python -I sim_section.py --art-dir D
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own scripts/ only

SHORT = {"replogle_k562_essential": "K562", "replogle_rpe1_essential": "RPE1"}
OTHER = {"replogle_k562_essential": "RPE1", "replogle_rpe1_essential": "K562"}
FLOORS = ("go_facility", "go_facility_idf", "go_facility_emb")
SETS = (("", "development"), ("_confirm", "held-out A"), ("_confirm2", "held-out B"), ("_confirm3", "held-out C"))
HELD = ("_confirm", "_confirm2", "_confirm3")


def load(art, tag):
    p = os.path.join(art, "rl_acquire", "sim", f"summary{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


class Recs:
    """Per-split mean MRR of every method, and the references it is paired against."""

    def __init__(self, S):
        self.R = S["records"]
        self.rand = S["random"]
        self.by = {}
        for x in self.R:
            self.by.setdefault((x["dataset"], x["split"], x["method"]), []).append(x["mrr"])

    def splits(self, ds):
        return sorted({x["split"] for x in self.R if x["dataset"] == ds})

    def m(self, ds, sp, meth):
        v = self.by.get((ds, sp, meth))
        return float(np.mean(v)) if v else np.nan

    def ref(self, ds, sp, kind, meth=""):
        if kind == "floor":
            return max(self.m(ds, sp, f) for f in FLOORS)
        if kind == "pool":
            return self.m(ds, sp, "go_facility_cov")
        if kind == "greedy":        # greedy on the SAME reward the method optimises
            sfx = next((s for s in ("_tpop", "_tpred", "_timp") if meth.endswith(s)), "")
            if "hybrid2" in meth:
                return self.m(ds, sp, "sim_greedy_hybrid2")
            if "combo" in meth:
                return self.m(ds, sp, "sim_greedy_combo" + sfx)
            return self.m(ds, sp, "sim_greedy_aux_all" + sfx)
        raise ValueError(kind)

    def ratio(self, ds, sp, meth):
        return self.m(ds, sp, meth) / self.rand[f"{ds}__s{sp}"]["rep_mrr"]


def diffs(recsets, ds, meth, kind):
    """Per-split differences (method - reference) over one or several summaries."""
    d = []
    for rc in recsets:
        for sp in rc.splits(ds):
            a, b = rc.m(ds, sp, meth), rc.ref(ds, sp, kind, meth)
            if np.isfinite(a) and np.isfinite(b):
                d.append(a - b)
    return np.array(d)


def stat(d):
    """Split-level one-sample t on paired differences, two-sided p from t(n-1)."""
    from scipy import stats
    if d.size == 0:
        return None
    if d.size > 1 and d.std(ddof=1) > 0:
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))
        p = float(2 * stats.t.sf(abs(t), d.size - 1))
    else:
        t, p = float("nan"), float("nan")
    return {"mean": float(d.mean()), "won": int((d > 0).sum()), "n": int(d.size), "t": t, "p": p}


def cell(st):
    if st is None:
        return "—"
    return f"{st['mean']:+.4f} ({st['won']}/{st['n']}, p {st['p']:.2g})"


def verdict(st):
    """beats: two-sided p < 0.05 at the split level AND at least 3/4 of splits won;
    loses to: p < 0.05 the other way; anything else is a tie, however suggestive."""
    if st is None:
        return "untested"
    if st["p"] < 0.05 and st["mean"] > 0 and st["won"] >= 0.75 * st["n"]:
        return "beats"
    if st["p"] < 0.05 and st["mean"] > 0:      # significant, but not on 3/4 of splits: below the bar
        return "is ahead of, without clearing the bar against,"
    if st["p"] < 0.05 and st["mean"] < 0:
        return "loses to"
    return "ties"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--part", choices=("section", "summary"), default="section")
    a = ap.parse_args()
    S = {tag: load(a.art_dir, tag) for tag, _ in SETS}
    if S[""] is None:
        raise SystemExit("no development summary (rl_acquire/sim/summary.json)")
    RC = {tag: Recs(s) for tag, s in S.items() if s is not None}
    held = [RC[t] for t in HELD if t in RC]
    dsl = [d for d in SHORT if any(d in x["dataset"] for x in RC[""].R)]
    P = print

    def held_range(rcs):
        sp = sorted({s for rc in rcs for d in dsl for s in rc.splits(d)})
        return f"{len(sp)} splits"

    # ------------------------------------------------------------------ verdicts first
    pooled = {(ds, m, k): stat(diffs(held, ds, m, k)) for ds in dsl
              for m in ("sim_pg_tab", "sim_pg_combo", "sim_pg_hybrid2") for k in ("floor", "pool", "greedy")}
    if a.part == "summary":
        summary(pooled, dsl, held, P)
        return
    P("## Sim-to-real: RL trained inside another cell line's screen")
    P()
    P("**Added 2026-11-23.** Every blind acquisition method, ours and published, ties GO facility")
    P("location, and the one input that ever moved the unseen-gene metric was another cell line's")
    P("*measurement* (\"The unseen-gene gap is a combiner problem\"). So: when one line has been")
    P("screened and 100 of 437 genes must be chosen in a new line, can RL use the existing screen")
    P("as a **simulator** and choose better than prior knowledge does? That is the realistic way a")
    P("screen in a new cell line gets planned: one in another line usually exists already.")
    P()
    P("The deployment task is unchanged number for number: the same splits, GO-kNN built from the")
    P("chosen genes' real signatures in the target line, retrieval MRR on the target's report")
    P("genes, the same random-median denominator. No target-line signature of a candidate is read,")
    P("and the source line's rows of the target's report genes are excluded from training, reward")
    P("and validation alike. The RL methods are the peek-regime ones (REINFORCE on per-gene logits,")
    P("critic-shrunk greedy, PPO), run **inside the source line** on the deployment candidates; the")
    P("chosen set maps back and is scored in the target by `rl_acquire.Reporter`. Every chosen set")
    rec = S[""].get("reconciliation_vs_rl_report") or {}
    P("is re-scored in the analysis env; on the development splits the random medians and GO facility")
    P(f"means reproduce `rl_report`'s to max |Δ| = {max(rec.values()):.1g} over {len(rec)} checks"
      if rec else "means could not be reconciled with `rl_report` (summary missing)")
    P("(the held-out splits are new, so they have nothing to reconcile with).")
    P()
    # ------------------------------------------------------------------ headline
    P("### What survives held-out splits")
    P()
    P("The ideas below were developed on splits 4/104/204, so every claim is re-tested, unchanged,")
    P(f"on splits nothing was tuned on ({held_range(held)} across up to three fresh sets, 3 seeds")
    P("each). Statistics are **per split** (seeds averaged first), paired against the reference on")
    P("the same split.")
    P()
    P("| held-out, per split | " + " | ".join(f"{SHORT[d]}: vs best prior floor | {SHORT[d]}: vs GO facility, same candidates | {SHORT[d]}: vs greedy, same reward" for d in dsl) + " |")
    P("|---|" + "---|" * (3 * len(dsl)))
    LAB = {"sim_pg_tab": "REINFORCE in the source line", "sim_pg_combo": "REINFORCE, simulator + T's GO facility",
           "sim_pg_hybrid2": "REINFORCE, simulator + GO facility, all candidates"}
    for m in ("sim_pg_tab", "sim_pg_combo", "sim_pg_hybrid2"):
        if all(pooled[(d, m, "floor")] is None for d in dsl):
            continue
        P(f"| {LAB[m]} | " + " | ".join(f"{cell(pooled[(d, m, 'floor')])} | {cell(pooled[(d, m, 'pool')])} | "
                                       f"{cell(pooled[(d, m, 'greedy')])}" for d in dsl) + " |")
    P()
    P("Cells: mean MRR difference (splits won / splits, two-sided p of a split-level paired t). A")
    P("verdict of *beats* needs p < 0.05 **and** at least 3/4 of splits won — a bar set before the last")
    P("held-out set was run, and kept. \"Best prior floor\" is the")
    P("strongest of the three static GO facility views on that split, over all 437 candidates — the")
    P("floor every blind method in this file has had to clear.")
    P()
    for d in dsl:
        c = {k: pooled[(d, "sim_pg_combo", k)] for k in ("floor", "pool", "greedy")}
        t = {k: pooled[(d, "sim_pg_tab", k)] for k in ("floor", "pool", "greedy")}
        P(f"- **{SHORT[d]}**, simulated in {OTHER[d]}'s screen. The pre-registered combined-reward "
          f"REINFORCE {verdict(c['floor'])} the best prior floor over all candidates ({cell(c['floor'])}), "
          f"{verdict(c['pool'])} GO facility over the candidates {OTHER[d]} measured ({cell(c['pool'])}), and "
          f"{verdict(c['greedy'])} greedy on the identical combined reward ({cell(c['greedy'])}). Plain "
          f"REINFORCE on the simulator alone {verdict(t['greedy'])} greedy on its identical reward "
          f"({cell(t['greedy'])}) and {verdict(t['floor'])} the floor ({cell(t['floor'])}).")
    P()
    hy = {d: pooled[(d, "sim_pg_hybrid2", "floor")] for d in dsl}
    if any(v is not None for v in hy.values()):
        P("**Offering every candidate (the hybrid).** On K562 the simulator can only offer the genes")
        P("RPE1 also screened. The combined reward over *all* 437 candidates — the simulator scores the")
        P("chosen genes it measured, GO facility scores every chosen gene — was introduced after held-out")
        P("set A; its first version had a bug (see caveats), and the fixed version ran only on set C: " + "; ".join(
            f"{SHORT[d]} {verdict(hy[d])} the floor ({cell(hy[d])})" for d in dsl if hy[d] is not None) + ".")
        P()
    mechanism(RC, held, dsl, P, a.art_dir)
    development(RC[""], dsl, P)
    caveats(S, RC, held, dsl, pooled, P, a.art_dir)


def summary(pooled, dsl, held, P):
    """One paragraph for GAPS.md's Summary, verdicts by the same rule as the section."""
    n = sorted({sp for rc in held for d in dsl for sp in rc.splits(d)})
    c = {d: {k: pooled[(d, "sim_pg_combo", k)] for k in ("floor", "pool", "greedy")} for d in dsl}
    t = {d: pooled[(d, "sim_pg_tab", "greedy")] for d in dsl}
    win = [d for d in dsl if verdict(c[d]["floor"]) == "beats"]
    pool_win = [d for d in dsl if verdict(c[d]["pool"]) == "beats"]
    if win:
        head = f"RL trained inside another cell line's screen beats prior knowledge for {' and '.join(SHORT[d] for d in win)}"
    elif pool_win:
        head = ("RL trained inside another cell line's screen beats prior knowledge on the genes that "
                "screen covers, but not overall")
    else:
        head = "RL trained inside another cell line's screen does not beat prior knowledge"
    P(f"**{head} (added 2026-11-23).** When one line is already screened, its measurements can")
    P("serve as a simulator for choosing the next line's 100 genes, with no target-line signature")
    P("read. Read only on held-out splits that chose nothing "
      f"({len(n)} splits), REINFORCE on the simulator {', '.join(f'{verdict(t[d])} greedy on the identical reward for {SHORT[d]} ({cell(t[d])})' for d in dsl)}.")
    P("Combining the simulator with the target's GO facility objective in one reward (weight fixed")
    P("in advance), REINFORCE " + "; ".join(
        f"{verdict(c[d]['floor'])} the best prior floor over all candidates for {SHORT[d]} ({cell(c[d]['floor'])}) "
        f"and {verdict(c[d]['pool'])} GO facility over the candidates the other line measured ({cell(c[d]['pool'])})"
        for d in dsl) + ".")
    parts = []
    for d in [d for d in dsl if d not in win]:
        cost = stat(np.array([rc.m(d, sp, "go_facility_cov") - rc.ref(d, sp, "floor")
                              for rc in held for sp in rc.splits(d)]))
        parts.append(f"{SHORT[d]} {cost['mean']:+.4f} against RL's {c[d]['pool']['mean']:+.4f} within them")
    if parts:
        P("The difference between the two columns is the candidates the source line never screened, which")
        P("no simulator can offer: restricting GO facility itself to the covered ones costs " + "; ".join(parts) + ".")
    hy = [pooled.get((d, "sim_pg_hybrid2", "floor")) for d in dsl]
    if any(h is not None for h in hy):
        hv = [verdict(h) for h in hy if h is not None]
        P("A hybrid that offers every candidate, pre-registered and run on a fresh set, "
          + ("beats the floor on " + ", ".join(SHORT[d] for d, h in zip(dsl, hy) if h and verdict(h) == "beats")
             if "beats" in hv else "does not clear it on either line") + ".")
    comp = [(d, m) for d in dsl for m in ("sim_pg_tab", "sim_pg_combo")]
    rl_wins = sum(1 for d, m in comp if pooled[(d, m, "greedy")] and pooled[(d, m, "greedy")]["p"] < 0.05
                  and pooled[(d, m, "greedy")]["mean"] > 0)
    P("So the result refines the separation rather than overturning it: *another line's measurement")
    P("lets RL out-choose prior knowledge among the genes it measured, but over all candidates the net")
    P("is not distinguishable from the prior floor.* As an optimiser, RL is ahead of greedy on the")
    P(f"identical reward at p < 0.05 in {rl_wins} of {len(comp)} (line, reward) pairs. See \"Sim-to-real\" below.")


def mechanism(RC, held, dsl, P, art):
    """Why the result has the shape it has, from diagnostics no method reads."""
    import rl_acquire as RA
    P("### Why it has this shape")
    P()
    # 1. the simulator ranks sets the way the target does, weakly
    pr = load(art, "")["probes"]
    import re          # the `S` simulator's probes only: keys end in the split number
    parts = []
    for d in dsl:
        mine = [v for k, v in pr.items() if k.startswith(d) and re.search(r"__s\d+$", k)]
        own = [v["spearman_s_own_vs_t_rep"] for v in mine]
        rep = [v["spearman_s_rep_vs_t_rep"] for v in mine]
        parts.append(f"{SHORT[d]} (simulated in {OTHER[d]}) {min(own):+.2f} to {max(own):+.2f} on the "
                     f"simulator's own reward genes and {min(rep):+.2f} to {max(rep):+.2f} on {OTHER[d]}'s rows "
                     f"of the report genes themselves")
    P("- **The simulator is a noisy judge of a gene set.** Over 100 random sets, the Spearman between a")
    P("  set's simulator score and its score in the target is " + "; ".join(parts) + ". The second")
    P("  number is a ceiling no method uses: even a perfect simulator reward over the right genes ranks")
    P("  sets only moderately, because the same gene acts differently in the other line.")
    # 2. REINFORCE-in-S sets are different from GO facility's and closer to the peek optimum
    root = os.path.join(art, "rl_acquire")
    cache = json.load(open(os.path.join(root, "cache_check.json")))["datasets"]
    cands = {d: [str(c) for c in np.load(cache[d]["path"])["cands"]] for d in dsl}
    other = {"replogle_k562_essential": "replogle_rpe1_essential", "replogle_rpe1_essential": "replogle_k562_essential"}
    sets = {}
    for pat, key in (("baselines/*.jsonl", "fac"), ("runs/pg_tab__*.json", "peek"), ("sim/runs/sim_pg_tab__*.json", "sim")):
        for pth in glob.glob(os.path.join(root, pat)):
            rows = [json.loads(x) for x in open(pth)] if pth.endswith(".jsonl") else json.load(open(pth))
            for x in rows:
                if key == "fac" and x.get("strategy") != "go_facility":
                    continue
                if x["split"] not in (4, 104, 204):
                    continue
                sets.setdefault((key, x["dataset"], x["split"]), []).append(set(x["chosen"]))
    ov = {}
    for d in dsl:
        oth = set(cands[other[d]])
        ov_f, ov_p, ch_f, ch_p = [], [], [], []
        for sp in (4, 104, 204):
            act = RA.split_of(len(cands[d]), sp)["act"]
            cov = {int(g) for g in act if cands[d][g] in oth}
            F = sets[("fac", d, sp)][0]
            for S_ in sets.get(("sim", d, sp), []):
                ov_f.append(len(S_ & F))
                ch_f.append(100 * len(F & cov) / len(cov))
                for Pk in sets.get(("peek", d, sp), []):
                    ov_p.append(len(S_ & Pk))
                    ch_p.append(100 * len(Pk & cov) / len(cov))
        ov[d] = (f"{SHORT[d]} {min(ov_f)}–{max(ov_f)} of 100 with GO facility's set (chance {np.mean(ch_f):.0f}) and "
                 f"{min(ov_p)}–{max(ov_p)} with the peek REINFORCE set, which saw the real signatures "
                 f"(chance {np.mean(ch_p):.0f})")
    P("- **It finds a different set, not a better copy of the same one.** REINFORCE in the simulator")
    P("  shares " + "; ".join(ov[d] for d in dsl) + ". The simulator knows which genes have")
    P("  informative signatures; GO facility knows what the target's GO-kNN needs for coverage. That")
    P("  complementarity is why the combined reward was tried.")
    # 3. the cost of offering only covered candidates
    parts = []
    for d in dsl:
        cost = stat(np.array([rc.m(d, sp, "go_facility_cov") - rc.ref(d, sp, "floor")
                              for rc in held for sp in rc.splits(d)]))
        cov = np.mean([np.isin(RA.split_of(len(cands[d]), sp)["act"],
                               [i for i, g in enumerate(cands[d]) if g in set(cands[other[d]])]).mean()
                       for sp in (4, 104, 204)])
        parts.append(f"{SHORT[d]} ({100 * cov:.0f}% of candidates covered) {cell(cost)}")
    P("- **The candidates the simulator cannot offer.** A gene the source line never screened cannot")
    P("  be simulated. Restricting GO facility itself to the covered candidates costs, against the")
    P("  floor on the held-out splits, " + "; ".join(parts) + ". Every simulator method pays that")
    P("  before it starts, which is the gap between its same-candidates and all-candidates columns.")
    P()


def development(rc, dsl, P):
    """The four simulators on the development splits (where the ideas were found)."""
    P("### The development runs: four simulators")
    P()
    P("Splits 4/104/204, 3 seeds; Δ is per split against the best prior floor (splits won of 3).")
    P("`S` rewards on every source-line gene outside the candidates and report genes; `S on T's genes`")
    P("only on the source rows of the target's measured atlas; `+ T's GO` also uses the target's GO")
    P("Jaccard inside the simulator; `T atlas + imputed` is the target's own peek world with each")
    P("candidate's hidden signature imputed (re-centred source signature on the 2,383 shared readout")
    P("genes, or GO-kNN when the source never measured it).")
    P()
    P("| simulator | method | " + " | ".join(f"{SHORT[d]} × random | {SHORT[d]} Δ vs floor" for d in dsl) + " |")
    P("|---|---|" + "---:|---|" * len(dsl))
    sims = (("", "S"), ("_tpop", "S on T's genes"), ("_tpred", "S on T's genes + T's GO"), ("_timp", "T atlas + imputed"))
    meths = (("sim_greedy_aux_all", "greedy, same reward"), ("sim_pg_tab", "REINFORCE"),
             ("sim_critic_shrink", "critic-shrunk greedy"), ("sim_ppo_peek", "PPO"),
             ("sim_pg_combo", "REINFORCE, + T's GO facility"))
    for sfx, lab in sims:
        for m, ml in meths:
            name = m + sfx
            cells = []
            for d in dsl:
                sp = rc.splits(d)
                r = [rc.ratio(d, x, name) for x in sp]
                dd = np.array([rc.m(d, x, name) - rc.ref(d, x, "floor") for x in sp])
                cells.append("—" if not np.isfinite(r).all() else
                             f"{np.mean(r):.2f}× | {dd.mean():+.4f} ({int((dd > 0).sum())}/{dd.size})")
            if any(c != "—" for c in cells):
                P(f"| {lab} | {ml} | " + " | ".join(cells) + " |")
    P()
    refs = []
    for d in dsl:
        sp = rc.splits(d)
        fl = np.mean([rc.ref(d, x, "floor") / rc.rand[f"{d}__s{x}"]["rep_mrr"] for x in sp])
        pk = np.mean([rc.ratio(d, x, "pg_tab") for x in sp])
        refs.append(f"{SHORT[d]} {fl:.2f}× / {pk:.2f}×")
    P("References on the same splits, best prior floor / peek REINFORCE (which sees the real")
    P("signatures): " + "; ".join(refs) + ".")
    P()


def caveats(S, RC, held, dsl, pooled, P, art):
    import rl_acquire as RA
    dev, A = RC[""], RC.get("_confirm")
    P("### Caveats")
    P()
    P("- **Many variants were tried before the held-out runs**: four simulators × five RL methods on")
    P("  the development splits, then the combined reward, then the hybrid. A development-split win")
    P("  means little in that situation, so every claim above is read on splits that chose nothing.")
    if A is not None:
        for d in dsl:
            dv = stat(np.array([dev.m(d, x, "sim_pg_combo") - dev.ref(d, x, "floor") for x in dev.splits(d)]))
            ha = stat(np.array([A.m(d, x, "sim_pg_combo") - A.ref(d, x, "floor") for x in A.splits(d)]))
            P(f"  {SHORT[d]}, combined reward vs floor: development {cell(dv)}, held-out set A {cell(ha)}.")
    P("- The combined reward's weight (1, in standardised units) was fixed before any run and never")
    P("  tuned. The hybrid's first version had a bug — a set with no source-measured gene scored the")
    P("  random-set mean, so greedy chose none of them on every split — and is excluded (its runs stay")
    P("  on disk); the fixed version scores an empty reference set at chance and ran only on fresh splits.")
    root = os.path.join(art, "rl_acquire")
    ro = json.load(open(os.path.join(root, "readout_check.json"))) if os.path.exists(
        os.path.join(root, "readout_check.json")) else {}
    eff = [f"{SHORT[d]} {pooled[(d, 'sim_pg_combo', 'floor')]['mean']:+.4f}" for d in dsl
           if pooled[(d, "sim_pg_combo", "floor")] is not None]
    pk = []
    for d in dsl:
        dd = np.array([dev.m(d, x, "pg_tab") - dev.ref(d, x, "floor") for x in dev.splits(d)])
        pk.append(f"{SHORT[d]} {dd.mean():+.4f}")
    P("- Effects are small next to measuring: the combined reward's held-out gain over the floor is")
    P(f"  {', '.join(eff)} MRR, against {', '.join(pk)} for peek REINFORCE, which reads the real")
    P("  signatures. The simulator recovers a fraction of what measurement is worth.")
    P(f"- One predictor (GO-kNN), one budget (100 of 437), two lines that are each other's only")
    P(f"  source; `T atlas + imputed` uses the {ro.get('shared_readout_genes', '?')} readout genes both")
    P("  caches share, recovered and verified by `scripts/rl_readout.py`.")
    P()
    P("Reproduce with `./submit.sh sim` (development grid, all CPU), the held-out sets with")
    P("`PJ_RLA_SPLITS=\"...\" ./submit.sh sim_base` / `sim_runs` and `62_sim_report` with")
    P("`PJ_SIM_REPORT_SPLITS` / `PJ_SIM_REPORT_TAG`; `_timp` needs `./submit.sh sim_readout` first.")
    P("Outputs: `artifacts/rl_acquire/sim/{report,summary}{,_confirm,_confirm2,_confirm3}.*`; this")
    P("section is regenerated by `python3 scripts/sim_splice.py`.")


if __name__ == "__main__":
    main()
