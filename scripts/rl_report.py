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

BASE_ORDER = ["go_coverage", "go_facility", "greedy_rew400_c120", "greedy_aux_all"]
RL_ORDER = ["pg_tab", "pg_tab_boot", "critic_only", "critic_shrink", "ppo_peek", "ppo_blind"]
CHEATS = ["cheat_rep_c120", "cheat_rep_all"]
LABEL = {
    "go_coverage": "GO coverage (static, blind)",
    "go_facility": "GO facility location (static, blind)",
    "greedy_rew400_c120": "greedy, GAPS protocol (peek)",
    "greedy_aux_all": "greedy, all aux genes (peek)",
    "pg_tab": "REINFORCE, per-gene logits",
    "pg_tab_boot": "REINFORCE, resampled reward",
    "critic_only": "learned critic, greedy",
    "critic_shrink": "critic-shrunk greedy",
    "ppo_peek": "PPO feature policy (peek)",
    "ppo_blind": "PPO feature policy (blind)",
    "cheat_rep_c120": "cheat, GAPS protocol",
    "cheat_rep_all": "cheat, all candidates",
}
BLIND = {"go_coverage", "go_facility", "ppo_blind"}
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
            reps[(ds, sp)] = (RA.Reporter(worlds[ds], dep["rep"], dep["act"]), dep)
        return reps[(ds, sp)]

    def rescore(row):
        fn, _ = reporter(row["dataset"], row["split"])
        r = fn(row["chosen"])
        row["mrr_train_env"], row["top1_train_env"] = row["rep_mrr"], row["rep_top1"]
        row["rep_mrr"], row["rep_top1"] = r["mrr"], r["top1"]

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
        return {"med": rnd["rep_mrr"], "p95": rnd["rep_mrr_p95"], "p05": rnd["rep_mrr_p05"],
                "max": rnd["rep_mrr_max"], "cheat": max(cheats) if cheats else np.nan,
                "greedy": mean("greedy_aux_all"), "facility": mean("go_facility")}

    # one record per (method, dataset, split, seed), everything relative to its split
    recs = []
    for (ds, sp), b in base.items():
        R = ref(ds, sp)
        if R is None:
            continue
        for s in BASE_ORDER + CHEATS:
            for x in b.get(s, []):
                recs.append({"method": s, "dataset": ds, "split": sp, "seed": x.get("seed"),
                             "mrr": x["rep_mrr"], "top1": x["rep_top1"],
                             **{f"ref_{k}": v for k, v in R.items()}})
    for r in runs:
        if "rep_mrr" not in r:
            continue
        R = ref(r["dataset"], r["split"])
        if R is None:
            continue
        recs.append({"method": r["method"], "dataset": r["dataset"], "split": r["split"], "seed": r["seed"],
                     "mrr": r["rep_mrr"], "top1": r["rep_top1"], "diag": r.get("diag"),
                     "best_iter": r.get("best_iter"), **{f"ref_{k}": v for k, v in R.items()}})
    for x in recs:
        x["ratio"] = x["mrr"] / x["ref_med"]
        x["above_p95"] = bool(x["mrr"] > x["ref_p95"])
        x["above_max"] = bool(x["mrr"] > x["ref_max"])
        x["cheat_frac"] = (x["mrr"] - x["ref_med"]) / (x["ref_cheat"] - x["ref_med"])
        x["vs_greedy"] = x["mrr"] - x["ref_greedy"]
        x["vs_facility"] = x["mrr"] - x["ref_facility"]

    datasets = [d for d in SHORT if any(x["dataset"] == d for x in recs)]
    methods = [m for m in BASE_ORDER + RL_ORDER + CHEATS if any(x["method"] == m for x in recs)]
    table = []
    for ds in datasets:
        for m in methods:
            xs = [x for x in recs if x["dataset"] == ds and x["method"] == m]
            if not xs:
                continue
            f = lambda k: np.array([x[k] for x in xs], float)  # noqa: E731
            # paired comparison against greedy and facility on the SAME split
            table.append({
                "dataset": ds, "method": m, "n": len(xs), "splits": sorted({x["split"] for x in xs}),
                "mrr_mean": float(f("mrr").mean()), "mrr_sd": float(f("mrr").std()),
                "top1_mean": float(f("top1").mean()),
                "ratio_mean": float(f("ratio").mean()), "ratio_min": float(f("ratio").min()),
                "ratio_max": float(f("ratio").max()),
                "frac_above_p95": float(f("above_p95").mean()), "frac_above_max": float(f("above_max").mean()),
                "cheat_frac_mean": float(f("cheat_frac").mean()),
                "vs_greedy_mean": float(f("vs_greedy").mean()),
                "frac_beats_greedy": float((f("vs_greedy") > 0).mean()),
                "vs_facility_mean": float(f("vs_facility").mean()),
                "frac_beats_facility": float((f("vs_facility") > 0).mean())})
    crit = [x["diag"] for x in recs if x.get("diag") and x["method"] == "critic_only"]
    diag = {}
    if crit:
        for k in ("corr_emp_vs_heldout_gain", "corr_critic_vs_heldout_gain", "corr_shrink_vs_heldout_gain",
                  "shrink_coef_critic", "shrink_coef_emp"):
            v = np.array([d[k] for d in crit], float)
            diag[k] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}
    chk = {f"{ds}__s{sp}": b["scorer_check"][0] for (ds, sp), b in base.items() if "scorer_check" in b}
    summary = {"table": table, "critic_diag": diag, "scorer_check": chk, "errors": errors,
               "random_gaps_env": {f"{d}__s{s}": v for (d, s), v in rand.items()},
               "train_env_vs_gaps_env_abs_mrr_diff": {"mean": float(np.mean(env_diff)) if env_diff else None,
                                                      "max": float(max(env_diff)) if env_diff else None,
                                                      "n": len(env_diff)},
               "n_runs": len(runs), "baseline_splits": sorted(f"{d}__s{s}" for d, s in base),
               "smoke": a.smoke}
    out_dir = os.path.dirname(runs_dir)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------ report.md
    L = ["# RL on the acquisition gap: results", "",
         "Report-set MRR, budget 100 of 437, pooled over splits x seeds. `x rand` is the multiple "
         "of that split's random-x100 median; `>p95` the share of runs beating that split's random "
         "p95; `cheat %` the share of (cheat - random median) captured, cheat = greedy on the report "
         "genes with every candidate scored; `vs greedy` the paired difference from greedy on all "
         "aux genes on the same split.", ""]
    for ds in datasets:
        L += [f"## {SHORT[ds]}", "",
              "| method | n | rep MRR | top-1 | x rand | >p95 | cheat % | vs greedy (wins) | vs facility (wins) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for t in [t for t in table if t["dataset"] == ds]:
            L.append(f"| {LABEL.get(t['method'], t['method'])} | {t['n']} | {t['mrr_mean']:.4f} ± {t['mrr_sd']:.4f} | "
                     f"{t['top1_mean']:.4f} | {t['ratio_mean']:.2f}× | {t['frac_above_p95']:.0%} | "
                     f"{t['cheat_frac_mean']:.0%} | {t['vs_greedy_mean']:+.4f} ({t['frac_beats_greedy']:.0%}) | "
                     f"{t['vs_facility_mean']:+.4f} ({t['frac_beats_facility']:.0%}) |")
        L.append("")
    if diag:
        L += ["## Is the reward noisy? (critic diagnostics, held-out sub-problems)", "",
              "| correlation with the held-out one-step gain | mean | min | max |", "|---|---:|---:|---:|"]
        for k, nm in (("corr_emp_vs_heldout_gain", "greedy's empirical gain (visible genes)"),
                      ("corr_critic_vs_heldout_gain", "learned critic"),
                      ("corr_shrink_vs_heldout_gain", "shrinkage of the two")):
            L.append(f"| {nm} | {diag[k]['mean']:.3f} | {diag[k]['min']:.3f} | {diag[k]['max']:.3f} |")
        L.append("")
    L += ["## Checks", "", "Fast scorer vs rl_headroom's own path (max |diff| over 30 random subsets):", ""]
    for k, v in sorted(chk.items()):
        L.append(f"- {k}: MRR {v['max_abs_diff_mrr']:.2e}, top-1 {v['max_abs_diff_top1']:.2e}")
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
            ax.text(ch, -0.9, f" cheat {ch:.2f}×", color=INK2, fontsize=8, va="center")
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
                col = C_BLIND if me == "ppo_blind" else C_PEEK
                axes[0][j].plot(it, [x["val_mrr"] / x["facility"] for x in c], color=col, lw=1.2, alpha=0.6)
                axes[1][j].plot(it, [x["rep_mrr"] / R["med"] for x in c], color=col, lw=1.2, alpha=0.6)
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
        fig.legend(handles=[Line2D([], [], color=C_PEEK, label="PPO peek"),
                            Line2D([], [], color=C_BLIND, label="PPO blind")],
                   loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        p = os.path.join(a.fig_dir, "fig16_rl_curves.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
