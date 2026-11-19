"""Redraw fig13/fig15/fig16/fig17 in the journal style used by fig18-fig21.

The originals (scripts/generalization_figures.py, scripts/rl_report.py) draw these
four in a casual sans-serif screen style with on-figure titles; the case studies
(scripts/story_draw.py) use a paper style -- serif STIX, bold panel letters, no
on-figure explanatory titles (the message lives in the caption), a red/blue accent
over greyscale, vector PDF beside the PNG. For a write-up every figure must look the
same, so this script re-renders the four screen figures in that same style, reading
only saved artifacts:

  fig13  <art>/generalization/summary.json          (headroom by axis)
  fig15  <art>/rl_acquire/records.jsonl             (RL vs baselines, per split x seed)
  fig16  <art>/rl_acquire/runs/ppo_*.curve.jsonl    (PPO learning curves)
  fig17  <art>/rl_acquire/records.jsonl             (published methods, same task)

Nothing is recomputed; it redraws in seconds. The style helpers (_style, _panel,
_save, _gi, palette) are imported from story_draw so the look is literally identical.

Usage: python -I paper_figs.py --art-dir D --fig-dir D
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own scripts/ only
from story_draw import _style, _panel, _save, _gi, RED, BLUE, INK, DARK, MID, LIGHT, FAINT, W2  # noqa: E402

# ------------------------------------------------------------------ method sets / labels
# (copied from rl_report.py so this stays standalone -- imports only numpy + matplotlib)
BASE_ORDER = ["go_coverage", "go_facility", "go_facility_idf", "go_facility_emb",
              "greedy_rew400_c120", "greedy_aux_all"]
BLIND_RL = ["ppo_blind", "ppo_blind_match", "ppo_blind_rich", "ppo_blind_xline",
            "ppo_blind_xline_rich"]
RL_ORDER = ["pg_tab", "pg_tab_boot", "critic_only", "critic_shrink", "ppo_peek"] + BLIND_RL
REL_BLIND = ["iterpert", "iterpert_go", "iterpert_cold", "kc_go", "kc_iterpert_priors",
             "fl_iterpert_priors", "al_coreset", "al_bald", "al_batchbald",
             "al_badge", "al_lcmd", "al_acsfw", "al_kmeans", "al_typiclust"]
REL_PEEK = ["dv_shapley", "dv_banzhaf", "dv_datamodels", "rep_facility_sig", "rep_coreset_sig"]
RELATED = REL_BLIND + REL_PEEK
BLIND = {"go_coverage", "go_facility", "go_facility_idf", "go_facility_emb", *BLIND_RL, *REL_BLIND}
OWN = {"blind": ["go_facility", "go_facility_idf", "ppo_blind", "ppo_blind_xline_rich"],
       "peek": ["greedy_aux_all", "pg_tab", "critic_shrink", "ppo_peek"]}
# compact labels sized for a 7-inch two-panel figure (full names are in the caption)
LABEL = {
    "go_coverage": "GO coverage (static)",
    "go_facility": "GO facility location (static)",
    "go_facility_idf": "facility loc., IDF-GO (static)",
    "go_facility_emb": "facility loc., GO embedding (static)",
    "greedy_rew400_c120": "greedy, GAPS protocol (peek)",
    "greedy_aux_all": "greedy, all aux genes (peek)",
    "pg_tab": "REINFORCE, per-gene logits",
    "pg_tab_boot": "REINFORCE, resampled reward",
    "critic_only": "learned critic, greedy",
    "critic_shrink": "critic-shrunk greedy",
    "ppo_peek": "PPO feature policy (peek)",
    "ppo_blind": "PPO blind (GO only)",
    "ppo_blind_match": "PPO blind, size-matched",
    "ppo_blind_rich": "PPO blind, richer priors",
    "ppo_blind_xline": "PPO blind, other cell line",
    "ppo_blind_xline_rich": "PPO blind, other line + priors",
    "iterpert": "IterPert (8 released priors)",
    "iterpert_go": "IterPert, GO prior only",
    "iterpert_cold": "IterPert, no random first batch",
    "kc_go": "k-centre on GO (static)",
    "kc_iterpert_priors": "k-centre on IterPert priors (static)",
    "fl_iterpert_priors": "facility loc. on IterPert priors (static)",
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
    "rep_facility_sig": "exemplar facility loc., signatures",
    "rep_coreset_sig": "Core-Set on true signatures",
}
CURVE_STYLE = {"ppo_blind": "-", "ppo_blind_match": (0, (4, 2)), "ppo_blind_rich": (0, (1, 1.5)),
               "ppo_blind_xline": (0, (6, 2, 1, 2)), "ppo_blind_xline_rich": (0, (3, 1, 1, 1, 1, 1))}
SHORT = {"replogle_k562_essential": "K562", "replogle_rpe1_essential": "RPE1"}
DS_ORDER = ["replogle_k562_essential", "replogle_rpe1_essential"]


def _load_jsonl(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def _colour(m):
    """Journal palette: grey = static/greedy baseline, red = peek result, blue = blind."""
    if m in BASE_ORDER:
        return DARK
    return BLUE if m in BLIND else RED


def _marker(m):
    return "D" if m in BASE_ORDER else ("s" if m in RELATED else "o")


# ================================================================== fig15: RL vs baselines
def fig_acq(recs, fig_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    datasets = [d for d in DS_ORDER if any(r["dataset"] == d for r in recs)]
    methods = [m for m in BASE_ORDER + RL_ORDER if any(r["method"] == m for r in recs)]
    nb = len([m for m in methods if m in BASE_ORDER])
    fig, axes = plt.subplots(1, len(datasets), figsize=(W2, 0.205 * len(methods) + 1.0),
                             sharey=True, squeeze=False)
    for k, (ax, ds) in enumerate(zip(axes[0], datasets)):
        xs = [r for r in recs if r["dataset"] == ds]
        p05 = np.mean([r["ref_p05"] / r["ref_med"] for r in xs])
        p95 = np.mean([r["ref_p95"] / r["ref_med"] for r in xs])
        ch = np.mean([r["ref_cheat"] / r["ref_med"] for r in xs])
        ax.axvspan(p05, p95, color=FAINT, zorder=0, lw=0)
        ax.axvline(1.0, color=MID, lw=0.6, zorder=1)
        ax.axvline(ch, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=1)
        for i, m in enumerate(methods):
            v = [r["ratio"] for r in xs if r["method"] == m]
            if not v:
                continue
            col = _colour(m)
            jit = np.linspace(-0.16, 0.16, len(v)) if len(v) > 1 else [0.0]
            ax.scatter(v, i + np.asarray(jit), s=4.5, color=col, alpha=0.55, lw=0, zorder=2)
            ax.scatter([np.mean(v)], [i], s=24, color=col, edgecolor="white", lw=0.5, zorder=3,
                       marker=_marker(m))
            ax.annotate(f"{np.mean(v):.2f}$\\times$", (max(np.max(v), np.mean(v)) + 0.015, i),
                        va="center", ha="left", fontsize=6.5, color=INK, zorder=4)
        ax.axhline(nb - 0.5, color=LIGHT, lw=0.6)
        ax.set_xlabel("report MRR $\\div$ random-$\\times$100 median")
        ax.set_ylim(len(methods) - 0.4, -1.0)
        ax.set_title(SHORT[ds], loc="left")
        _panel(ax, "ab"[k], x=-0.9 if k == 0 else -0.06, y=1.0)
        ax.text(ch, -0.75, f" cheat {ch:.2f}$\\times$", color=INK, fontsize=6, va="center")
        ax.text((p05 + p95) / 2, -0.75, "random\np05-p95", color=MID, fontsize=5.5, ha="center",
                va="center", linespacing=0.9)
    axes[0][0].set_yticks(range(len(methods)))
    axes[0][0].set_yticklabels([LABEL[m] for m in methods])
    axes[0][0].tick_params(axis="y", length=0)
    h = [Line2D([], [], marker="D", ls="", color=DARK, ms=4, label="baseline (static / greedy)"),
         Line2D([], [], marker="o", ls="", color=RED, ms=4, label="RL, peek (greedy's information)"),
         Line2D([], [], marker="o", ls="", color=BLUE, ms=4,
                label="RL, blind (signature seen only once measured)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    _save(fig, fig_dir, "fig15_rl_acquisition")


# ================================================================== fig17: published methods
def fig_related(recs, fig_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    datasets = [d for d in DS_ORDER if any(r["dataset"] == d for r in recs)]
    have = {r["method"] for r in recs}
    groups = [("Blind: a candidate's signature is read only once measured",
               [m for m in OWN["blind"] + REL_BLIND if m in have]),
              ("Peek: every non-report signature is visible (greedy's information)",
               [m for m in OWN["peek"] + REL_PEEK if m in have])]
    rows = []
    for title, ms in groups:
        rows.append(("hdr", title))
        rows += [("m", m) for m in ms]
    fig, axes = plt.subplots(1, len(datasets), figsize=(W2, 0.185 * len(rows) + 1.0),
                             sharey=True, squeeze=False)
    for k, (ax, ds) in enumerate(zip(axes[0], datasets)):
        xs = [r for r in recs if r["dataset"] == ds]
        p05 = np.mean([r["ref_p05"] / r["ref_med"] for r in xs])
        p95 = np.mean([r["ref_p95"] / r["ref_med"] for r in xs])
        fb = np.mean([r["ref_facility_best"] / r["ref_med"] for r in xs])
        ax.axvspan(p05, p95, color=FAINT, zorder=0, lw=0)
        ax.axvline(1.0, color=MID, lw=0.6, zorder=1)
        ax.axvline(fb, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=1)
        for i, (kind, m) in enumerate(rows):
            if kind == "hdr":
                continue
            v = [r["ratio"] for r in xs if r["method"] == m]
            if not v:
                continue
            col = _colour(m)
            mk = _marker(m)
            jit = np.linspace(-0.15, 0.15, len(v)) if len(v) > 1 else [0.0]
            ax.scatter(v, i + np.asarray(jit), s=4, color=col, alpha=0.5, lw=0, zorder=2, marker=mk)
            ax.scatter([np.mean(v)], [i], s=21, color=col, edgecolor="white", lw=0.5, zorder=3, marker=mk)
            ax.annotate(f"{np.mean(v):.2f}$\\times$", (max(np.max(v), np.mean(v)) + 0.015, i),
                        va="center", ha="left", fontsize=6, color=INK, zorder=4)
        for i, (kind, _) in enumerate(rows):
            if kind == "hdr" and i > 0:
                ax.axhline(i - 0.5, color=LIGHT, lw=0.6)
        ax.set_xlabel("report MRR $\\div$ random-$\\times$100 median")
        ax.set_ylim(len(rows) - 0.4, -1.0)
        ax.set_title(SHORT[ds], loc="left")
        _panel(ax, "ab"[k], x=-1.02 if k == 0 else -0.06, y=1.0)
        ax.text(fb, -0.72, f" static floor {fb:.2f}$\\times$", color=INK, fontsize=6, va="center")
    lab = []
    for kind, t in rows:
        if kind == "hdr":
            lab.append(t)
        else:
            lab.append(LABEL.get(t, t))
    axes[0][0].set_yticks(range(len(rows)))
    axes[0][0].set_yticklabels(lab)
    axes[0][0].tick_params(axis="y", length=0)
    for lb, (kind, _) in zip(axes[0][0].get_yticklabels(), rows):
        if kind == "hdr":
            lb.set_fontweight("bold")
            lb.set_fontstyle("italic")
    h = [Line2D([], [], marker="D", ls="", color=DARK, ms=4, label="static / greedy baseline (ours)"),
         Line2D([], [], marker="o", ls="", color=INK, ms=4, label="our RL"),
         Line2D([], [], marker="s", ls="", color=INK, ms=4, label="published method"),
         Line2D([], [], marker="s", ls="", color=RED, ms=4, label="peek regime"),
         Line2D([], [], marker="s", ls="", color=BLUE, ms=4, label="blind regime")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    _save(fig, fig_dir, "fig17_related_work")


# ================================================================== fig16: PPO curves
def fig_curves(runs_dir, ref_med, fig_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    curves = sorted(glob.glob(os.path.join(runs_dir, "ppo_*.curve.jsonl")))
    datasets = [d for d in DS_ORDER if any(f"__{d}__" in os.path.basename(p) for p in curves)]
    if not datasets:
        print("[paper] no ppo curves; skip fig16")
        return
    fig, axes = plt.subplots(2, len(datasets), figsize=(W2, 4.3), squeeze=False, sharex=True)
    seen = set()
    for j, ds in enumerate(datasets):
        for p in curves:
            nm = os.path.basename(p)
            if f"__{ds}__" not in nm:
                continue
            me = nm.split("__")[0]
            sp = int(nm.split("__s")[1].split("__")[0])
            med = ref_med.get((ds, sp))
            c = _load_jsonl(p)
            if not c or med is None:
                continue
            seen.add(me)
            it = [x["iter"] for x in c]
            col = RED if me == "ppo_peek" else BLUE
            st = "-" if me == "ppo_peek" else CURVE_STYLE.get(me, "-")
            axes[0][j].plot(it, [x["val_mrr"] / x["facility"] for x in c], color=col, lw=0.7, alpha=0.6, ls=st)
            axes[1][j].plot(it, [x["rep_mrr"] / med for x in c], color=col, lw=0.7, alpha=0.6, ls=st)
        axes[0][j].axhline(1.0, color=INK, lw=0.6, ls=(0, (4, 2)))
        axes[1][j].axhline(1.0, color=MID, lw=0.6)
        axes[0][j].set_title(SHORT[ds], loc="left")
        _panel(axes[0][j], "ab"[j] if len(datasets) > 1 else "a", x=-0.17, y=1.0)
        _panel(axes[1][j], "cd"[j] if len(datasets) > 1 else "b", x=-0.17, y=1.0)
        axes[1][j].set_xlabel("PPO iteration")
        if j == 0:
            axes[0][j].set_ylabel("validation MRR $\\div$\nfacility location")
            axes[1][j].set_ylabel("report MRR $\\div$ random median\n(diagnostic only, never selects)")
    h = [Line2D([], [], color=RED, lw=1.0, label="PPO peek")] + [
        Line2D([], [], color=BLUE, lw=1.0, ls=CURVE_STYLE[m], label=LABEL[m])
        for m in BLIND_RL if m in seen]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    _save(fig, fig_dir, "fig16_rl_curves")


# ================================================================== fig13: headroom by axis
def fig_headroom(summary, fig_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    H = summary["headroom"]

    def pick(axis, setting, space):
        for h in H:
            if h["axis"] == axis and h["setting"] == setting and h["space"] == space:
                return h
        return None

    # the rows, in the order GAPS presents them (masked space, i.e. target-gene fingerprint removed)
    want = [
        ("gene", "norman", "masked", "unseen gene $\\cdot$ Norman"),
        ("gene", "adamson", "masked", "unseen gene $\\cdot$ Adamson"),
        ("gene", "replogle_rpe1_essential", "masked", "unseen gene $\\cdot$ RPE1"),
        ("gene", "replogle_k562_essential", "masked", "unseen gene $\\cdot$ K562"),
        None,
        ("combo", "norman (GEARS gene space)", "masked", "unseen pair $\\cdot$ Norman combos"),
        None,
        ("context", "replogle_k562_essential -> adamson", "masked", "unseen context $\\cdot$ K562$\\to$Adamson"),
        ("context", "adamson -> replogle_rpe1_essential", "masked", "unseen context $\\cdot$ Adamson$\\to$RPE1"),
        ("context", "adamson -> replogle_k562_essential", "masked", "unseen context $\\cdot$ Adamson$\\to$K562"),
        ("context", "replogle_k562_essential -> replogle_rpe1_essential", "masked",
         "unseen context $\\cdot$ K562$\\to$RPE1"),
        ("context", "replogle_rpe1_essential -> replogle_k562_essential", "masked",
         "unseen context $\\cdot$ RPE1$\\to$K562"),
    ]
    rows = []
    for w in want:
        if w is None:
            rows.append(None)
            continue
        h = pick(w[0], w[1], w[2])
        if h is not None:
            rows.append((w[3] + f"  ($K$={h['K']})", h))
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 0.30 * n + 0.9),
                             gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.08})
    axL, axR = axes
    ymax = n - 1
    for i, row in enumerate(rows):
        y = ymax - i
        if row is None:
            continue
        _, h = row
        ch = 100 * h["chance_top1"]
        cheap = 100 * h["best_cheap_top1"]
        ceil = 100 * h["ceiling_top1"]
        axL.plot([ch, ceil], [y, y], color=LIGHT, lw=2.2, solid_capstyle="round", zorder=1)
        axL.scatter([ch], [y], s=16, facecolors="white", edgecolors=MID, linewidths=0.7, zorder=3)
        is_combo = h["axis"] == "combo"
        axL.scatter([cheap], [y], s=22, color=(RED if is_combo else BLUE), zorder=4,
                    marker=("D" if is_combo else "o"))
        axL.scatter([ceil], [y], s=22, color=INK, zorder=4)
        axL.annotate(f"{ceil - ch:.0f} pts", (ceil, y), xytext=(5, 0), textcoords="offset points",
                     va="center", ha="left", fontsize=6, color=INK)
        # right: fraction of (ceiling - chance) the cheap predictor recovers, by MRR
        rec = 100 * h["frac_recovered_mrr"]
        axR.barh(y, rec, height=0.5, color=(RED if is_combo else BLUE), zorder=2)
        axR.annotate(f"{rec:.0f}%  {h['best_cheap']}", (rec, y), xytext=(4, 0),
                     textcoords="offset points", va="center", ha="left", fontsize=6, color=INK)
    axL.set_yticks([ymax - i for i, r in enumerate(rows) if r is not None])
    axL.set_yticklabels([r[0] for r in rows if r is not None])
    axL.tick_params(axis="y", length=0)
    axL.set_ylim(-0.7, ymax + 0.7)
    axL.set_xlim(-2, 100)
    axL.set_xlabel("top-1 accuracy (%); target = $\\leq$32 held-out cells, target genes masked")
    axR.set_ylim(-0.7, ymax + 0.7)
    axR.set_xlim(0, 118)
    axR.set_yticks([])
    axR.spines["left"].set_visible(False)
    axR.set_xlabel("% of (ceiling $-$ chance) recovered")
    _panel(axL, "a", x=-0.86, y=1.01)
    _panel(axR, "b", x=-0.03, y=1.01)
    h = [Line2D([], [], marker="o", ls="", mfc="white", mec=MID, ms=4, label="chance (1/$K$)"),
         Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="best cheap predictor"),
         Line2D([], [], marker="o", ls="", color=INK, ms=4, label="ceiling (candidate measured)"),
         Line2D([], [], marker="D", ls="", color=RED, ms=4, label="additive fit (Norman combos)")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    _save(fig, fig_dir, "fig13_headroom_by_axis")


# ================================================================== entry
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    a = ap.parse_args()
    _style()
    os.makedirs(a.fig_dir, exist_ok=True)

    rec_p = os.path.join(a.art_dir, "rl_acquire", "records.jsonl")
    recs = _load_jsonl(rec_p) if os.path.exists(rec_p) else []
    ref_med = {(r["dataset"], r["split"]): r["ref_med"] for r in recs}
    runs_dir = os.path.join(a.art_dir, "rl_acquire", "runs")
    gen_p = os.path.join(a.art_dir, "generalization", "summary.json")

    todo = [("fig13", lambda: fig_headroom(json.load(open(gen_p)), a.fig_dir) if os.path.exists(gen_p) else None),
            ("fig15", lambda: fig_acq(recs, a.fig_dir) if recs else None),
            ("fig16", lambda: fig_curves(runs_dir, ref_med, a.fig_dir)),
            ("fig17", lambda: fig_related(recs, a.fig_dir) if recs else None)]
    for name, fn in todo:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[paper] {name} FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
