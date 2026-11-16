#!/usr/bin/env python
"""Redraw the generalization-headroom figures from artifacts/generalization/ only.

fig13_headroom_by_axis   chance / best cheap predictor / ceiling, per setting
fig14_context_transfer   cross-context: few-shot curve + disattenuated similarity

Usage: python -I generalization_figures.py --artifacts DIR --figures DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# same palette and style as make_figures.py so the figure set reads as one
PALETTE = ["#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0891b2"]
GREY = "#6b7280"
INK = "#111827"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "600", "axes.labelsize": 9.5,
    "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e7eb",
    "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})

SHORT = {
    "replogle_k562_essential": "K562", "replogle_rpe1_essential": "RPE1",
    "adamson": "Adamson", "norman": "Norman", "norman (GEARS gene space)": "Norman",
}
AXIS_LABEL = {"gene": "unseen gene", "combo": "unseen pair", "context": "unseen context"}
AXIS_ORDER = {"gene": 0, "combo": 1, "context": 2}
PRED_LABEL = {
    "linear": "linear (Ahlmann-Eltze)", "coresp_knn": "co-response kNN", "go_knn": "GO kNN",
    "mean": "mean effect", "barcode": "target-only", "additive": "additive singles",
    "additive_fit": "additive, fitted", "naive": "other line, as is",
    "shifted": "other line, re-centred", "ridge_map": "other line, ridge map",
    "within_linear": "this line, linear", "within_coresp_knn": "this line, co-response",
    "mean_tgt": "mean effect",
}


def short_setting(axis, s):
    if axis == "context":
        a, b = s.split(" -> ")
        return f"{SHORT.get(a, a)} → {SHORT.get(b, b)}"
    return SHORT.get(s, s)


def finish(fig, path, note=None):
    if note:
        w = max(110, int(fig.get_size_inches()[0] * 15))
        fig.text(0.005, -0.02, "\n".join(textwrap.wrap(note, w)), fontsize=7.5, color=GREY, va="top")
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path, flush=True)


def load(artdir):
    p = os.path.join(artdir, "generalization", "results.jsonl")
    with open(p) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    sp = os.path.join(artdir, "generalization", "summary.json")
    summ = json.load(open(sp)) if os.path.exists(sp) else {}
    return rows, summ


# ================================================================ fig 13
def fig_headroom(rows, summ, figdir):
    H = [h for h in summ.get("headroom", []) if h["space"] == "masked"]
    if not H:
        print("  (no masked headroom rows; skipping fig13)")
        return
    H.sort(key=lambda h: (AXIS_ORDER.get(h["axis"], 9), -h["ceiling_top1"]))
    gears = {(r["axis"], r["setting"]): r["top1"] for r in rows
             if r.get("predictor") == "gears_seen_all" and r.get("space") == "masked"
             and "kind" not in r}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.48 * len(H) + 1.9),
                             gridspec_kw={"width_ratios": [1.65, 1]})
    fig.subplots_adjust(wspace=0.08)
    ax = axes[0]
    y = np.arange(len(H))[::-1]
    for yi, h in zip(y, H):
        ch, be, ce = 100 * h["chance_top1"], 100 * h["best_cheap_top1"], 100 * h["ceiling_top1"]
        ax.plot([be, ce], [yi, yi], color="#cbd5e1", lw=5, solid_capstyle="round", zorder=1)
        ax.scatter([ch], [yi], s=46, color="white", edgecolor=GREY, lw=1.4, zorder=3)
        ax.scatter([be], [yi], s=70, color=PALETTE[0], edgecolor="white", lw=1.5, zorder=4)
        ax.scatter([ce], [yi], s=70, color=INK, edgecolor="white", lw=1.5, zorder=4)
        g = gears.get((h["axis"], h["setting"]))
        if g is not None:
            ax.scatter([100 * g], [yi], s=70, marker="D", color=PALETTE[1], edgecolor="white",
                       lw=1.5, zorder=5)
        gap = ce - be
        ax.text(max(ce, be) + 1.2, yi,
                f"{gap:.0f} pts" if gap >= 0.5 else ("≈0" if gap > -0.5 else f"−{-gap:.0f} pts"),
                va="center", fontsize=8, color=INK if gap >= 0.5 else GREY)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{AXIS_LABEL[h['axis']]}  ·  {short_setting(h['axis'], h['setting'])}"
                        f"  (K={h['K']})" for h in H], fontsize=8.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("top-1 accuracy, % (target = pseudo-bulk of ≤32 held-out cells; target genes masked)")
    ax.set_title("Headroom = ceiling − best cheap predictor", loc="left")
    ax.grid(axis="y", visible=False)
    for k in range(1, len(H)):
        if H[k]["axis"] != H[k - 1]["axis"]:
            ax.axhline(y[k] + 0.5, color="#e5e7eb", lw=1)
    leg = [Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, ms=7, label="chance (1/K)"),
           Line2D([], [], marker="o", ls="", color=PALETTE[0], ms=8, label="best cheap predictor"),
           Line2D([], [], marker="o", ls="", color=INK, ms=8, label="ceiling (candidate measured)")]
    if gears:
        leg.append(Line2D([], [], marker="D", ls="", color=PALETTE[1], ms=7,
                          label="GEARS, trained on these pairs"))
    ax.legend(handles=leg, loc="lower right", fontsize=8)

    ax = axes[1]
    fr = [h["frac_recovered_mrr"] for h in H]
    ax.barh(y, [100 * max(0, f) for f in fr], color=PALETTE[0], height=0.55)
    for yi, f, h in zip(y, fr, H):
        ax.text(max(0, 100 * f) + 1.5, yi, f"{100 * f:.0f}%  {PRED_LABEL.get(h['best_cheap'], h['best_cheap'])}",
                va="center", fontsize=7.8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 130)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of (ceiling − chance) recovered, by mean reciprocal rank")
    ax.set_title("How much a cheap predictor already closes", loc="left")
    ax.grid(axis="y", visible=False)
    finish(fig, os.path.join(figdir, "fig13_headroom_by_axis.png"),
           "Source: artifacts/generalization/summary.json. Every predictor is scored on the same queries "
           "against ALL K candidates; learned predictors are out-of-fold (5 folds over candidates). "
           "Masked = every gene targeted by any candidate removed from both target and signatures, so a "
           "target carries no knocked-down-gene fingerprint. 'Best cheap' is chosen by MRR among "
           "non-GEARS predictors.")


# ================================================================ fig 14
def fig_context(rows, figdir):
    pairs = [("replogle_k562_essential -> replogle_rpe1_essential"),
             ("replogle_rpe1_essential -> replogle_k562_essential")]
    few = [r for r in rows if r.get("kind") == "fewshot"]
    dis = [r for r in rows if r.get("kind") == "disattenuated" and r.get("space") == "masked"]
    main = [r for r in rows if r.get("axis") == "context" and "kind" not in r
            and r.get("space") == "masked"]
    if not main:
        print("  (no context rows; skipping fig14)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.1))
    fig.subplots_adjust(wspace=0.52)
    for k, pr in enumerate(pairs):
        ax = axes[k]
        m = {r["predictor"]: r for r in main if r["setting"] == pr}
        if not m:
            ax.set_visible(False)
            continue
        for p, col in (("ridge_map", PALETTE[0]), ("shifted", PALETTE[2])):
            rs = [r for r in few if r["setting"] == pr and r["predictor"] == p]
            if not rs:
                continue
            ns = sorted({r["n_train"] for r in rs})
            mu = [100 * np.mean([r["top1"] for r in rs if r["n_train"] == n]) for n in ns]
            ax.plot(ns, mu, "-o", color=col, lw=2, ms=5, label=PRED_LABEL[p])
            ax.text(ns[-1] * 1.08, mu[-1], f"{mu[-1]:.0f}%", va="center", fontsize=8, color=INK)
        for p, ls, col in (("ceiling", "-", INK), ("naive", "--", GREY),
                           ("within_linear", ":", PALETTE[3])):
            if p in m:
                v = 100 * m[p]["top1"]
                ax.axhline(v, ls=ls, color=col, lw=1.4)
                if p == "ceiling":
                    ax.text(10.5, v + 1.0, f"ceiling: this perturbation measured here  {v:.0f}%",
                            fontsize=7.8, color=col)
                else:
                    lab = {"naive": "other line, as is",
                           "within_linear": "this line only, linear"}[p]
                    ax.text(1250, v, f"{lab}  {v:.0f}%", fontsize=7.8, color=col, va="center")
        ax.set_xscale("log")
        ax.set_xlim(9, 1200)
        ax.set_ylim(0, 48)
        ax.set_xlabel("perturbations measured in BOTH lines (training pairs)")
        ax.set_ylabel("top-1 accuracy, % (masked)")
        a, b = pr.split(" -> ")
        ax.set_title(f"{SHORT[a]} atlas → design in {SHORT[b]}  (K={m['ceiling']['K']})", loc="left")
        ax.legend(loc="center right", fontsize=8)

    ax = axes[2]
    # the comparison is symmetric: keep one curve per unordered pair of contexts
    seen = set()
    i = 0
    for r in sorted(dis, key=lambda r: -r["median_disattenuated"]):
        v = np.sort(np.asarray(r["values_disattenuated"]))
        a, b = r["setting"].split(" -> ")
        if v.size < 5 or frozenset((a, b)) in seen:
            continue
        seen.add(frozenset((a, b)))
        ax.plot(v, 100 * np.arange(1, v.size + 1) / v.size, lw=2, color=PALETTE[i % len(PALETTE)],
                label=f"{SHORT[a]} vs {SHORT[b]}  (n={v.size}, median {np.median(v):.2f})")
        i += 1
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.text(0.98, 72, "same effect,\nup to noise ", fontsize=7.5, color=GREY, ha="right")
    ax.set_xlim(-0.2, 1.3)
    ax.set_xlabel("disattenuated cosine of one perturbation across contexts")
    ax.set_ylabel("% of shared perturbations ≤ x")
    ax.set_title("Is it the same effect in the other context?", loc="left")
    ax.legend(loc="lower right", fontsize=7.5)
    finish(fig, os.path.join(figdir, "fig14_context_transfer.png"),
           "Source: artifacts/generalization/results.jsonl. Left/middle: few-shot curves are means over 3 "
           "subsamples of the training pairs per fold; the candidate set is all shared perturbations. "
           "Right: cross-context cosine divided by sqrt(R_a R_b), the value two noisy measurements of an "
           "identical effect would reach (R = split-half reliability, Spearman-Brown corrected); "
           "perturbations with ceiling < 0.2 dropped. Target genes masked throughout.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--figures", required=True)
    a = ap.parse_args()
    os.makedirs(a.figures, exist_ok=True)
    rows, summ = load(a.artifacts)
    print(f"{len(rows)} result rows")
    for fn in (fig_headroom, fig_context):
        try:
            if fn is fig_headroom:
                fn(rows, summ, a.figures)
            else:
                fn(rows, a.figures)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {fn.__name__} failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
