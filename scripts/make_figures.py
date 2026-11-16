#!/usr/bin/env python
"""Redraw every figure from the artifacts on disk. No model, no recomputation.

Reads  : <artifacts>/dataset_profiles.json, condition_stats.jsonl,
         retrieval.jsonl, npz/*_signatures.npz, oracle_probe.json (optional)
Writes : <figures>/*.png

Usage: python -I make_figures.py --artifacts DIR --figures DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- style
# Brand-neutral categorical palette, ordered so adjacent series stay separable
# in both light and dark viewing and under common colour-vision deficiencies.
PALETTE = ["#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0891b2"]
GREY = "#6b7280"
GRID = "#e5e7eb"
INK = "#111827"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#9ca3af",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
    }
)

DS_ORDER = [
    "norman",
    "adamson",
    "dixit",
    "replogle_k562_essential",
    "replogle_rpe1_essential",
]
SHORT = {
    "norman": "Norman\n(combos)",
    "adamson": "Adamson",
    "dixit": "Dixit",
    "replogle_k562_essential": "Replogle\nK562",
    "replogle_rpe1_essential": "Replogle\nRPE1",
}


def short(d):
    return SHORT.get(d, d)


def colour_for(datasets):
    return {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(datasets)}


def jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def order_datasets(names):
    known = [d for d in DS_ORDER if d in names]
    return known + sorted(set(names) - set(known))


def finish(fig, path, note=None):
    if note:
        import textwrap
        w = max(110, int(fig.get_size_inches()[0] * 15))
        fig.text(0.005, -0.02, "\n".join(textwrap.wrap(note, w)),
                 fontsize=7.5, color=GREY, va="top")
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path, flush=True)


# ================================================================ figure 1
def fig_action_space(profiles, figdir):
    """How big is the design space the generator chooses from, really?"""
    names = order_datasets([p["dataset"] for p in profiles])
    P = {p["dataset"]: p for p in profiles}
    cmap = colour_for(names)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9))
    fig.subplots_adjust(wspace=0.58)

    ax = axes[0]
    genes = [P[d]["n_unique_perturbed_genes"] for d in names]
    bars = ax.barh(range(len(names)), genes, color=[cmap[d] for d in names], height=0.62)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([short(d).replace("\n", " ") for d in names])
    ax.invert_yaxis()
    ax.set_xlabel("unique perturbable genes")
    ax.set_title("Action space: genes that can be perturbed")
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    for b, v in zip(bars, genes):
        ax.text(b.get_width() * 1.12, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=8.5, color=INK)
    ax.set_xlim(right=max(genes) * 22)
    # ~20k protein-coding genes is the space a practitioner actually cares about
    ax.axvline(20000, color=GREY, ls=":", lw=1.1)
    ax.text(20000, len(names) - 0.35, "  ≈20k protein-coding\n  (what a practitioner wants)",
            fontsize=7.5, color=GREY, ha="left", va="bottom")

    ax = axes[1]
    w = 0.38
    xs = np.arange(len(names))
    sing = [P[d]["n_single_conditions"] for d in names]
    comb = [P[d]["n_combo_conditions"] for d in names]
    ax.bar(xs - w / 2, sing, w, label="single-gene", color=PALETTE[0])
    ax.bar(xs + w / 2, comb, w, label="combinatorial", color=PALETTE[1])
    ax.set_xticks(xs)
    ax.set_xticklabels([short(d) for d in names], fontsize=8)
    ax.set_ylabel("labelled conditions")
    ax.set_title("Training signal: conditions with data")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    for x, v in zip(xs - w / 2, sing):
        if v:
            ax.text(x, v, f"{v}", ha="center", va="bottom", fontsize=7.5, color=INK)
    for x, v in zip(xs + w / 2, comb):
        if v:
            ax.text(x, v, f"{v}", ha="center", va="bottom", fontsize=7.5, color=PALETTE[1])

    ax = axes[2]
    frac = [100 * P[d].get("frac_perturbed_genes_measured", np.nan) for d in names]
    bars = ax.barh(range(len(names)), frac, color=[cmap[d] for d in names], height=0.62)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([short(d).replace("\n", " ") for d in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of perturbed genes also measured")
    ax.set_title("Cycle consistency: can the oracle\nsee the gene you perturbed?")
    ax.grid(axis="y", visible=False)
    for b, v in zip(bars, frac):
        ax.text(min(b.get_width() + 2, 101), b.get_y() + b.get_height() / 2,
                f"{v:.0f}%", va="center", fontsize=8.5, color=INK)

    finish(fig, os.path.join(figdir, "fig1_action_space.png"),
           "Source: artifacts/dataset_profiles.json. GEARS-processed perturb-seq releases.")


# ================================================================ figure 2
def fig_identifiability(retr, profiles, figdir):
    """The headline: how many cells must you see before a target state names
    its own cause? This is the ceiling on any inverse model."""
    names = order_datasets({r["dataset"] for r in retr})
    cmap = colour_for(names)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    for d in names:
        rows = sorted([r for r in retr if r["dataset"] == d], key=lambda r: r["n_cells_observed"])
        if not rows:
            continue
        x = [r["n_cells_observed"] for r in rows]
        y = [100 * r["top1_acc"] for r in rows]
        ax.plot(x, y, "-o", color=cmap[d], ms=4.5, lw=1.9,
                label=f"{short(d).replace(chr(10),' ')} (K={rows[0]['n_candidate_conditions']})")
        ax.axhline(100 * rows[0]["chance_top1"], color=cmap[d], ls=":", lw=1.0, alpha=0.75)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("cells of the same unknown perturbation observed")
    ax.set_ylabel("top-1 accuracy  (%)")
    ax.set_title("Identifiability of the inverse map\nnearest-centroid retrieval, held-out cells")
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=GREY, ls=":", lw=1.0))
    l.append("chance (1/K)")
    ax.legend(h, l, loc="upper left")

    ax = axes[1]
    for d in names:
        rows = sorted([r for r in retr if r["dataset"] == d], key=lambda r: r["n_cells_observed"])
        if not rows:
            continue
        x = [r["n_cells_observed"] for r in rows]
        y = [100 * r["top10_acc"] for r in rows]
        ax.plot(x, y, "-o", color=cmap[d], ms=4.5, lw=1.9,
                label=short(d).replace("\n", " "))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("cells of the same unknown perturbation observed")
    ax.set_ylabel("top-10 accuracy  (%)")
    ax.set_title("Top-10: is the right perturbation even\nin a shortlist a practitioner would screen?")
    ax.legend(loc="upper left")

    finish(fig, os.path.join(figdir, "fig2_identifiability.png"),
           "Nearest-centroid on delta-from-control signatures; half of each condition's cells build "
           "the centroid, the other half are queries. This is a training-free lower bound every "
           "learned inverse model must beat.")


# ================================================================ figure 3
def fig_effect_noise(cond, figdir):
    """Most perturbations barely move the cell. Those are the ones where a
    target state cannot possibly identify a cause."""
    names = order_datasets({r["dataset"] for r in cond})
    cmap = colour_for(names)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    ax = axes[0]
    for d in names:
        v = np.array([r["effect_to_noise"] for r in cond if r["dataset"] == d])
        if v.size < 2:
            continue
        v = v[np.isfinite(v)]
        xs = np.sort(v)
        ax.plot(xs, 100 * np.arange(1, xs.size + 1) / xs.size, lw=1.9, color=cmap[d],
                label=f"{short(d).replace(chr(10),' ')} (n={xs.size})")
    ax.axvline(1.0, color=GREY, ls="--", lw=1.1)
    ax.text(1.06, 4, "shift = per-cell scatter", fontsize=7.5, color=GREY,
            rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("‖Δ mean‖ / mean per-cell residual norm")
    ax.set_ylabel("% of conditions ≤ x")
    ax.set_title("Single-cell SNR: the perturbation shift is\nsmall next to per-cell scatter")
    ax.legend(loc="lower right")

    ax = axes[1]
    for d in names:
        v = np.array([r["nn_cosine"] for r in cond if r["dataset"] == d])
        if v.size < 2:
            continue
        xs = np.sort(v[np.isfinite(v)])
        ax.plot(xs, 100 * np.arange(1, xs.size + 1) / xs.size, lw=1.9, color=cmap[d],
                label=short(d).replace("\n", " "))
    ax.set_xlabel("cosine to nearest OTHER perturbation signature")
    ax.set_ylabel("% of conditions ≤ x")
    ax.set_title("Signature confusability\n(high ⇒ one-to-many inverse map)")
    ax.legend(loc="upper left")

    ax = axes[2]
    for d in names:
        rows = [r for r in cond if r["dataset"] == d]
        if len(rows) < 2:
            continue
        x = np.array([r["effect_to_noise"] for r in rows])
        y = np.array([r["frac_de_z2"] for r in rows]) * 100
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=11, alpha=0.5, color=cmap[d], lw=0,
                   label=short(d).replace("\n", " "))
    ax.set_xscale("log")
    ax.set_xlabel("effect-to-noise")
    ax.set_ylabel("% of measured genes responding (|z| > 2)")
    ax.set_title("Sparsity reward vs biology:\nstrong perturbations move many genes")
    ax.legend(loc="upper left", markerscale=1.8)

    finish(fig, os.path.join(figdir, "fig3_effect_and_confusability.png"),
           "Source: artifacts/condition_stats.jsonl. Left: a ratio below 1 is expected in ~5,000 "
           "dimensions and means a SINGLE cell's deviation is noise-dominated, not that the "
           "perturbation has no effect — pooling cells recovers it (fig 2). z computed against "
           "per-gene control SD: an effect-size proxy, not a calibrated DE test.")


# ================================================================ figure 4
def fig_data_per_condition(profiles, cond, figdir):
    """RL sample budget: how much real data backs each action?"""
    names = order_datasets([p["dataset"] for p in profiles])
    P = {p["dataset"]: p for p in profiles}
    cmap = colour_for(names)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax = axes[0]
    data, labs, cols = [], [], []
    for d in names:
        v = [r["n_cells"] for r in cond if r["dataset"] == d]
        if v:
            data.append(v)
            labs.append(short(d))
            cols.append(cmap[d])
    if data:
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops=dict(color="white", lw=1.6),
                        whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_edgecolor(c)
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(labs) + 1))
        ax.set_xticklabels(labs, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("cells per condition")
    ax.set_title("Data behind each action")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    xs = np.arange(len(names))
    ctrl = [P[d]["n_control_cells"] for d in names]
    tot = [P[d]["n_cells"] for d in names]
    ax.bar(xs, tot, 0.6, color=GRID, label="all cells", edgecolor="none")
    ax.bar(xs, ctrl, 0.6, color=PALETTE[0], label="control cells", edgecolor="none")
    ax.set_xticks(xs)
    ax.set_xticklabels([short(d) for d in names], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("cells")
    ax.set_title("Control pool (the generator's start state)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    for x, c, t in zip(xs, ctrl, tot):
        ax.text(x, t * 1.15, f"{100*c/t:.0f}% ctrl", ha="center", fontsize=7.5, color=GREY)

    finish(fig, os.path.join(figdir, "fig4_data_budget.png"),
           "Source: artifacts/dataset_profiles.json + condition_stats.jsonl.")


# ================================================================ figure 5
def fig_signature_structure(artdir, figdir):
    """PCA of perturbation signatures: does the target-state space have enough
    structure for a generator to navigate, or is it one blob plus outliers?"""
    npzdir = os.path.join(artdir, "npz")
    if not os.path.isdir(npzdir):
        return
    # exclude *_oracle_signatures.npz (inverse_gap.py) -- same suffix, different keys
    files = sorted(
        f for f in os.listdir(npzdir)
        if f.endswith("_signatures.npz") and not f.endswith("_oracle_signatures.npz")
    )
    if not files:
        return
    order = {d: i for i, d in enumerate(DS_ORDER)}
    files.sort(key=lambda f: order.get(f.replace("_signatures.npz", ""), 99))
    n = len(files)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.7 * nrow), squeeze=False)
    cmap = colour_for([f.replace("_signatures.npz", "") for f in files])

    for k, f in enumerate(files):
        ax = axes[k // ncol][k % ncol]
        d = f.replace("_signatures.npz", "")
        z = np.load(os.path.join(npzdir, f), allow_pickle=False)
        S = z["signatures"].astype(np.float64)
        conds = z["conditions"]
        if S.shape[0] < 3:
            ax.set_visible(False)
            continue
        Sc = S - S.mean(0, keepdims=True)
        U, sv, Vt = np.linalg.svd(Sc, full_matrices=False)
        pc = U[:, :2] * sv[:2]
        var = sv**2 / np.sum(sv**2)
        mag = np.linalg.norm(S, axis=1)
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=mag, s=22, cmap="viridis", lw=0.3,
                        edgecolor="white")
        # how many PCs to reach 90% of signature variance -> effective dimension
        k90 = int(np.searchsorted(np.cumsum(var), 0.90) + 1)
        ax.set_title(f"{d}\nK={S.shape[0]} conditions · {k90} PCs for 90% var",
                     fontsize=9.5, color=cmap[d])
        ax.set_xlabel(f"PC1 ({100*var[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({100*var[1]:.0f}%)")
        cb = fig.colorbar(sc, ax=ax, fraction=0.045)
        cb.set_label("‖signature‖", fontsize=8)
        cb.ax.tick_params(labelsize=7.5)
        # label the few strongest perturbations
        for i in np.argsort(-mag)[:4]:
            ax.annotate(str(conds[i])[:16], (pc[i, 0], pc[i, 1]), fontsize=6.5,
                        color=INK, xytext=(3, 3), textcoords="offset points")
    for k in range(len(files), nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)

    fig.suptitle("Target-state space the generator must navigate (PCA of Δ-from-control signatures)",
                 fontsize=11, y=1.005)
    finish(fig, os.path.join(figdir, "fig5_signature_space.png"),
           "Source: artifacts/npz/*_signatures.npz.")


# ================================================================ figure 6
def fig_oracle(artdir, figdir):
    """Reward-oracle quality. If the oracle is barely better than a constant
    prediction, an RL reward built on it is mostly measuring the constant."""
    path = os.path.join(artdir, "oracle_probe.json")
    if not os.path.exists(path):
        print("  (no oracle_probe.json; skipping fig6)")
        return
    with open(path) as f:
        O = json.load(f)
    rows = O.get("per_condition", [])
    base = O.get("baselines", {})
    if not rows:
        print("  (oracle_probe.json has no per_condition rows; skipping fig6)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    series = [
        ("GEARS", "gears_pearson_de", PALETTE[0]),
        ("predict control mean", "ctrl_pearson_de", GREY),
        ("predict mean perturbed cell", "meanpert_pearson_de", PALETTE[3]),
    ]
    present = []
    for lab, key, col in series:
        v = np.array([r[key] for r in rows if key in r and np.isfinite(r.get(key, np.nan))])
        if v.size < 2:
            continue
        xs = np.sort(v)
        ax.plot(xs, 100 * np.arange(1, xs.size + 1) / xs.size, lw=2.0, color=col, label=lab)
        present.append((lab, float(np.median(v))))
    ax.set_xlabel("Pearson r on top-20 DE genes (per held-out perturbation)")
    ax.set_ylabel("% of perturbations ≤ x")
    ax.set_title("Reward-oracle quality vs trivial baselines")
    ax.legend(loc="upper left")

    ax = axes[1]
    labs = [p[0] for p in present]
    vals = [p[1] for p in present]
    cols = [dict(series and {l: c for l, _, c in series}).get(l, GREY) for l in labs]
    bars = ax.barh(range(len(labs)), vals, color=cols, height=0.55)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("median Pearson r (top-20 DE)")
    ax.set_title("Headroom the oracle actually has")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(1.0, max(vals) * 1.25) if vals else 1.0)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8.5)

    note = (f"Source: artifacts/oracle_probe.json · checkpoint: {O.get('checkpoint','?')} · "
            f"{len(rows)} held-out perturbations · split: {O.get('split','?')}")
    finish(fig, os.path.join(figdir, "fig6_oracle_quality.png"), note)


# ================================================================ figure 9
def fig_trivial_rule(artdir, figdir):
    """How much of 'identify the perturbation' is reading the knocked-down gene?

    CDN (Wu et al., ICML 2025) call these 'trivial' perturbations and exclude them
    from their headline table, because for most perturbations the target simply IS
    the most down-regulated gene. If that holds here, high retrieval accuracy on
    perturb-seq is barcode-reading, and will not transfer to a design task whose
    target is a desired phenotype carrying no such fingerprint."""
    path = os.path.join(artdir, "trivial_rule.json")
    if not os.path.exists(path):
        print("  (no trivial_rule.json; skipping fig9)")
        return
    with open(path) as f:
        T = json.load(f)
    rows = [d for d in T.get("datasets", []) if "frac_trivial_rank1" in d]
    if not rows:
        print("  (trivial_rule.json has no usable rows; skipping fig9)")
        return
    order = {d: i for i, d in enumerate(DS_ORDER)}
    rows.sort(key=lambda d: order.get(d["dataset"], 99))
    names = [d["dataset"] for d in rows]
    cmap = colour_for(names)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.1))
    fig.subplots_adjust(wspace=0.62)

    ax = axes[0]
    xs = np.arange(len(rows))
    w = 0.26
    r1 = [100 * d["frac_trivial_rank1"] for d in rows]
    r10 = [100 * d["frac_top10"] for d in rows]
    r100 = [100 * d["frac_top100"] for d in rows]
    ax.bar(xs - w, r1, w, color=PALETTE[1], label="rank 1 (largest |Δ|)")
    ax.bar(xs, r10, w, color=PALETTE[0], label="top 10 by |Δ|")
    ax.bar(xs + w, r100, w, color=GREY, label="top 100 by |Δ|")
    ax.set_xticks(xs)
    ax.set_xticklabels([short(n) for n in names], fontsize=8)
    ax.set_ylabel("% of conditions")
    ax.set_title("Where does the perturbed gene's own |Δ| rank?\n(the 'trivial' rule)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 105)

    ax = axes[1]
    gd = [100 * d["frac_own_gene_goes_down"] for d in rows]
    bars = ax.barh(xs, gd, color=[cmap[n] for n in names], height=0.6)
    ax.set_yticks(xs)
    ax.set_yticklabels([short(n).replace("\n", " ") for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(50, color=GREY, ls=":", lw=1.1)
    ax.set_xlabel("% of conditions where the perturbed gene goes DOWN")
    ax.set_title("Direction of the fingerprint\n(low = CRISPRa, high = CRISPRi/KO)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 100)
    # label INSIDE the bar: at 100% an outside label collides with the next panel
    for b, v in zip(bars, gd):
        inside = v > 25
        ax.text(b.get_width() - 3 if inside else b.get_width() + 3,
                b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center",
                ha="right" if inside else "left", fontsize=8.5,
                color="white" if inside else INK,
                weight="700" if inside else "400")

    ax = axes[2]
    mr = [d["median_own_gene_abs_rank"] for d in rows]
    G = [d["n_genes"] for d in rows]
    bars = ax.barh(xs, mr, color=[cmap[n] for n in names], height=0.6)
    ax.set_yticks(xs)
    ax.set_yticklabels([short(n).replace("\n", " ") for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("median |Δ| rank of the perturbed gene")
    ax.set_title("Lower = more barcode-like")
    ax.grid(axis="y", visible=False)
    for b, v, g in zip(bars, mr, G):
        ax.text(b.get_width() * 1.15, b.get_y() + b.get_height() / 2,
                f"{v:.0f} / {g:,}", va="center", fontsize=8)
    ax.set_xlim(right=max(mr) * 6)

    finish(fig, os.path.join(figdir, "fig9_trivial_rule.png"),
           "Source: artifacts/trivial_rule.json. Ranks genes by |Δ-from-control|, so rank 1 is the "
           "single most-changed gene. Direction-agnostic on purpose: Norman 2019 is CRISPRa (the "
           "target goes UP) while Adamson and Replogle are CRISPRi (it goes DOWN), and a down-only "
           "ranking inverts the conclusion for Norman. Replogle's bars cover only the 38-42% of "
           "targets that are measured at all. Dixit is CRISPR cutting: the transcript often "
           "survives, which is why it leaves no usable fingerprint.")


# ================================================================ figure 8
def fig_inverse_ladder(artdir, figdir):
    """ceiling / oracle / non-conditional / chance on one comparable task.
    The ceiling-to-oracle distance is headroom lost to the reward model."""
    path = os.path.join(artdir, "inverse_gap.json")
    if not os.path.exists(path):
        print("  (no inverse_gap.json; skipping fig8)")
        return
    with open(path) as f:
        G = json.load(f)
    lad = G.get("ladder") or {}
    if not lad.get("ceiling"):
        print(f"  (inverse_gap.json status={G.get('status')}; no ladder; skipping fig8)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    fig.subplots_adjust(wspace=0.32)
    styles = [
        ("ceiling", "empirical centroid (ceiling)", PALETTE[2], "-o"),
        ("oracle", "GEARS forward-model search", PALETTE[0], "-s"),
        ("mean", "non-conditional (avg effect)", PALETTE[3], "-^"),
    ]

    ax = axes[0]
    K = G.get("n_candidates", 0)
    for key, lab, col, mk in styles:
        rows = sorted(lad.get(key, []), key=lambda r: r["n_cells_pooled"])
        if not rows:
            continue
        ax.plot([r["n_cells_pooled"] for r in rows], [100 * r["top1_acc"] for r in rows],
                mk, color=col, lw=2.0, ms=5, label=lab)
    if K:
        ax.axhline(100.0 / K, color=GREY, ls=":", lw=1.1)
        ax.text(1.05, 100.0 / K * 1.25, f"chance (1/{K})", fontsize=7.5, color=GREY)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("cells pooled into the target state")
    ax.set_ylabel("top-1 accuracy  (%)")
    ax.set_title(f"Inverse-design ladder on Norman\nK={K} candidate perturbations")
    ax.legend(loc="upper left")

    ax = axes[1]
    h = (G.get("headline") or {}).get("per_pool_size") or {}
    ns = sorted(h, key=int)
    if ns:
        xs = np.arange(len(ns))
        ce = [100 * h[n]["ceiling_top1"] for n in ns]
        oc = [100 * h[n]["oracle_top1"] for n in ns]
        w = 0.38
        ax.bar(xs - w / 2, ce, w, color=PALETTE[2], label="ceiling")
        ax.bar(xs + w / 2, oc, w, color=PALETTE[0], label="oracle-driven")
        for x, c, o in zip(xs, ce, oc):
            ax.annotate("", xy=(x + w / 2, o), xytext=(x - w / 2, c),
                        arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=1.4))
            ax.text(x, max(c, o) + 2.5, f"−{c-o:.0f} pts", ha="center",
                    fontsize=8, color=PALETTE[1], weight="600")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{n} cell" if n == "1" else f"{n} cells" for n in ns],
                           fontsize=8.5)
        ax.set_ylabel("top-1 accuracy  (%)")
        ax.set_title("Headroom lost to the reward model\n(ceiling − oracle-driven)")
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(ce) * 1.25)

    fid = G.get("oracle_fidelity") or {}
    note = ("Source: artifacts/inverse_gap.json. "
            f"oracle fidelity: median cos(GEARS Δ, measured Δ)="
            f"{fid.get('median_cos_gears_vs_measured', float('nan')):.3f} vs "
            f"non-conditional {fid.get('median_cos_meanpert_vs_measured', float('nan')):.3f}; "
            f"GEARS wins on {100*fid.get('frac_conditions_gears_beats_meanpert', float('nan')):.0f}% "
            "of conditions. " + G.get("caveat", ""))
    finish(fig, os.path.join(figdir, "fig8_inverse_ladder.png"), note)


# ================================================================ figure 7
# Published SOTA on the genetic inverse-design task, read off PDGrapher's own
# Extended Data Fig. 2 / Fig. E3 axis tick labels (vector PDF: the tick LABELS are
# text, the bar heights are paths — so these are the axis ceilings that bound the
# plotted bars, not the bar values). Source: bioRxiv 2024.01.03.573985v5
# extended-data PDF, panels for Genetic-PPI-Breast-MCF7 vs Chemical-PPI-Breast-MDAMB231.
PUBLISHED = [
    # (metric, genetic axis max, chemical axis max)
    ("recall@1", 0.008, 0.07),
    ("recall@100", 0.05, 0.50),
    ("nDCG", 0.200, 0.40),
]
# PDGrapher ranks all nodes of a BioGRID PPI: 10,716 candidate genes.
PUBLISHED_K = 10716


# CDN / Wu et al., ICML 2025 (arXiv 2410.03380v4) Table 1 — normalized rank of the
# true target on five Perturb-seq datasets. 1.0 = target ranked first, 0.5 = chance.
# Transcribed from the paper's table text, "seen cell line" block unless noted.
CDN_DATASETS = ["K562 gw", "K562 es", "RPE1", "HepG2", "Jurkat"]
CDN_TABLE1 = [
    # (method, ranks, kind)  kind: dl = published deep-learning baseline
    ("Linear",                 [0.51, 0.48, 0.49, 0.51, 0.50], "dl"),
    ("MLP",                    [0.45, 0.49, 0.53, 0.48, 0.45], "dl"),
    ("GenePt",                 [0.52, 0.42, 0.41, 0.31, 0.51], "dl"),
    ("GEARS",                  [0.56, 0.45, 0.49, 0.54, 0.50], "dl"),
    ("PDGrapher",              [0.54, 0.49, 0.48, 0.54, 0.52], "dl"),
    ("Wilcoxon test (DGE)*",   [0.82, 0.89, 0.79, 0.86, 0.84], "stat"),
    ("CDN",                    [0.92, 0.95, 0.92, 0.94, 0.98], "cdn"),
]
CDN_KIND_COLOUR = {"dl": PALETTE[0], "stat": PALETTE[3], "cdn": PALETTE[2]}


def fig_published_gap(retr, artdir, figdir):
    """Cross-method check: is the weak published performance specific to one paper,
    or does every baseline sit in the same band?"""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    fig.subplots_adjust(wspace=0.45)

    # ---- left: every method, five datasets, one metric
    ax = axes[0]
    ys = np.arange(len(CDN_TABLE1))
    for i, (name, ranks, kind) in enumerate(CDN_TABLE1):
        col = CDN_KIND_COLOUR[kind]
        ax.plot(ranks, [i] * len(ranks), "o", color=col, ms=7, alpha=0.8,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.plot([min(ranks), max(ranks)], [i, i], "-", color=col, lw=2.2, alpha=0.35)
    ax.axvline(0.5, color=GREY, ls="--", lw=1.3)
    ax.text(0.5, len(CDN_TABLE1) - 0.35, " chance", fontsize=8, color=GREY, va="top")
    ax.set_yticks(ys)
    ax.set_yticklabels([m[0] for m in CDN_TABLE1], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.25, 1.02)
    ax.set_xlabel("normalized rank of the true target  (1.0 = ranked first)")
    ax.set_title("Every published deep-learning baseline sits at chance\n"
                 "5 Perturb-seq datasets · CDN (ICML 2025) Table 1")
    ax.grid(axis="y", visible=False)
    # band annotation
    ax.annotate("", xy=(0.31, 4.62), xytext=(0.57, 4.62),
                arrowprops=dict(arrowstyle="<->", color=PALETTE[0], lw=1.3))
    ax.text(0.44, 4.95, "0.31 – 0.57", ha="center", fontsize=8.5,
            color=PALETTE[0], weight="600")

    # ---- right: our own ladder, same question, measured here
    ax = axes[1]
    names = order_datasets({r["dataset"] for r in retr})
    rows = []
    for d in names:
        rs = [r for r in retr if r["dataset"] == d]
        if not rs:
            continue
        best = max(rs, key=lambda r: r["top1_acc"])
        rows.append((short(d).replace("\n", " "), 100 * best["top1_acc"],
                     best["n_candidate_conditions"], best["n_cells_observed"]))
    cols = [PALETTE[i % len(PALETTE)] for i in range(len(rows))]
    ys = np.arange(len(rows))
    bars = ax.barh(ys, [r[1] for r in rows], color=cols, height=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0]}  (K={r[2]:,})" for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("top-1 accuracy  (%)")
    ax.set_title("…yet a parameter-free centroid reaches 73–95%\n"
                 "on pooled perturb-seq — measured here")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 118)
    for b, r in zip(bars, rows):
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2,
                f"{r[1]:.1f}% @{r[3]} cells", va="center", fontsize=8, color=INK)

    finish(fig, os.path.join(figdir, "fig7_published_gap.png"),
           "Left: *DGE is reported in CDN's unseen-cell-line block (it needs no training); every other "
           "row is the seen-cell-line block. CDN ranks within a ≤1,000-gene shortlist that already "
           "contains the answer, and this table is their non-trivial subset. Right: not like-for-like "
           "with the left panel — different data, candidate sets and metric. The point is that the "
           "setting, not the model class, is what moves performance.")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--figures", required=True)
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)

    prof_path = os.path.join(args.artifacts, "dataset_profiles.json")
    if not os.path.exists(prof_path):
        print(f"GATE: {prof_path} missing; run jobs/20_profile.sbatch first.")
        return 0
    with open(prof_path) as f:
        meta = json.load(f)
    profiles = [p for p in meta.get("profiles", []) if "skip_reason" not in p]
    cond = jsonl(os.path.join(args.artifacts, "condition_stats.jsonl"))
    retr = jsonl(os.path.join(args.artifacts, "retrieval.jsonl"))
    print(f"loaded {len(profiles)} profiles, {len(cond)} condition rows, {len(retr)} retrieval rows")

    for name, fn in [
        ("fig1", lambda: fig_action_space(profiles, args.figures)),
        ("fig2", lambda: fig_identifiability(retr, profiles, args.figures)),
        ("fig3", lambda: fig_effect_noise(cond, args.figures)),
        ("fig4", lambda: fig_data_per_condition(profiles, cond, args.figures)),
        ("fig5", lambda: fig_signature_structure(args.artifacts, args.figures)),
        ("fig6", lambda: fig_oracle(args.artifacts, args.figures)),
        ("fig7", lambda: fig_published_gap(retr, args.artifacts, args.figures)),
        ("fig8", lambda: fig_inverse_ladder(args.artifacts, args.figures)),
        ("fig9", lambda: fig_trivial_rule(args.artifacts, args.figures)),
    ]:
        try:
            if (name in ("fig2", "fig7") and not retr) or (name in ("fig3", "fig4") and not cond):
                print(f"  (no data for {name}; skipped)")
                continue
            fn()
        except Exception as e:
            import traceback
            print(f"  ERROR in {name}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("figures in", args.figures)
    for f in sorted(os.listdir(args.figures)):
        print("  ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
