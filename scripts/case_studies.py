#!/usr/bin/env python
"""Case-study figures: individual perturbations, not aggregate metrics.

Three questions, each shown on named examples rather than curves:

  fig10  Is identification just reading the knocked-down (or activated) gene?
         -> for one representative perturbation per dataset, the 15 most-changed
            genes, with the perturbed gene itself marked.

  fig11  Does the GEARS oracle predict the perturbation, or the average effect?
         -> for named Norman perturbations, measured Δ vs GEARS Δ vs the global
            mean effect, over that condition's own top differential genes.

  fig12  What does the oracle actually rank above the right answer?
         -> for named target states, the candidates GEARS scores highest, with
            the true perturbation marked wherever it lands.

Selection criteria are printed on every panel, and representative cases are
chosen by MEDIAN rank, not by being flattering.

CPU only. Reads artifacts/npz/*.npz; writes figures/fig10..12.
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

PALETTE = ["#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0891b2"]
GREY = "#6b7280"
GRID = "#e5e7eb"
INK = "#111827"
ACCENT = "#db2777"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.facecolor": "white", "font.size": 10, "axes.titlesize": 10.5,
    "axes.titleweight": "600", "axes.labelsize": 9, "axes.edgecolor": "#9ca3af",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.fontsize": 8.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

SHORT = {"norman": "Norman (CRISPRa)", "adamson": "Adamson (CRISPRi)",
         "dixit": "Dixit (CRISPR cutting)",
         "replogle_k562_essential": "Replogle K562 (CRISPRi)",
         "replogle_rpe1_essential": "Replogle RPE1 (CRISPRi)"}


def finish(fig, path, note=None):
    if note:
        import textwrap
        w = max(110, int(fig.get_size_inches()[0] * 15))
        fig.text(0.005, -0.015, "\n".join(textwrap.wrap(note, w)),
                 fontsize=7.5, color=GREY, va="top")
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path, flush=True)


def targets_of(cond):
    return [p for p in str(cond).split("+") if p != "ctrl"]


def load_sig(npzdir, ds):
    z = np.load(os.path.join(npzdir, f"{ds}_signatures.npz"), allow_pickle=False)
    return ([str(c) for c in z["conditions"]], z["signatures"].astype(np.float64),
            [str(g) for g in z["gene_names"]])


# ============================================================ fig 10
def fig_barcode(npzdir, figdir, datasets):
    panels = []
    for ds in datasets:
        try:
            conds, sig, genes = load_sig(npzdir, ds)
        except FileNotFoundError:
            continue
        gidx = {g: i for i, g in enumerate(genes)}
        recs = []
        for i, c in enumerate(conds):
            present = [t for t in targets_of(c) if t in gidx]
            if not present:
                continue
            absr = np.argsort(np.argsort(-np.abs(sig[i]))) + 1
            j = min(present, key=lambda t: absr[gidx[t]])
            recs.append((i, c, j, int(absr[gidx[j]])))
        if not recs:
            continue
        ranks = np.array([r[3] for r in recs])
        # representative = the case closest to this dataset's MEDIAN rank
        pick = recs[int(np.argmin(np.abs(ranks - np.median(ranks))))]
        panels.append((ds, pick, sig, genes, float(np.median(ranks))))

    if not panels:
        return
    n = len(panels)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 4.5 * nrow), squeeze=False)
    fig.subplots_adjust(hspace=0.62, wspace=0.42)

    for k, (ds, (i, cond, tgt, trank), sig, genes, medr) in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        d = sig[i]
        top = np.argsort(-np.abs(d))[:15][::-1]
        vals = d[top]
        names = [genes[t] for t in top]
        cols = [ACCENT if genes[t] == tgt else GREY for t in top]
        ax.barh(np.arange(len(top)), vals, color=cols, height=0.72)
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(names, fontsize=7.5)
        for lab, t in zip(ax.get_yticklabels(), top):
            if genes[t] == tgt:
                lab.set_color(ACCENT)
                lab.set_fontweight("700")
        ax.axvline(0, color="#9ca3af", lw=0.8)
        ax.set_xlabel("Δ expression vs control")
        hit = trank <= 15
        # selection rule goes in the title, not over the bars
        ax.set_title(f"{SHORT.get(ds, ds)} · {cond}\n"
                     f"representative case — dataset median rank {medr:.0f}",
                     fontsize=9.5)
        tag = (f"target {tgt} is rank {trank} of {len(genes):,}"
               if hit else
               f"target {tgt} NOT in top 15 — rank {trank} of {len(genes):,}")
        # anchor the callout under the axes so it can never sit on a bar
        ax.text(0.5, -0.195, tag, transform=ax.transAxes, ha="center", fontsize=8.5,
                color=ACCENT if hit else INK, weight="700" if hit else "500",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=ACCENT if hit else GREY, lw=0.9, alpha=0.98))
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.18)

    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)

    fig.suptitle("Is \"identify the perturbation\" just reading the perturbed gene?",
                 fontsize=12, y=1.005)
    finish(fig, os.path.join(figdir, "fig10_barcode_cases.png"),
           "Each panel is ONE perturbation, chosen as the case closest to its dataset's median rank — "
           "not cherry-picked. Bars are the 15 genes with the largest |Δ| from control; the perturbed "
           "gene itself is pink where it appears. Norman is CRISPRa so its target goes UP; the others "
           "are CRISPRi/cutting so theirs go DOWN. Where the target sits at the top, naming the "
           "perturbation is barcode-reading, and that skill will not transfer to designing for a "
           "phenotype that carries no such mark.")


# ============================================================ fig 11
def fig_oracle_cases(npzdir, figdir, label="norman"):
    p = os.path.join(npzdir, f"{label}_oracle_signatures.npz")
    if not os.path.exists(p):
        print("  (no oracle signatures; skipping fig11)")
        return
    z = np.load(p, allow_pickle=False)
    conds = [str(c) for c in z["conditions"]]
    orc, emp = z["oracle_sig"].astype(np.float64), z["empirical_sig"].astype(np.float64)
    mean_d = z["mean_pert_delta"].astype(np.float64)
    # Gene names must come from THIS archive: the dataloader's gene space differs
    # from the profiling release's, so borrowing names across them would mislabel.
    if "gene_names" in z and len(z["gene_names"]) == emp.shape[1]:
        genes = [str(g) for g in z["gene_names"]]
    else:
        print("  WARNING: oracle npz has no matching gene_names; rerun jobs/23 "
              "to regenerate it. Falling back to positional labels.")
        genes = [f"gene[{i}]" for i in range(emp.shape[1])]

    def u(v):
        return v / (np.linalg.norm(v) + 1e-9)

    cg = np.array([float(u(orc[i]) @ u(emp[i])) for i in range(len(conds))])
    cm = np.array([float(u(mean_d) @ u(emp[i])) for i in range(len(conds))])
    eff = np.linalg.norm(emp, axis=1)

    # Four named cases spanning the range, each labelled with why it was chosen.
    order_eff = np.argsort(-eff)
    picks = [
        (int(order_eff[0]), "strongest measured effect"),
        (int(np.argmax(cg - cm)), "GEARS's best win over the mean"),
        (int(np.argsort(np.abs((cg - cm) - np.median(cg - cm)))[0]), "median case"),
        (int(np.argmin(cg - cm)), "GEARS's worst loss to the mean"),
    ]
    seen, uniq = set(), []
    for i, why in picks:
        if i not in seen:
            seen.add(i)
            uniq.append((i, why))

    fig, axes = plt.subplots(1, len(uniq), figsize=(4.3 * len(uniq), 4.0), squeeze=False)
    for k, (i, why) in enumerate(uniq):
        ax = axes[0][k]
        top = np.argsort(-np.abs(emp[i]))[:12][::-1]
        ys = np.arange(len(top))
        h = 0.26
        ax.barh(ys + h, emp[i][top], h, color=PALETTE[2], label="measured")
        ax.barh(ys, orc[i][top], h, color=PALETTE[0], label="GEARS predicted")
        ax.barh(ys - h, mean_d[top], h, color=PALETTE[3], label="average effect")
        ax.set_yticks(ys)
        ax.set_yticklabels([genes[t] for t in top], fontsize=7)
        ax.axvline(0, color="#9ca3af", lw=0.8)
        ax.set_xlabel("Δ expression vs control")
        ax.set_title(f"{conds[i]}\n{why}", fontsize=9.5)
        ax.text(0.98, 0.03,
                f"cos(GEARS, measured) = {cg[i]:.2f}\ncos(average, measured) = {cm[i]:.2f}",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=0.7, alpha=0.95))
        ax.grid(axis="y", visible=False)
        if k == 0:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Does the reward oracle predict the perturbation, or the average effect?",
                 fontsize=12, y=1.01)
    # figures are read on their own, so the aggregate context is computed here
    # rather than hardcoded -- a stale hardcoded count is how a figure starts lying.
    win = float(np.mean(cg > cm))
    finish(fig, os.path.join(figdir, "fig11_oracle_cases.png"),
           "Named Norman perturbations (canonical names: 'X+ctrl' and 'ctrl+X' are the same "
           "perturbation and are merged). Bars are that condition's 12 largest measured |Δ| genes. "
           "Green is what was measured, blue is what GEARS predicts, orange is the average effect over "
           "ALL perturbations — a predictor that ignores which perturbation was applied. Where blue "
           "tracks orange rather than green, the oracle is reporting the average, and a reward built on "
           f"it cannot tell perturbations apart. Across all {len(conds)} canonical conditions GEARS "
           f"beats the average on only {100*win:.0f}%, median cos {np.median(cg):.2f} vs "
           f"{np.median(cm):.2f} — on a checkpoint trained on every one of them.")


# ============================================================ fig 12
def fig_retrieval_cases(npzdir, figdir, label="norman", n_cases=3, topn=8):
    p = os.path.join(npzdir, f"{label}_oracle_signatures.npz")
    if not os.path.exists(p):
        print("  (no oracle signatures; skipping fig12)")
        return
    z = np.load(p, allow_pickle=False)
    conds = np.array([str(c) for c in z["conditions"]])
    orc, emp = z["oracle_sig"].astype(np.float64), z["empirical_sig"].astype(np.float64)
    U_orc = orc / (np.linalg.norm(orc, axis=1, keepdims=True) + 1e-9)
    U_emp = emp / (np.linalg.norm(emp, axis=1, keepdims=True) + 1e-9)

    # rank every candidate by how well its GEARS-PREDICTED signature matches the
    # measured target state; where does the true perturbation land?
    S = U_emp @ U_orc.T                       # (target, candidate)
    true_rank = np.array([
        int(np.flatnonzero(np.argsort(-S[i]) == i)[0]) + 1 for i in range(len(conds))
    ])
    eff = np.linalg.norm(emp, axis=1)
    # take strong-effect targets (a designer would ask for these), median-ranked
    strong = np.argsort(-eff)[: max(20, len(conds) // 4)]
    picks = strong[np.argsort(np.abs(true_rank[strong] - np.median(true_rank[strong])))][:n_cases]

    fig, axes = plt.subplots(1, len(picks), figsize=(4.5 * len(picks), 4.0), squeeze=False)
    for k, i in enumerate(picks):
        ax = axes[0][k]
        order = np.argsort(-S[i])[:topn]
        rows = list(order)[::-1]
        vals = [S[i][j] for j in rows]
        labs = [conds[j] for j in rows]
        cols = [PALETTE[2] if j == i else GREY for j in rows]
        ax.barh(np.arange(len(rows)), vals, color=cols, height=0.7)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(labs, fontsize=7.5)
        for lab, j in zip(ax.get_yticklabels(), rows):
            if j == i:
                lab.set_color(PALETTE[2])
                lab.set_fontweight("700")
        ax.set_xlabel("GEARS score against the target state (cosine)")
        ax.set_title(f"target state: {conds[i]}", fontsize=9.5)
        if i not in order:
            ax.text(0.98, 0.04,
                    f"correct answer {conds[i]}\nranked {true_rank[i]} of {len(conds)}"
                    f" — off this list",
                    transform=ax.transAxes, ha="right", fontsize=8, color=ACCENT,
                    weight="700",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT, lw=0.9))
        else:
            ax.text(0.98, 0.04, f"correct answer at rank {true_rank[i]}",
                    transform=ax.transAxes, ha="right", fontsize=8, color=PALETTE[2],
                    weight="700",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE[2], lw=0.9))
        ax.grid(axis="y", visible=False)
        ax.set_xlim(min(vals) - 0.04, max(vals) + 0.06)

    fig.suptitle("What the oracle ranks above the right answer",
                 fontsize=12, y=1.01)
    finish(fig, os.path.join(figdir, "fig12_retrieval_cases.png"),
           "Each panel asks GEARS for the perturbation that best reproduces one measured target state, "
           f"scoring all {len(conds)} canonical candidates by its own predicted signature ('X+ctrl' "
           "and 'ctrl+X' are the same perturbation and are merged, so a twin can no longer be counted "
           "as a miss). Grey bars are what it returns; green is the correct answer. Cases are "
           "strong-effect targets at the median rank for that group — the typical outcome, not the "
           "worst. Note how little separates the top scores: the ordering is close to noise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--figures", required=True)
    ap.add_argument("--datasets", nargs="*",
                    default=["norman", "replogle_k562_essential", "adamson", "dixit",
                             "replogle_rpe1_essential"])
    args = ap.parse_args()
    npzdir = os.path.join(args.artifacts, "npz")
    os.makedirs(args.figures, exist_ok=True)

    for name, fn in [
        ("fig10", lambda: fig_barcode(npzdir, args.figures, args.datasets)),
        ("fig11", lambda: fig_oracle_cases(npzdir, args.figures)),
        ("fig12", lambda: fig_retrieval_cases(npzdir, args.figures)),
    ]:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ERROR in {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
