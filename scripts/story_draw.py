"""Draw the gap / improvement case-study figures (fig18-fig21) from artifacts/story/.

Reads only what story_cases.py `compute` wrote; no model, no h5ad, no recomputation.

Journal style: 7-inch double-column width, a Times-like serif (STIX, bundled with
matplotlib, so every node renders identically), 7-8 pt text, bold panel letters,
greyscale with a red/blue accent and viridis for continuous values, gene symbols in
italics, no titles on the figure -- the message lives in the caption (GAPS.md).
Each figure is written as vector PDF (for the paper) and 300-dpi PNG (for markdown).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

# ------------------------------------------------------------------ palette
RED = "#B2182B"        # ColorBrewer RdBu: improvement / the highlighted case
BLUE = "#2166AC"       # ColorBrewer RdBu: deterioration
INK = "#000000"
DARK = "0.25"
MID = "0.55"
LIGHT = "0.78"
FAINT = "0.88"

SHORT = {"replogle_k562_essential": "K562", "replogle_rpe1_essential": "RPE1"}
W2 = 7.0               # double-column width, inches


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.titlepad": 4, "axes.labelpad": 2,
        "lines.linewidth": 0.9, "patch.linewidth": 0.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})


def _gi(name):
    """A gene symbol in italics (journal convention for human genes)."""
    name = str(name)
    return rf"$\mathit{{{name}}}$" if re.fullmatch(r"[A-Za-z0-9]+", name) else name


def _panel(ax, letter, x=-0.09, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="left", va="bottom")


def _save(fig, fig_dir, stem):
    import matplotlib.pyplot as plt
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        print(f"[story] wrote {p}")
    plt.close(fig)


def _rank(r):
    r = float(r)
    return f"{int(round(r))}" if abs(r - round(r)) < 1e-9 else f"{r:.1f}"


# ================================================================== fig18: the gap
def fig_gap(S, J, ds, fig_dir):
    """Measured vs predicted-from-GO, one held-out query against every candidate."""
    import matplotlib.pyplot as plt
    rm, rp = S["gap_rank_meas"], S["gap_rank_pred"]
    cands = S["cands"]
    ok = np.flatnonzero(rm <= 1)                       # the measurement identifies it
    rp_ok = rp[ok]
    picks = []
    for q in (25, 50, 75):
        v = np.percentile(rp_ok, q)
        i = int(ok[np.argmin(np.abs(rp_ok - v) + 1e-9 * np.arange(ok.size))])
        picks.append((i, q))
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(3, 1, figsize=(W2, 4.1), gridspec_kw={"hspace": 0.75})
    for r, (ax, (i, q)) in enumerate(zip(axes, picks)):
        sm = S["gap_S_meas"][i].astype(np.float32)
        sp = S["gap_S_pred"][i].astype(np.float32)
        lo, hi = min(sm.min(), sp.min()), max(sm.max(), sp.max())
        pad = 0.04 * (hi - lo)
        others = np.arange(sm.size) != i
        for yc, s, rank in ((1, sm, rm[i]), (0, sp, rp[i])):
            jit = rng.uniform(-0.28, 0.28, size=s.size)
            ax.scatter(s[others], yc + jit[others], s=1.6, color=LIGHT, linewidths=0, rasterized=True,
                       zorder=2)
            ax.scatter([s[i]], [yc], s=26, color=RED, edgecolors=INK, linewidths=0.6, zorder=4)
            right = (s[i] - lo) / (hi - lo) > 0.82
            ax.annotate(f"rank {_rank(rank)}", (s[i], yc), xytext=(-6 if right else 6, 0),
                        textcoords="offset points", ha="right" if right else "left", va="center",
                        fontsize=7, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.55, 1.55)
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["Measured", "GO-kNN predicted"])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{_gi(cands[i])}  ({q}th percentile of predicted rank; "
                     f"cos(prediction, measured) = {S['gap_fidelity'][i]:.2f})", loc="left")
        _panel(ax, "abc"[r], x=-0.155, y=1.0)
        if r == len(picks) - 1:
            ax.set_xlabel("Cosine similarity between the held-out target and each candidate signature")
    _save(fig, fig_dir, "fig18_gap_crowd")
    return [str(cands[i]) for i, _ in picks]


# ================================================================== fig19: rank flow
def fig_flow(SS, JJ, fig_dir, lead=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator
    dss = list(SS)
    fig, axes = plt.subplots(1, len(dss), figsize=(W2, 3.6), gridspec_kw={"wspace": 0.42})
    axes = np.atleast_1d(axes)
    stats = {}
    for k, (ax, ds) in enumerate(zip(axes, dss)):
        S = SS[ds]
        cands, rep = S["cands"], S["flow_rep"]
        a, b = np.median(S["flow_R_rand"], 0), np.median(S["flow_R_rl"], 0)
        n = a.size
        ratio = a / b
        cls = np.where(ratio >= 2, 1, np.where(ratio <= 0.5, -1, 0))
        stats[SHORT[ds]] = {"up2x": int((cls > 0).sum()), "down2x": int((cls < 0).sum()), "n": int(n),
                            "rank1_random": int((a <= 1).sum()), "rank1_rl": int((b <= 1).sum())}
        X = np.array([0.0, 1.0])
        for c, col, lw, al, z in ((0, LIGHT, 0.45, 0.9, 1), (-1, BLUE, 0.7, 0.85, 2), (1, RED, 0.7, 0.85, 3)):
            for j in np.flatnonzero(cls == c):
                ax.plot(X, [a[j], b[j]], color=col, lw=lw, alpha=al, zorder=z, solid_capstyle="round")
        an = (JJ.get(ds, {}).get("anatomy") or []) if ds == lead else []
        if an:                                       # the gene fig20 dissects, followed through
            j = int(an[0]["rep_pos"])
            ax.plot(X, [a[j], b[j]], color=INK, lw=1.5, zorder=6)
            ax.scatter(X, [a[j], b[j]], s=12, color=INK, zorder=7, linewidths=0)
        ax.set_yscale("log")
        ax.set_ylim(n * 1.1, 0.85)
        ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200) if t <= n]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([str(t) for t in ticks])
        ax.set_xlim(-0.62, 1.32)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Random atlas", "REINFORCE atlas"])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=4)
        if k == 0:
            ax.set_ylabel("Rank of the true gene among report genes")
        st = stats[SHORT[ds]]
        ax.set_title(f"{SHORT[ds]}  ($n$ = {n} held-out genes)", loc="left")
        # genes at rank 1 under each atlas, written on the rank-1 line
        ax.text(-0.04, 1, f"{st['rank1_random']} at rank 1", ha="right", va="center", fontsize=6.5,
                color=DARK)
        ax.text(1.04, 1, f"{st['rank1_rl']} at rank 1", ha="left", va="center", fontsize=6.5, color=DARK)
        # the largest rescues, named at their starting rank
        best = np.argsort(-ratio, kind="stable")[:6]
        best = best[ratio[best] >= 2]
        if an and int(an[0]["rep_pos"]) not in best:
            best = np.append(best, int(an[0]["rep_pos"]))
        placed = []
        for yv, j in sorted([(a[j], j) for j in best]):
            ly = yv
            for p in placed:
                if abs(np.log(ly) - np.log(p)) < 0.15:
                    ly = p * np.exp(0.15)
            placed.append(ly)
            ax.annotate(_gi(cands[rep[j]]), (0, yv), xytext=(-0.06, ly), textcoords="data", ha="right",
                        va="center", fontsize=6.5,
                        arrowprops=dict(arrowstyle="-", color=MID, lw=0.4, shrinkA=0, shrinkB=1.5))
        _panel(ax, "ab"[k], x=-0.2, y=1.0)
    handles = [Line2D([], [], color=RED, lw=1.0, label="Rank improved $\\geq$2$\\times$"),
               Line2D([], [], color=BLUE, lw=1.0, label="Rank worsened $\\geq$2$\\times$"),
               Line2D([], [], color=LIGHT, lw=1.0, label="Within 2$\\times$"),
               Line2D([], [], color=INK, lw=1.5, marker="o", ms=3,
                      label=f"{_gi(JJ[lead]['anatomy'][0]['gene'])} (Fig. 20)" if lead in JJ and JJ[lead].get("anatomy")
                      else "Case in Fig. 20")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    _save(fig, fig_dir, "fig19_rescue_flow")
    return stats


# ================================================================== fig20: anatomy
def fig_anatomy(S, J, ds, fig_dir):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm as mcm
    from matplotlib.colors import Normalize
    case = J["anatomy"][0]
    gene = case["gene"]
    A_r, A_l = case["atlases"]["random"], case["atlases"]["rl"]
    fig = plt.figure(figsize=(W2, 4.35))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.85, 1.0], width_ratios=[1, 1, 0.045],
                          hspace=0.22, wspace=0.08)
    norm = Normalize(vmin=-0.2, vmax=0.8)
    cmap = mpl.colormaps["viridis"]
    for k, (A, title) in enumerate(((A_r, "Random atlas"), (A_l, "REINFORCE atlas"))):
        ax = fig.add_subplot(gs[0, k])
        ax.set_aspect("equal")
        ax.axis("off")
        for jr, ang_lab in ((0.25, 0.0), (0.5, np.pi)):
            rr = 1.0 - jr
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, color=LIGHT, lw=0.5, ls=(0, (2, 2)), zorder=0))
            ax.text(rr * np.cos(ang_lab), 0.0, f"$J$ = {jr:.2f}", fontsize=6, color=MID, ha="center",
                    va="center", zorder=2, bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
        n = len(A["neighbours"])
        ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
        for t in range(n):
            rad = 1.0 - A["jaccard"][t]
            x, y = rad * np.cos(ang[t]), rad * np.sin(ang[t])
            wt = A["weight"][t]
            ax.plot([0, x], [0, y], color=MID, lw=0.3 + 5 * wt, zorder=1, solid_capstyle="butt")
            ax.scatter([x], [y], s=12 + 420 * wt, color=cmap(norm(A["agree"][t])), edgecolors=INK,
                       linewidths=0.4, zorder=3)
            lx, ly = (rad + 0.15) * np.cos(ang[t]), (rad + 0.15) * np.sin(ang[t])
            ha = "left" if np.cos(ang[t]) > 0.2 else ("right" if np.cos(ang[t]) < -0.2 else "center")
            ax.text(lx, ly, _gi(A["neighbours"][t]), fontsize=7, ha=ha, va="center", zorder=5)
        ax.scatter([0], [0], s=40, marker="s", color=INK, zorder=4)
        ax.text(0, -0.13, _gi(gene), fontsize=7.5, ha="center", va="top", zorder=6,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.25, 1.25)
        ax.set_title(f"{title}  (rank {_rank(A['rank'])}; cos = {A['pred_cos']:.2f})", fontsize=8)
        _panel(ax, "ab"[k], x=-0.02, y=0.99)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(f"Signature cosine of neighbour with {_gi(gene)}", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    cax.set_box_aspect(9)
    # the readout: what gets predicted, gene by gene
    g = case["strip_genes"][:36]
    rows = [("Measured", np.array(case["strip_truth"][:36])),
            ("Random atlas", np.array(case["strip_random"][:36])),
            ("REINFORCE atlas", np.array(case["strip_rl"][:36]))]
    M = np.stack([v / (np.abs(v).max() + 1e-12) for _, v in rows])
    ax = fig.add_subplot(gs[1, 0:2])
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_yticks(range(3))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g, rotation=90, fontsize=6, fontstyle="italic")
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    for yb in (0.5, 1.5):
        ax.axhline(yb, color="white", lw=1.2)
    _panel(ax, "c", x=-0.13, y=1.02)
    cax2 = fig.add_subplot(gs[1, 2])
    cb2 = fig.colorbar(im, cax=cax2, ticks=[-1, 0, 1])
    cb2.set_label("$\\Delta$ expression\n(row-scaled)", fontsize=7)
    cb2.ax.tick_params(labelsize=6.5, width=0.5, length=2)
    cb2.outline.set_linewidth(0.5)
    _save(fig, fig_dir, "fig20_rescue_anatomy")
    return gene


# ================================================================== fig21: noise
def fig_noise(S, J, ds, fig_dir):
    import matplotlib.pyplot as plt
    cands = S["cands"]
    st, cand, dv, dh = S["noise_step"], S["noise_cand"], S["noise_dvis"], S["noise_dhid"]
    r = S["noise_r"]
    picked = S["noise_picked"]
    steps = np.arange(1, r.size + 1)
    med = float(np.nanmedian(r))
    t = int(steps[np.nanargmin(np.abs(r - med))])        # the most typical decision
    m = st == t
    x, y, c = dv[m] * 1e3, dh[m] * 1e3, cand[m]
    gp = int(np.flatnonzero(c == picked[t - 1])[0])
    bh = int(np.argmax(y))
    rank_h = int((y > y[gp]).sum()) + 1
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.3, 1, 1], "wspace": 0.42})
    ax, ax2, ax3 = axes
    ax.axhline(0, color=MID, lw=0.4, ls=(0, (3, 2)), zorder=1)
    ax.axvline(0, color=MID, lw=0.4, ls=(0, (3, 2)), zorder=1)
    ax.scatter(x, y, s=5, color=LIGHT, edgecolors=MID, linewidths=0.2, zorder=2, rasterized=True)
    ax.scatter([x[gp]], [y[gp]], s=28, color=RED, edgecolors=INK, linewidths=0.6, zorder=4,
               label=f"Greedy's pick ({_gi(cands[c[gp]])})")
    ax.scatter([x[bh]], [y[bh]], s=30, marker="^", color=INK, linewidths=0, zorder=4,
               label=f"Best held-out gain ({_gi(cands[c[bh]])})")
    ax.annotate(f"held-out rank {rank_h}/{x.size}", (x[gp], y[gp]), xytext=(-7, -1), textcoords="offset points",
                ha="right", va="center", fontsize=6.5, zorder=5,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
    rr = float(np.corrcoef(x, y)[0, 1])
    ax.text(0.97, 0.03, f"$r$ = {rr:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("Gain on scored genes (MRR $\\times 10^3$)")
    ax.set_ylabel("Gain on held-out genes (MRR $\\times 10^3$)")
    span = y.max() - y.min()
    ax.set_ylim(y.min() - 0.05 * span, y.max() + 0.42 * span)       # room for the legend
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.1, fontsize=6.5)
    _panel(ax, "a", x=-0.2, y=1.0)
    # b: the same correlation at every one of greedy's decisions
    ax2.axhline(0, color=MID, lw=0.4, ls=(0, (3, 2)))
    ax2.plot(steps, r, color=INK, lw=0.8)
    ax2.axhline(med, color=RED, lw=0.6, ls=(0, (4, 2)), label=f"Median $r$ = {med:.2f}")
    ax2.scatter([t], [r[t - 1]], s=14, color=RED, edgecolors=INK, linewidths=0.5, zorder=3)
    ax2.legend(loc="upper right", fontsize=6.5, borderaxespad=0.1)
    ax2.set_xlim(1, steps[-1])
    ax2.set_ylim(min(-0.2, float(np.nanmin(r)) - 0.05), max(0.7, float(np.nanmax(r)) + 0.05))
    ax2.set_xlabel("Decision (genes already chosen)")
    ax2.set_ylabel("Pearson $r$, scored vs held-out gain")
    _panel(ax2, "b", x=-0.27, y=1.0)
    # c: where each actual pick ranked on held-out gain, in deciles
    pct = []
    for s_ in steps:
        mm = st == s_
        hh, cc = dh[mm], cand[mm]
        hp = hh[np.flatnonzero(cc == picked[s_ - 1])[0]]
        pct.append(((hh > hp).sum() + 0.5 * ((hh == hp).sum() - 1)) / max(1, hh.size - 1))
    pct = np.clip(np.array(pct), 0, 0.999999)
    cnt = np.bincount((pct * 10).astype(int), minlength=10)
    ax3.bar(np.arange(1, 11), cnt, width=0.75, color=MID, edgecolor=INK, linewidth=0.4, zorder=2)
    ax3.axhline(steps.size / 10, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=3, label="Random pick (uniform)")
    ax3.legend(loc="upper left", fontsize=6.5, borderaxespad=0.1)
    ax3.set_xticks(range(1, 11))
    ax3.set_xlim(0.4, 10.6)
    ax3.set_ylim(0, max(cnt.max(), steps.size / 10) * 1.32)
    from matplotlib.ticker import MaxNLocator
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.set_xlabel("Held-out gain decile of greedy's pick (1 = best)")
    ax3.set_ylabel(f"Picks (of {steps.size})")
    _panel(ax3, "c", x=-0.24, y=1.0)
    _save(fig, fig_dir, "fig21_greedy_noise")
    return {"step": t + 1, "r": rr, "pick": str(cands[c[gp]]), "pick_rank_heldout": rank_h,
            "n": int(x.size), "best": str(cands[c[bh]]), "median_r": med,
            "picks_by_heldout_decile": [int(v) for v in cnt]}


# ================================================================== entry
def draw(a):
    _style()
    os.makedirs(a.fig_dir, exist_ok=True)
    od = os.path.join(a.art_dir, "story")
    J = json.load(open(os.path.join(od, "story.json")))
    SS = {}
    for ds in ("replogle_k562_essential", "replogle_rpe1_essential"):
        p = os.path.join(od, f"story_{ds}.npz")
        if os.path.exists(p) and ds in J:
            SS[ds] = dict(np.load(p, allow_pickle=False))
    if not SS:
        print("[story] no story npz; nothing to draw")
        return 0
    lead = "replogle_k562_essential" if "replogle_k562_essential" in SS else next(iter(SS))
    out = {}
    for name, fn in (("gap", lambda: fig_gap(SS[lead], J[lead], lead, a.fig_dir)),
                     ("flow", lambda: fig_flow(SS, {d: J[d] for d in SS}, a.fig_dir, lead)),
                     ("anatomy", lambda: fig_anatomy(SS[lead], J[lead], lead, a.fig_dir)),
                     ("noise", lambda: fig_noise(SS[lead], J[lead], lead, a.fig_dir))):
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            out[name] = f"FAILED: {type(e).__name__}: {e}"
    with open(os.path.join(od, "drawn.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    return 0
