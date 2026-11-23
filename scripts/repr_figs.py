"""Draw the representation-anatomy figure (fig22) from artifacts/repr_*/.

Reads only what repr_probe.py, repr_select.py and repr_curve.py wrote; no model, no
h5ad, no recomputation -- so the figure re-renders cheaply and cannot drift from the
numbers in GAPS.md.

Journal style, matching fig18-fig21 (scripts/story_draw.py): 7-inch double-column width,
STIX serif, 7-8 pt, bold lowercase panel letters, greyscale with a red/blue accent, no
titles on the figure -- the message lives in the caption in GAPS.md. Vector PDF beside a
300-dpi PNG.

Usage: python -I repr_figs.py --art-dir D --fig-dir D [--datasets A B]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

RED = "#B2182B"
BLUE = "#2166AC"
INK = "#000000"
DARK = "0.25"
MID = "0.55"
LIGHT = "0.78"
FAINT = "0.88"
SHORT = {"replogle_k562_essential": "K562", "replogle_rpe1_essential": "RPE1"}
W2 = 7.0


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
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.titlepad": 4, "axes.labelpad": 2,
        "lines.linewidth": 0.9, "patch.linewidth": 0.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})


def _panel(ax, letter, x=-0.16, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="left", va="bottom")


def _save(fig, fig_dir, stem):
    import matplotlib.pyplot as plt
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        print(f"[repr] wrote {p}")
    plt.close(fig)


def fig22(art, fig_dir, datasets):
    """(a) the ladder, (b) how far GO is from the signature ordering, (c) the
    quality -> retrieval transfer function with the real representations on it."""
    import matplotlib.pyplot as plt

    probe = json.load(open(os.path.join(art, "repr_probe", "summary.json")))
    sel = json.load(open(os.path.join(art, "repr_select", "summary.json")))
    curve_p = os.path.join(art, "repr_curve", "summary.json")
    curve = json.load(open(curve_p)) if os.path.exists(curve_p) else {"datasets": {}}

    fig = plt.figure(figsize=(W2, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.85, 1.15], wspace=0.55)

    # ---------------------------------------------------------- (a) the ladder
    ax = fig.add_subplot(gs[0, 0])
    rows = [("GO-kNN", "go__knn_avg", MID),
            ("+ ridge combiner", "go__gls", MID),
            ("oracle sim., kNN", "oracle__knn_avg", BLUE),
            ("oracle sim., ridge", "oracle__gls", RED),
            ("own measurement", "ceiling__measured", DARK)]
    ds_list = [d for d in datasets if d in sel["datasets"]]
    h, gap = 0.36, 0.26
    yt, yl = [], []
    for i, (lab, key, col) in enumerate(rows):
        base = i * (len(ds_list) * h + gap)
        for j, ds in enumerate(ds_list):
            v = sel["datasets"][ds]["results"].get(key)
            if v is None:
                continue
            y = base + j * h
            ax.barh(y, v["metrics"]["top1"] * 100, height=h * 0.86, color=col,
                    alpha=1.0 if j == 0 else 0.45, edgecolor="none")
            ax.text(v["metrics"]["top1"] * 100 + 0.8, y, f"{v['metrics']['top1']*100:.1f}",
                    va="center", ha="left", fontsize=6, color=INK)
        yt.append(base + (len(ds_list) - 1) * h / 2)
        yl.append(lab)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.set_xlabel("top-1 retrieval (%)")
    ax.set_xlim(0, 58)
    ax.invert_yaxis()
    hs = [plt.Rectangle((0, 0), 1, 1, color=DARK, alpha=a) for a in (1.0, 0.45)]
    ax.legend(hs, [SHORT.get(d, d) for d in ds_list], loc="lower right", fontsize=6)
    _panel(ax, "a", x=-0.52)

    # ------------------------------------------- (b) GO's ordering vs the truth
    ax = fig.add_subplot(gs[0, 1])
    for j, ds in enumerate(ds_list):
        z = np.load(os.path.join(art, "repr_probe", f"probe_{ds}.npz"))
        r = np.sort(z["go_rank_of_true_nn"])
        K = z["cands"].size
        ax.plot(r / K, np.arange(1, r.size + 1) / r.size, color=(BLUE if j == 0 else RED),
                lw=1.0, label=SHORT.get(ds, ds))
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xlabel("GO rank of the true\nnearest neighbour (fraction of $K$)")
    ax.set_ylabel("cumulative fraction of genes")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=6)
    _panel(ax, "b", x=-0.34)

    # --------------------------------------------------- (c) transfer function
    ax = fig.add_subplot(gs[0, 2])
    drew = False
    for j, ds in enumerate(ds_list):
        p = os.path.join(art, "repr_curve", f"curve_{ds}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        drew = True
        xk = "real_recall" if "real_recall" in z.files else "real_spearman"
        o = np.argsort(z["recall"])
        ax.plot(z["recall"][o] * 100, z["knn_top1"][o] * 100, color=BLUE, lw=1.0,
                ls="-" if j == 0 else (0, (3, 2)),
                label=f"kNN average, {SHORT.get(ds, ds)}")
        ax.plot(z["recall"][o] * 100, z["gls_top1"][o] * 100, color=RED, lw=1.0,
                ls="-" if j == 0 else (0, (3, 2)),
                label=f"ridge combiner, {SHORT.get(ds, ds)}")
        if j == 0:
            sx = z[xk] * (100 if xk == "real_recall" else 1)
            for nm, sp, t1 in zip(z["real_names"], sx, z["real_gls_top1"]):
                ax.plot([sp], [t1 * 100], marker="o", ms=3, mfc="white", mec=INK, mew=0.7,
                        zorder=5, ls="none")
                ax.annotate(str(nm).split("(")[0], (sp, t1 * 100), textcoords="offset points",
                            xytext=(3, 3), fontsize=5.5, color=DARK)
    if drew:
        ax.set_xlabel("ordering quality\n(recall@10 of the true nearest neighbours, %)")
        ax.set_ylabel("top-1 retrieval (%)")
        ax.legend(loc="upper left", fontsize=5.5)
    else:
        ax.text(0.5, 0.5, "repr_curve not run yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color=MID)
        ax.set_xticks([])
        ax.set_yticks([])
    _panel(ax, "c", x=-0.30)

    _save(fig, fig_dir, "fig22_representation_anatomy")


# ===================================================================== statistics
def paired_stats(art, datasets, out_path):
    """Paired tests over GENES, not queries.

    Each gene contributes one rank per method, so a comparison is paired on the gene and
    the ~1.8 queries a gene owns are not counted as independent evidence. Wilcoxon signed
    rank on the per-gene rank (lower is better) plus the win rate, against the predictor
    every number in GAPS.md uses."""
    from scipy.stats import wilcoxon
    out = {"baseline": "go__knn_avg", "datasets": {}}
    for ds in datasets:
        p = os.path.join(art, "repr_select", f"result_{ds}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        keys = [k[5:] for k in z.files if k.startswith("rank_")]
        if "go__knn_avg" not in keys:
            continue
        base = z["rank_go__knn_avg"]
        rows = {}
        for k in keys:
            if k == "go__knn_avg":
                continue
            v = z[f"rank_{k}"]
            d = base - v                       # > 0 means the method ranks the gene better
            nz = d[d != 0]
            try:
                stat, pv = wilcoxon(nz) if nz.size else (float("nan"), 1.0)
            except Exception:
                stat, pv = float("nan"), float("nan")
            rows[k] = {"n_genes": int(d.size), "wins": int((d > 0).sum()),
                       "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                       "win_rate_excl_ties": float((d > 0).sum() / max((d != 0).sum(), 1)),
                       "median_rank_baseline": float(np.median(base)),
                       "median_rank_method": float(np.median(v)),
                       "wilcoxon_p": float(pv)}
        # The cross-line kernel is mean-imputed for genes the other line never measured
        # (45% on RPE1, 0.6% on K562). If that is what drives RPE1's "top-1 up, median rank
        # down" pattern, the effect must separate on exactly that mask -- so test it rather
        # than assert it. The mask is recomputed from the aligned priors, no rerun needed.
        pp = os.path.join(art, "rl_acquire", "related", f"priors_{ds}.npz")
        if os.path.exists(pp) and "rank_xline__gls" in z.files:
            zp = np.load(pp, allow_pickle=False)
            pnames = [str(x) for x in zp["names"]]
            mk = [n for n in pnames if n in ("rpe1_kernel", "k562_perturbseq_kernel")]
            if mk:
                R = zp["kernels"][pnames.index(mk[0])]
                _, inv, cnt = np.unique(np.round(R, 5), axis=0,
                                        return_inverse=True, return_counts=True)
                imp = cnt[inv] > 1
                strat = {}
                for grp, sel_m in (("real_prior", ~imp), ("mean_imputed", imp)):
                    if sel_m.sum() == 0:
                        continue
                    strat[grp] = {
                        "n_genes": int(sel_m.sum()),
                        "median_rank_go_knn": float(np.median(base[sel_m])),
                        "median_rank_xline_gls": float(np.median(z["rank_xline__gls"][sel_m])),
                        "top1_go_knn": float((base[sel_m] <= 1).mean()),
                        "top1_xline_gls": float((z["rank_xline__gls"][sel_m] <= 1).mean())}
                rows["_xline_by_prior_coverage"] = {"kernel": mk[0], "groups": strat}
        out["datasets"][ds] = rows
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[repr] wrote {out_path}")
    for ds, rows in out["datasets"].items():
        # pop the coverage entry FIRST: it has no per-method fields, so anything that
        # indexes rows positionally must not see it
        cov = rows.pop("_xline_by_prior_coverage", None)
        if not rows:
            continue
        print(f"  {ds}: baseline median rank {list(rows.values())[0]['median_rank_baseline']:.0f}")
        if cov:
            for g, v in cov["groups"].items():
                print(f"    [{cov['kernel']} {g}] n={v['n_genes']:4d}  "
                      f"medrank GO {v['median_rank_go_knn']:6.1f} -> xline+gls "
                      f"{v['median_rank_xline_gls']:6.1f}   "
                      f"top1 {v['top1_go_knn']*100:4.1f}% -> {v['top1_xline_gls']*100:4.1f}%")
        for k, v in sorted(rows.items(), key=lambda kv: kv[1]["median_rank_method"]):
            print(f"    {k:26s} medrank {v['median_rank_method']:6.1f}  "
                  f"win {v['win_rate_excl_ties']*100:5.1f}%  p={v['wilcoxon_p']:.2e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--datasets", nargs="+",
                    default=["replogle_k562_essential", "replogle_rpe1_essential"])
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    _style()
    for fn, label in ((lambda: fig22(a.art_dir, a.fig_dir, a.datasets), "fig22"),
                      (lambda: paired_stats(a.art_dir, a.datasets,
                                            os.path.join(a.art_dir, "repr_select",
                                                         "paired_stats.json")), "paired_stats")):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[repr] {label} FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
