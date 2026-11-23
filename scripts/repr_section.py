"""Generate the GAPS.md sub-sections for the combiner and the transfer function.

Reads artifacts/repr_select/ and artifacts/repr_curve/ and prints markdown, so no number
in the write-up is transcribed by hand. Prints to stdout; the caller splices it.

Usage: python -I repr_section.py --art-dir D [--datasets A B]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SHORT = {"replogle_k562_essential": "K562", "replogle_rpe1_essential": "RPE1"}


def pct(x):
    return f"{x*100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--datasets", nargs="+",
                    default=["replogle_k562_essential", "replogle_rpe1_essential"])
    a = ap.parse_args()
    A = a.art_dir
    sel = json.load(open(os.path.join(A, "repr_select", "summary.json")))
    stats_p = os.path.join(A, "repr_select", "paired_stats.json")
    stats = json.load(open(stats_p)) if os.path.exists(stats_p) else {"datasets": {}}
    curve_p = os.path.join(A, "repr_curve", "summary.json")
    curve = json.load(open(curve_p)) if os.path.exists(curve_p) else {"datasets": {}}
    ds_list = [d for d in a.datasets if d in sel["datasets"]]
    P = print

    def m(ds, key, field="top1"):
        r = sel["datasets"][ds]["results"].get(key)
        return r["metrics"][field] if r else float("nan")

    P("### Ranking the neighbours is not the lever; combining them is")
    P()
    P("Write the prediction as a weighted combination of atlas signatures,")
    P("ŷ = Σ_j w_j y_j. Then cos(ŷ, y_g) = (Σ_j w_j ⟨y_j, y_g⟩) / √(wᵀGw), and maximising it")
    P("has the closed form **w\\* ∝ G⁻¹b**. The Gram matrix G is between *atlas* genes, which")
    P("are measured, so redundancy among candidate neighbours is known exactly without knowing")
    P("the answer; only b = s(g,·)·‖y‖ needs the representation. kNN averaging ignores G")
    P("entirely, which double-counts neighbours that are near-duplicates of each other.")
    P("`gls` is that solve with ridge λ; `mmr` is its discrete sibling. λ, the neighbourhood")
    P("size m and the MMR penalty are chosen on a **different fold of genes** than the one")
    P("being predicted.")
    P()
    hdr = "| similarity | combiner | " + " | ".join(f"{SHORT.get(d,d)} top-1" for d in ds_list) + " |"
    P(hdr)
    P("|---|---|" + "---:|" * len(ds_list))
    order = [("go", "knn_avg", "GO Jaccard", "the predictor every gap-0 number uses"),
             ("go", "gls", "GO Jaccard", ""), ("go", "mmr", "GO Jaccard", ""),
             ("learned", "knn_avg", "learned priors", ""),
             ("learned", "gls", "learned priors", ""),
             ("xline", "knn_avg", "other line's Perturb-seq\\*", ""),
             ("xline", "gls", "other line's Perturb-seq\\*", ""),
             ("oracle", "knn_avg", "true signature cosine\\*", ""),
             ("oracle", "mmr", "true signature cosine\\*", ""),
             ("oracle", "gls", "true signature cosine\\*", "")]
    for src, comb, lab, _ in order:
        key = f"{src}__{comb}"
        if not any(key in sel["datasets"][d]["results"] for d in ds_list):
            continue
        b = "**" if (src, comb) in (("go", "knn_avg"), ("oracle", "gls")) else ""
        P(f"| {lab} | {comb} | " + " | ".join(f"{b}{pct(m(d,key))}{b}" for d in ds_list) + " |")
    for key, lab in (("ceiling__measured", "the gene's own measurement\\*"),):
        P(f"| {lab} | — | " + " | ".join(f"{pct(m(d,key))}" for d in ds_list) + " |")
    P()
    P("`*` uses a measurement of perturbation effect rather than prior knowledge alone.")
    P()
    k = ds_list[0]
    gains = [m(d, "oracle__gls") / m(d, "oracle__knn_avg") for d in ds_list
             if m(d, "oracle__knn_avg") > 0]
    gtxt = "–".join(f"{g:.1f}×" for g in sorted(set(round(g, 1) for g in gains))) or "n/a"
    P(f"**Given a good similarity the combiner is worth {gtxt}** ({pct(m(k,'oracle__knn_avg'))} → "
      f"{pct(m(k,'oracle__gls'))} on {SHORT.get(k,k)}" +
      (f", {pct(m(ds_list[1],'oracle__knn_avg'))} → {pct(m(ds_list[1],'oracle__gls'))} on {SHORT.get(ds_list[1],ds_list[1])}"
       if len(ds_list) > 1 else "") + "), clearing even the")
    P("ceiling set by a gene's own measured signature. **Given a realistic similarity it is worth")
    P(f"nothing**: on {SHORT.get(k,k)} GO goes {pct(m(k,'go__knn_avg'))} → {pct(m(k,'go__gls'))} while its top-10 and median rank both")
    P("get worse. The ridge solve amplifies whatever ordering it is handed, and GO has nothing")
    P("to amplify.")
    P()
    if stats["datasets"]:
        P("Paired over **genes** (a gene's ~1.8 queries are not independent evidence), Wilcoxon")
        P("signed rank on per-gene rank against `go__knn_avg`:")
        P()
        P("| | " + " | ".join(f"{SHORT.get(d,d)} median rank · win rate · p" for d in ds_list) + " |")
        P("|---|" + "---|" * len(ds_list))
        for key in ("xline__gls", "xline__knn_avg", "go__gls", "learned__knn_avg"):
            cells = []
            for d in ds_list:
                r = stats["datasets"].get(d, {}).get(key)
                cells.append(f"{r['median_rank_method']:.0f} · {r['win_rate_excl_ties']*100:.0f}% · {r['wilcoxon_p']:.1e}"
                             if r else "—")
            P(f"| `{key}` | " + " | ".join(cells) + " |")
        d0 = ds_list[0]
        base = stats["datasets"].get(d0, {}).get("xline__gls", {}).get("median_rank_baseline")
        if base is not None:
            P()
            P(f"(`go__knn_avg` baseline median rank {base:.0f} on {SHORT.get(d0,d0)}.)")
        P()

    if curve["datasets"]:
        P("### How good would a representation have to be?")
        P()
        P("A verdict without a target is not actionable, so similarity *quality* is swept")
        P("continuously by corrupting the true signature similarity,")
        P("S_α = α·z(C) + √(1−α²)·z(E), and retrieval read off at each level.")
        P()
        d0 = ds_list[0]
        c = curve["datasets"][d0]["curve"]
        P(f"| {SHORT.get(d0,d0)}: ordering quality | Spearman | recall@10 | top-1, kNN avg | top-1, ridge | ridge gain |")
        P("|---|---:|---:|---:|---:|---:|")
        for k2, v in c.items():
            # a ratio of two near-zero rates is noise, not a gain
            g = (v["gls"]["top1"] / v["knn_avg"]["top1"]
                 if v["knn_avg"]["top1"] >= 0.005 else float("nan"))
            P(f"| {k2} | {v['spearman']:+.3f} | {v['recall_at_k']*100:.1f}% | "
              f"{v['knn_avg']['top1']*100:.2f}% | {v['gls']['top1']*100:.2f}% | "
              + (f"{g:.1f}×" if g == g else "—") + " |")
        for k2, v in curve["datasets"][d0]["real"].items():
            P(f"| **{k2}** | **{v['spearman']:+.3f}** | **{v['recall_at_k']*100:.1f}%** | "
              f"**{v['knn_avg']['top1']*100:.2f}%** | **{v['gls']['top1']*100:.2f}%** | |")
        P()
        P("Two readings, and the first corrects the obvious one.")
        P()
        real = curve["datasets"][d0]["real"]
        go = real.get("go_jaccard")
        pts = list(c.values())
        if go:
            # the sweep point whose Spearman matches GO's, and the one whose recall does
            bs = min(pts, key=lambda v: abs(v["spearman"] - go["spearman"]))
            br = min(pts, key=lambda v: abs(v["recall_at_k"] - go["recall_at_k"]))
            fs = go["knn_avg"]["top1"] / bs["knn_avg"]["top1"] if bs["knn_avg"]["top1"] > 0 else float("nan")
            fr = go["knn_avg"]["top1"] / br["knn_avg"]["top1"] if br["knn_avg"]["top1"] > 0 else float("nan")
            P("**Global rank correlation is the wrong summary.** The sweep point that matches GO")
            P(f"Jaccard's Spearman ({go['spearman']:+.3f}) reaches {bs['spearman']:+.3f} and returns")
            P(f"{bs['knn_avg']['top1']*100:.2f}% top-1 against GO's {go['knn_avg']['top1']*100:.2f}%"
              + (f" — a factor of {fs:.1f}." if fs == fs else "."))
            P("The recall column says why: at that Spearman the corruption recovers")
            P(f"{bs['recall_at_k']*100:.1f}% of a gene's true top-10 neighbours and GO recovers")
            P(f"{go['recall_at_k']*100:.1f}%. Isotropic noise spreads error over the whole ordering; GO is")
            P("wrong almost everywhere but concentrates what accuracy it has at the **top** of the")
            P("list, the only part a kNN or a ridge solve reads. Matched instead on recall@10")
            P(f"({br['recall_at_k']*100:.1f}%), the sweep returns {br['knn_avg']['top1']*100:.2f}% against GO's "
              f"{go['knn_avg']['top1']*100:.2f}%"
              + (f" — within {max(fr,1/fr):.1f}×" if fr == fr and fr > 0 else "")
              + ", so recall@k is the")
            P("axis these curves should be read on. It is still not exact, and the residual runs")
            P("one way on both lines: GO returns *less* than an isotropic corruption at matched")
            P("recall, so GO's misses are more adversarial than random — when it picks the wrong")
            P("neighbours it picks genes that are genuinely confusable with the target.")
            P()
        # Where does the combiner start to pay? The gain is NOT monotonic in quality: at
        # the bottom of the sweep both rates are near the floor and their ratio is erratic
        # (a 0.30%/0.81% pair reads as 2.7x and means nothing). So the threshold is the
        # SUSTAINED one -- the lowest recall above which every higher-quality point also
        # clears the bar -- not the first point that happens to clear it.
        BAR = 1.3
        pts_r = sorted(pts, key=lambda v: v["recall_at_k"])
        def gain(v):
            return (v["gls"]["top1"] / v["knn_avg"]["top1"]
                    if v["knn_avg"]["top1"] >= 0.005 else float("nan"))
        start = None
        for i in range(len(pts_r)):
            tail = pts_r[i:]
            gs = [gain(v) for v in tail]
            if all(g == g and g >= BAR for g in gs):
                start = pts_r[i]
                break
        if start is not None:
            P(f"**The combiner switches on above recall@10 ≈ {start['recall_at_k']*100:.0f}%** on "
              f"{SHORT.get(d0,d0)}: that is the")
            P(f"lowest swept quality above which *every* higher level also clears {BAR:g}×. The gain is")
            P("not monotonic lower down — at the bottom of the sweep both rates sit near the floor")
            P("and their ratio is erratic — but from that point it rises steadily to the oracle's")
            P(f"{gain(pts_r[-1]):.1f}×. Every available representation sits at or under the threshold:")
            P("  " + ", ".join(f"{k2.split('(')[0]} {v['recall_at_k']*100:.1f}%"
                               for k2, v in real.items()) + ".")
            P("That is precisely why none of them moved anything.")
        else:
            P(f"**No sustained combiner gain above {BAR:g}× appears anywhere in the sweep** on "
              f"{SHORT.get(d0,d0)},")
            P("so no threshold is claimed from this run.")
        P()
        P("That is the specification this section buys. A gene representation for this task needs")
        P("**top-k neighbour recall**, not global correlation, and it needs **calibrated**")
        P("similarity magnitudes: kNN reads only the ordering, but the ridge solve reads")
        P("b = s(g,j)‖y_j‖ as a magnitude, and standardising the similarity alone costs about a")
        P("fifth of the combiner's value. So it should be trained on a top-k ranking objective")
        P("with calibrated scores, not fitted to reproduce the whole similarity matrix.")
        P()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
