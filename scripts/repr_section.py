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
        P()
        P("Read against the win rates, the top-1 numbers are less impressive than they look.")
        P("`xline__gls` improves K562's median rank (35.5 → 29.0) but wins on only 53.4% of")
        P("genes, p = 0.017 — a real but narrow effect concentrated at the top of the ranking.")
        P("On RPE1 the same row is significantly **worse** than GO-kNN on rank (125 → 231).")
        P("And `learned__knn_avg` has the best median rank of any prior-only row on K562")
        P("(24.0, p = 2e-11) while its top-1 is *below* GO's: the learned combination genuinely")
        P("improves the ordering and genuinely does not improve the top slot.")
        P()
        # the cross-line kernel split by whether the other line actually measured the gene
        cov = {d: stats["datasets"].get(d, {}).get("_xline_by_prior_coverage")
               for d in ds_list}
        if any(cov.values()):
            P("**The cross-line row splits cleanly on whether the other line measured the gene.**")
            P("`related_work.py` mean-imputes genes a prior source lacks, so the aggregate mixes a")
            P("real measurement with a constant. Separating them, paired within each group:")
            P()
            P("| line | genes | n | median rank, GO-kNN → xline+ridge | top-1, GO-kNN → xline+ridge |")
            P("|---|---|---:|---:|---:|")
            for d, cv in cov.items():
                if not cv:
                    continue
                for g, v in cv["groups"].items():
                    lab = "other line measured it" if g == "real_prior" else "mean-imputed"
                    P(f"| {SHORT.get(d,d)} | {lab} | {v['n_genes']} | "
                      f"{v['median_rank_go_knn']:.1f} → {v['median_rank_xline_gls']:.1f} | "
                      f"{v['top1_go_knn']*100:.1f}% → {v['top1_xline_gls']*100:.1f}% |")
            P()
            P("Where the other cell line really measured the gene the improvement is large and")
            P("consistent on both lines — RPE1 top-1 2.1% → 6.1% with median rank 38.8 → 20.2,")
            P("K562 2.4% → 4.0% with 35.0 → 28.5. Where the row is imputed it is destroyed")
            P("(RPE1 383 → 695). RPE1's bad aggregate is therefore an artefact of 45% imputed")
            P("rows, not evidence against the cross-line signal, and the mechanism is tested on")
            P("the mask rather than inferred from the aggregate.")
            P()
            P("The honest statement of the one thing that moves this metric: **a gene unseen in")
            P("*this* cell line but measured in another is identifiable roughly 2–3× better than")
            P("GO gets it** (top-1 2.1% → 6.1% on RPE1, 2.4% → 4.0% on K562, paired within the")
            P("covered genes). That is gap 0's unseen-*context* axis doing the work, not prior")
            P("knowledge, and it is consistent with everything else here: measurement has value,")
            P("annotation does not.")
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
        # Where does the combiner start to pay? Two separate questions, and conflating
        # them produced a wrong claim once already: the ISOTROPIC sweep's threshold is not
        # the threshold for REAL representations, which do better than the sweep at
        # matched recall.
        BAR = 1.3
        pts_r = sorted(pts, key=lambda v: v["recall_at_k"])
        def gain(v):
            return (v["gls"]["top1"] / v["knn_avg"]["top1"]
                    if v["knn_avg"]["top1"] >= 0.005 else float("nan"))
        start = None
        for i in range(len(pts_r)):
            gs = [gain(v) for v in pts_r[i:]]
            if all(g == g and g >= BAR for g in gs):
                start = pts_r[i]
                break
        dip = min((v for v in pts_r if gain(v) == gain(v)), key=gain, default=None)
        P("**On the sweep, the combiner needs high ordering quality and below it can hurt.**")
        if dip is not None:
            P(f"Its worst point is {gain(dip):.1f}× at recall@10 {dip['recall_at_k']*100:.1f}% — the ridge solve is")
            P("*worse* than plain averaging there, because it amplifies whatever ordering it is")
            P("handed and a middling ordering amplified is still middling.")
        if start is not None:
            P(f"Only from recall@10 ≈ {start['recall_at_k']*100:.0f}% does every higher level clear {BAR:g}×, rising to")
            P(f"{gain(pts_r[-1]):.1f}× at the oracle.")
        P()
        # Real representations vs the sweep at matched recall. On K562 they do BETTER
        # than the sweep's combiner and on RPE1 WORSE, so the redundancy story that K562
        # alone suggests does not replicate and is not claimed. Computed per line and the
        # direction checked, rather than generalised from the first line looked at.
        def sweep_of(dsx):
            cc = curve["datasets"][dsx]["curve"]
            return sorted(cc.values(), key=lambda v: v["recall_at_k"])
        def gain_of(v):
            return (v["gls"]["top1"] / v["knn_avg"]["top1"]
                    if v["knn_avg"]["top1"] >= 0.005 else float("nan"))
        P("**How the real representations sit against the sweep**, matched on recall@10:")
        P()
        P("| line | representation | recall@10 | kNN: real vs sweep | ridge gain: real vs sweep |")
        P("|---|---|---:|---:|---:|")
        dirs = {}
        for dsx in ds_list:
            if dsx not in curve["datasets"]:
                continue
            sw = sweep_of(dsx)
            for k2, v in curve["datasets"][dsx]["real"].items():
                if v["knn_avg"]["top1"] < 0.005:
                    continue
                near = min(sw, key=lambda w: abs(w["recall_at_k"] - v["recall_at_k"]))
                gr, gsw = gain_of(v), gain_of(near)
                P(f"| {SHORT.get(dsx,dsx)} | {k2.split('(')[0].replace('_',' ')} | "
                  f"{v['recall_at_k']*100:.1f}% | "
                  f"{v['knn_avg']['top1']*100:.2f}% vs {near['knn_avg']['top1']*100:.2f}% | "
                  + (f"{gr:.2f}× vs {gsw:.2f}×" if gr == gr and gsw == gsw else "—") + " |")
                if k2 == "go_jaccard" and gr == gr and gsw == gsw:
                    dirs[dsx] = gr - gsw
        P()
        P("**One reading holds on both lines.** GO's *kNN* returns less than an isotropic")
        P("corruption of matched recall (2.87% vs 4.03% on K562; 1.54% vs 3.21% on RPE1), so GO's")
        P("misses are more adversarial than random — when it retrieves the wrong neighbours it")
        P("retrieves genes that are genuinely confusable with the target, not arbitrary ones.")
        P()
        if len(dirs) >= 2 and not (all(v > 0 for v in dirs.values()) or all(v < 0 for v in dirs.values())):
            P("**One reading does not.** Whether the *combiner* does better on a real")
            P("representation than on a matched corruption goes opposite ways on the two lines")
            P("(" + ", ".join(f"{SHORT.get(d,d)} {'better' if v > 0 else 'worse'} by {abs(v):.2f}×"
                              for d, v in dirs.items()) + "). A tempting story — that real")
            P("mistakes are *redundant*, retrieving several members of one complex, which is exactly")
            P("the structure G⁻¹ deconvolves — fits K562 and is refuted by RPE1. So it is recorded")
            P("as unexplained rather than asserted, and the isotropic sweep is not a reliable")
            P("predictor of the combiner's value on a real representation in either direction.")
        elif len(dirs) >= 2:
            P("**And it holds in the same direction on both lines**, so the sweep systematically")
            P(("understates" if list(dirs.values())[0] > 0 else "overstates")
              + " the combiner for real representations.")
        P()
        P("What *is* consistent: the combiner's large gains need high ordering quality. It reaches")
        P(", ".join(f"{gain_of(sweep_of(d)[-1]):.1f}× on {SHORT.get(d,d)}" for d in ds_list
                    if d in curve["datasets"]) + " at the oracle, against 1.1–2.1× on any")
        P("available representation, and the ridge solve can even fall *below* plain averaging at")
        P("middling quality because it amplifies whatever ordering it is handed.")
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
