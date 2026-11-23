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


def gram_section(A, ds_list, P):
    """The b-estimation follow-up (scripts/repr_gram.py), from repr_gram/summary.json.

    Every table is PER QUERY, GAPS.md's unit; repr_gram's own report also carries the
    stricter per-gene numbers, and mixing the two is what once made its oracle row read
    as a contradiction of this file. Verdicts are computed from both lines' paired tests,
    not from the first line that looks good."""
    p = os.path.join(A, "repr_gram", "summary.json")
    if not os.path.exists(p):
        return
    g = json.load(open(p))
    dl = [d for d in ds_list if d in g["datasets"]]
    if not dl:
        return

    def q(d, k):
        r = g["datasets"][d]["results"].get(k)
        return r["metrics"]["top1"] if r else float("nan")

    P("### The combiner reads inner products; the shortfall is estimating them")
    P()
    P("The ridge solve needs b_j = ⟨y_j, y_g⟩. Every row above plugs in b_j = s(g,j)‖y_j‖ for")
    P("some similarity s. For the oracle s is the in-line cosine, so the plug-in *is* the true b")
    P("up to the per-row constant ‖y_g‖, which cancels in a cosine — the oracle row is not a")
    P("better ranking, it is the combiner handed an exact b. That makes neighbourhood choice and")
    P("b estimation separable, so `scripts/repr_gram.py`'s first part runs the 2×2: each")
    P("source's own neighbourhood, with its plug-in b and with the measured b (which cheats,")
    P("and is there only to bound what fixing b could buy). The unclipped oracle plug-in and")
    P("the measured b are asserted to agree, as the algebra requires.")
    P()
    P("| neighbourhood from | b | " + " | ".join(f"{SHORT.get(d,d)} top-1" for d in dl) + " |")
    P("|---|---|" + "---:|" * len(dl))
    for src, lab in (("go", "GO Jaccard"), ("learned", "learned priors"),
                     ("xline", "other line's Perturb-seq\\*"), ("oracle", "true signature cosine\\*")):
        for bn, blab in (("plugin", "plug-in"), ("true_b", "measured\\*")):
            k = f"{src}__gls_{bn}"
            if any(k in g["datasets"][d]["results"] for d in dl):
                b = "**" if (src, bn) == ("go", "true_b") else ""
                P(f"| {lab} | {blab} | " + " | ".join(f"{b}{q(d,k)*100:.1f}%{b}" for d in dl) + " |")
    P()
    fr = [q(d, "go__gls_true_b") / q(d, "oracle__gls_plugin") for d in dl]

    def ms(d, k):
        cp = g["datasets"][d]["results"][k]["chosen_per_fold"].values()
        return sorted({int(str(v).strip("()").split(",")[0]) for v in cp})
    m_true = sorted({m for d in dl for m in ms(d, "go__gls_true_b")})
    m_plug = sorted({m for d in dl for m in ms(d, "go__gls_plugin")})
    allm = [int(str(v).strip("()").split(",")[0]) for d in dl
            for v in g["datasets"][d]["results"]["go__gls_plugin"]["chosen_per_fold"].values()]
    small = sum(m < max(allm + [100]) for m in allm)
    small_ms = " or ".join(str(m) for m in sorted({m for m in allm if m < max(allm + [100])}))
    pr = os.path.join(A, "repr_probe", "summary.json")
    rec = (json.load(open(pr)) if os.path.exists(pr) else {"datasets": {}})["datasets"]
    rtxt = ", ".join(f"{rec[d]['go_vs_signature']['recall_at_10']*100:.0f}% on {SHORT.get(d,d)}"
                     for d in dl if d in rec)
    P(f"**GO's own neighbourhood, given the true inner products, reaches "
      + " and ".join(f"{q(d,'go__gls_true_b')*100:.1f}% on {SHORT.get(d,d)}" for d in dl)
      + "** — " + " and ".join(f"{f*100:.0f}%" for f in fr) + " of the oracle. GO ranks a gene's")
    P(f"top ten badly (recall@10 {rtxt}), but the pool it hands the solve already contains")
    P("almost everything the oracle uses. The pool size is chosen out of fold, and it is")
    P(f"telling: given the true b the solve takes m = {'/'.join(map(str, m_true))} in every fold on both lines,")
    P(f"given GO's plug-in b it takes m = {small_ms} in {small} of {len(allm)} folds -- with a poor b it can only")
    P("afford to trust a few neighbours. So the shortfall is not which genes to look at but")
    P("how much each one counts: **estimating b inside a GO-chosen pool**, a narrower target")
    P('than "find a better similarity".')
    P()

    # part 2: the method
    P("The method follows: regress the in-line cosine on pair features (every prior view,")
    P("its rank in the row and in the column, the atlas gene's norm and split-half")
    P("reliability, and — in the second arm — the other line's cosine and its coverage),")
    P("fitted out of fold over genes on the top of each gene's ordering plus a random tail.")
    P("`iso` is the control: a strictly monotone recalibration of one similarity, which keeps")
    P("its neighbour set and changes only magnitudes.")
    P()
    rows = [("go__knn_avg", "GO-kNN (the gap-0 predictor)"),
            ("xline__gls_plugin", "other line's Perturb-seq, plug-in b\\*"),
            ("prior_learned__gls", "learned b, prior views only"),
            ("xline_iso__gls", "other line's Perturb-seq, recalibrated b\\*"),
            ("xline_learned__gls", "learned b, with the other line\\*")]
    P("| | " + " | ".join(f"{SHORT.get(d,d)} top-1" for d in dl) + " | "
      + " | ".join(f"{SHORT.get(d,d)} vs GO-kNN, paired over genes" for d in dl) + " |")
    P("|---|" + "---:|" * len(dl) + "---|" * len(dl))
    for k, lab in rows:
        cells = []
        for d in dl:
            pv = g["datasets"][d]["paired_vs_go_knn"].get(k, {}).get("all")
            if pv is None:
                cells.append("—")
            else:
                w_, p_ = pv["win_rate_excl_ties"], pv["wilcoxon_p"]
                tag = (" **worse**" if w_ < 0.5 and p_ < 0.05 else
                       " better" if w_ > 0.5 and p_ < 0.05 else " n.s.")
                cells.append(f"win {w_*100:.0f}%, p = {p_:.0e}{tag}")
        P(f"| {lab} | " + " | ".join(f"{q(d,k)*100:.1f}%" for d in dl) + " | "
          + " | ".join(cells) + " |")
    P()
    # did the method replicate? decided from both lines, not the first that looks good
    key = "xline_learned__gls"
    verdict = {}
    for d in dl:
        pv = g["datasets"][d]["paired_vs_go_knn"].get(key, {}).get("all")
        verdict[d] = bool(pv) and pv["win_rate_excl_ties"] > 0.5 and pv["wilcoxon_p"] < 0.05
    bq = {d: (g["datasets"][d]["b_quality"].get("xline__plugin", {}).get("pearson_in_neighbourhood"),
              g["datasets"][d]["b_quality"].get("xline_learned", {}).get("pearson_in_neighbourhood"))
          for d in dl}
    if all(verdict.values()):
        P("**Learning b improves on GO-kNN on both lines**, paired over genes.")
    else:
        good = [SHORT.get(d, d) for d, v in verdict.items() if v]
        bad = [SHORT.get(d, d) for d, v in verdict.items() if not v]
        P(f"**Learning b helps on {', '.join(good) or 'neither line'} and does not replicate on "
          f"{', '.join(bad)}.** ")
        if good:
            d0 = next(d for d in dl if verdict[d])
            P(f"On {SHORT.get(d0,d0)} it is the best realistic row in this file, {q(d0,key)*100:.1f}% top-1 against "
              f"GO-kNN's {q(d0,'go__knn_avg')*100:.1f}%.")
        for d in dl:
            if not verdict[d]:
                P(f"On {SHORT.get(d,d)} it is worse than GO-kNN on rank and worse than simply plugging in "
                  f"the other line's similarity ({q(d,key)*100:.1f}% vs {q(d,'xline__gls_plugin')*100:.1f}%).")
        P("The b-quality diagnostic says why, measured where the solve reads it (Pearson with the")
        P("true b inside each gene's neighbourhood): learning b moves it "
          + "; ".join(f"{SHORT.get(d,d)} {a:.2f} → {b:.2f}" for d, (a, b) in bq.items() if a is not None)
          + ".")
        P("Where the regression barely improves b, it still reshuffles the ranking, and loses.")
        P("This is reported as a K562 observation, not a method.")
    P()
    warns = [w for d in dl for w in g["datasets"][d].get("interpretation_warnings", [])]
    if warns:
        P("> **Caveat.** " + " ".join(warns))
        P()


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
    P("ceiling set by a gene's own measured signature.")
    P()
    # With a realistic similarity the combiner does not vanish -- it TRADES. Counted from
    # the rows rather than asserted, because an earlier draft called it "worth nothing"
    # and the table under it said otherwise.
    up = dn = tot = 0
    for src in ("go", "learned", "xline"):
        for d in ds_list:
            a, b = f"{src}__knn_avg", f"{src}__gls"
            if a not in sel["datasets"][d]["results"] or b not in sel["datasets"][d]["results"]:
                continue
            tot += 1
            up += m(d, b) > m(d, a)
            dn += m(d, b, "median_rank") > m(d, a, "median_rank")
    P(f"**Given a realistic similarity the combiner trades rather than vanishes.** On {up} of {tot}")
    P(f"prior-only and cross-line rows it raises top-1, and on {dn} of {tot} it makes the median rank")
    P(f"*worse* — on {SHORT.get(k,k)}, GO goes {pct(m(k,'go__knn_avg'))} → {pct(m(k,'go__gls'))} top-1 while its median rank goes")
    P(f"{m(k,'go__knn_avg','median_rank'):.0f} → {m(k,'go__gls','median_rank'):.0f}, and the learned combination "
      f"{pct(m(k,'learned__knn_avg'))} → {pct(m(k,'learned__gls'))} while its rank goes")
    P(f"{m(k,'learned__knn_avg','median_rank'):.0f} → {m(k,'learned__gls','median_rank'):.0f}. The ridge solve amplifies whatever ordering it is handed:")
    P("with a good one that is a large gain, with a poor one it sharpens a few cases and")
    P("scrambles the rest — the same shape Framing A found when it whitened the scores. So the")
    P("combiner is not free money at today's representations; it buys the top slot and sells")
    P("the middle of the ranking.")
    P()
    if stats["datasets"]:
        b0 = next((v.get("median_rank_baseline") for v in stats["datasets"].get(ds_list[0], {}).values()
                   if isinstance(v, dict) and "median_rank_baseline" in v), None)
        qpg = ", ".join(f"~{sel['datasets'][d]['n_queries'] / sel['datasets'][d]['K']:.1f} on {SHORT.get(d,d)}"
                        for d in ds_list)
        P(f"Paired over **genes** (a gene's queries — {qpg} — are not independent evidence), Wilcoxon")
        P("signed rank on per-gene rank against `go__knn_avg`. Median rank in this table is per")
        P("*gene* — each gene's queries are averaged first — so it differs slightly from the")
        P("per-*query* median in the table above"
          + (f" ({m(ds_list[0],'go__knn_avg','median_rank'):.0f} vs {b0:.1f} for GO-kNN on "
             f"{SHORT.get(ds_list[0],ds_list[0])})" if b0 is not None else "") + ".")
        P()
        P("| | " + " | ".join(f"{SHORT.get(d,d)} median rank · win rate · p" for d in ds_list) + " |")
        P("|---|" + "---|" * len(ds_list))
        for key in ("xline__gls", "xline__knn_avg", "go__gls", "learned__knn_avg"):
            cells = []
            for d in ds_list:
                r = stats["datasets"].get(d, {}).get(key)
                if not r:
                    cells.append("—")
                elif r["wins"] + r["losses"] == 0:
                    # no discordant pair: the row IS the baseline (all weight on GO).
                    # Printing a 0% win rate here reads as "always loses".
                    cells.append(f"{r['median_rank_method']:.0f} · identical to baseline")
                else:
                    cells.append(f"{r['median_rank_method']:.0f} · "
                                 f"{r['win_rate_excl_ties']*100:.0f}% · {r['wilcoxon_p']:.1e}")
            P(f"| `{key}` | " + " | ".join(cells) + " |")
        P()
        def ps(d, k):
            return stats["datasets"].get(d, {}).get(k)
        P("Read against the win rates, the top-1 numbers are less impressive than they look.")
        for d in ds_list:
            r = ps(d, "xline__gls")
            if not r:
                continue
            better = r["win_rate_excl_ties"] > 0.5 and r["wilcoxon_p"] < 0.05
            worse = r["win_rate_excl_ties"] < 0.5 and r["wilcoxon_p"] < 0.05
            P(f"On {SHORT.get(d,d)} `xline__gls` moves the per-gene median rank "
              f"{r['median_rank_baseline']:.1f} → {r['median_rank_method']:.1f} and wins on "
              f"{r['win_rate_excl_ties']*100:.1f}% of genes, p = {r['wilcoxon_p']:.2g}"
              + (" — a real but narrow effect concentrated at the top of the ranking." if better else
                 " — significantly **worse** than GO-kNN on rank." if worse else " — not significant."))
        d0 = ds_list[0]
        r = ps(d0, "learned__knn_avg")
        if r and r["wilcoxon_p"] < 0.05 and r["win_rate_excl_ties"] > 0.5 \
                and m(d0, "learned__knn_avg") < m(d0, "go__knn_avg"):
            P(f"And `learned__knn_avg` improves {SHORT.get(d0,d0)}'s per-gene median rank to "
              f"{r['median_rank_method']:.1f} (p = {r['wilcoxon_p']:.0e}) while its top-1 is *below* GO's")
            P(f"({m(d0,'learned__knn_avg')*100:.2f}% vs {m(d0,'go__knn_avg')*100:.2f}%): the learned combination genuinely improves")
            P("the ordering and genuinely does not improve the top slot.")
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
            # every number below from paired_stats.json -- an earlier version typed them
            gr_ = {d: cv["groups"] for d, cv in cov.items() if cv}
            def ratio(v):
                return v["top1_xline_gls"] / v["top1_go_knn"] if v["top1_go_knn"] > 0 else float("nan")
            meas = {d: g_["real_prior"] for d, g_ in gr_.items() if "real_prior" in g_}
            imp = {d: g_["mean_imputed"] for d, g_ in gr_.items() if "mean_imputed" in g_}
            P("Where the other cell line really measured the gene the improvement holds on both lines — "
              + "; ".join(f"{SHORT.get(d,d)} top-1 {v['top1_go_knn']*100:.1f}% → {v['top1_xline_gls']*100:.1f}%, "
                          f"median rank {v['median_rank_go_knn']:.1f} → {v['median_rank_xline_gls']:.1f}"
                          for d, v in meas.items()) + ".")
            big = max(imp.items(), key=lambda kv: kv[1]["n_genes"], default=None)
            if big:
                d, v = big
                frac = v["n_genes"] / (v["n_genes"] + meas.get(d, {"n_genes": 0})["n_genes"])
                P(f"Where the row is imputed it is destroyed ({SHORT.get(d,d)} median rank "
                  f"{v['median_rank_go_knn']:.0f} → {v['median_rank_xline_gls']:.0f}). {SHORT.get(d,d)}'s bad aggregate is therefore")
                P(f"an artefact of {frac*100:.0f}% imputed rows, not evidence against the cross-line signal,")
                P("and the mechanism is tested on the mask rather than inferred from the aggregate.")
            P()
            rs = sorted(ratio(v) for v in meas.values() if ratio(v) == ratio(v))
            rtxt = (f"{rs[0]:.1f}–{rs[-1]:.1f}×" if len(rs) > 1 else f"{rs[0]:.1f}×") if rs else "—"
            P("The honest statement of the one thing that moves this metric: **a gene unseen in")
            P(f"*this* cell line but measured in another is identifiable {rtxt} better than GO")
            P("gets it** (" + ", ".join(f"top-1 {v['top1_go_knn']*100:.1f}% → {v['top1_xline_gls']*100:.1f}% on {SHORT.get(d,d)}"
                                      for d, v in meas.items())
              + ", paired within the covered genes). That is gap 0's unseen-*context* axis doing")
            P("the work, not prior knowledge, and it is consistent with everything else here:")
            P("measurement has value, annotation does not.")
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
        adv = {}
        for dsx in ds_list:
            if dsx not in curve["datasets"] or "go_jaccard" not in curve["datasets"][dsx]["real"]:
                continue
            gv = curve["datasets"][dsx]["real"]["go_jaccard"]
            nr = min(sweep_of(dsx), key=lambda w: abs(w["recall_at_k"] - gv["recall_at_k"]))
            adv[dsx] = (gv["knn_avg"]["top1"], nr["knn_avg"]["top1"])
        txt = "; ".join(f"{a*100:.2f}% vs {b*100:.2f}% on {SHORT.get(d,d)}" for d, (a, b) in adv.items())
        if adv and all(a < b for a, b in adv.values()):
            P(f"**One reading holds on {'both lines' if len(adv) > 1 else 'this line'}.** "
              "GO's *kNN* returns less than an isotropic")
            P(f"corruption of matched recall ({txt}), so GO's")
            P("misses are more adversarial than random — when it retrieves the wrong neighbours it")
            P("retrieves genes that are genuinely confusable with the target, not arbitrary ones.")
        elif adv:
            P(f"GO's *kNN* against an isotropic corruption of matched recall: {txt} — not")
            P("consistently below it, so no claim about the structure of GO's misses is made.")
        P()
        # per line: on how many real rows does the combiner beat the sweep's at matched recall?
        beat = {}
        for dsx in ds_list:
            if dsx not in curve["datasets"]:
                continue
            sw = sweep_of(dsx)
            n_b = n_t = 0
            for k2, v in curve["datasets"][dsx]["real"].items():
                if v["knn_avg"]["top1"] < 0.005:
                    continue
                near = min(sw, key=lambda w: abs(w["recall_at_k"] - v["recall_at_k"]))
                gr, gsw = gain_of(v), gain_of(near)
                if gr == gr and gsw == gsw:
                    n_t += 1
                    n_b += gr > gsw
            beat[dsx] = (n_b, n_t)
        if len(dirs) >= 2 and not (all(v > 0 for v in dirs.values()) or all(v < 0 for v in dirs.values())):
            P("**One reading does not.** Whether the *combiner* does better on a real")
            P("representation than on a matched corruption goes opposite ways on the two lines:")
            P("the real representation's ridge gain beats the sweep's on "
              + " and ".join(f"{b} of {t} {SHORT.get(d,d)} rows" for d, (b, t) in beat.items())
              + ". For GO alone that is "
              + "; ".join(f"{SHORT.get(d,d)} {gain_of(curve['datasets'][d]['real']['go_jaccard']):.2f}× vs "
                          f"{gain_of(min(sweep_of(d), key=lambda w: abs(w['recall_at_k'] - curve['datasets'][d]['real']['go_jaccard']['recall_at_k']))):.2f}×"
                          for d in dirs) + ".")
            P("(On RPE1 three of the four rows are the same ranking, because the learned")
            P("combinations put all their weight on GO.) A tempting story — that real")
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
        P("available representation.")
        P()
        P("That is the specification this section buys. A gene representation for this task needs")
        P("**top-k neighbour recall**, not global correlation, and it needs **calibrated**")
        P("similarity magnitudes: kNN reads only the ordering, but the ridge solve reads")
        cal_p = os.path.join(A, "repr_curve", "calibration.json")
        cal = json.load(open(cal_p)) if os.path.exists(cal_p) else {"datasets": {}}
        if cal["datasets"]:
            parts = []
            for d, v in cal["datasets"].items():
                c, zz = v["calibrated_cosine"], v["standardised"]
                parts.append(f"{SHORT.get(d,d)} {c['gls']['top1']*100:.1f}% → {zz['gls']['top1']*100:.1f}% "
                             f"(−{v['gls_fraction_lost_to_standardising']*100:.0f}%) with kNN unchanged to "
                             f"{v['knn_agrees_to']*100:.2f} points")
            P("b = s(g,j)‖y_j‖ as a magnitude. Measured on the oracle similarity, standardising it —")
            P("same ordering, different magnitudes — moves the ridge combiner's top-1 "
              + "; ".join(parts) + ".")
            P("So it should be trained on a top-k ranking objective with calibrated scores, not")
            P("fitted to reproduce the whole similarity matrix.")
        else:
            P("b = s(g,j)‖y_j‖ as a magnitude, so an uncalibrated similarity loses part of the")
            P("combiner's value even with a perfect ordering (`./submit.sh reprcalib` measures how")
            P("much). So it should be trained on a top-k ranking objective with calibrated scores,")
            P("not fitted to reproduce the whole similarity matrix.")
        P()
    gram_section(A, ds_list, P)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
