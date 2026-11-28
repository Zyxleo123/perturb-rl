#!/usr/bin/env python
"""Extra source screens for sim-to-real acquisition (rl_sim.py, PJ_SIM_SOURCE).

Why. The sim-to-real result (GAPS.md "Sim-to-real") ties the full prior floor for
one diagnosed reason: a candidate the source line never screened cannot be simulated
(K562's candidates are 78% covered by RPE1, RPE1's 53% by K562), and dropping them
costs about what RL gains on the covered ones. That is a testable mechanism: give
the simulator a source that covers the candidates and the same-candidates win should
become an all-candidates win. Two kinds of extra source:

  k562gw      Replogle 2022 genome-wide K562 CRISPRi Perturb-seq, the authors'
              normalised pseudobulk (one row per perturbation, z-scored against
              controls per gem group). Covers ~all of RPE1's candidates, and is a
              DIFFERENT line from RPE1, so it is a legal source for the RPE1 target
              only (for K562 it would be the same line's replicate screen).
              Pseudobulk only: no held-out cells, so a gene's query is its own
              signature (Q = Y, owner = i) -- a noise-free reward inside the simulator.
  lincs_<L>   LINCS L1000 CRISPR knockouts in line L, as packaged by PDGrapher
              (10 lines, ~3-4k perturbed genes each, ~10.7k inferred readout genes).
              Several replicate samples per gene: the signature is the mean of all of
              them minus the control mean, the queries are single samples of a held-out
              half -- the same honest split rl_headroom.build uses for Perturb-seq.

Every source is cut to the gene universe of the two Replogle caches (the only genes a
deployment can ask about), its readout genes that are themselves perturbed are masked
(as rl_headroom masks targets), and it gets its own GO feature matrix built by
rl_headroom.go_features -- so it is a drop-in rl_acquire.World cache.

Modes (each run as its own job step):
  lincs     --line L --pt FILE --out-dir D     raw per-line signatures -> D/raw_lincs_L.npz
  gw        --h5 FILE --out-dir D              raw pseudobulk signatures -> D/raw_k562gw.npz
  assemble  --src-dir D --cache-dir C --gene2go F
            every raw_*.npz -> D/cache_<name>.npz, plus D/src_check.json: coverage of
            each target's candidates on every split, the signal ceiling, and the
            gene-level transfer diagnostic (does the source's gene-gene similarity
            rank T's gene pairs the way T's own signatures do?), with K562<->RPE1 as
            the reference pair whose simulator result is known.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own scripts/ only
import numpy as np  # noqa: E402

import rl_headroom as RH  # noqa: E402

TARGETS = ("replogle_k562_essential", "replogle_rpe1_essential")
DEV_SPLITS = (4, 104, 204)
HELD_SPLITS = tuple(range(304, 1705, 100))
NEW_SPLITS = tuple(range(1804, 3305, 100))   # never used by any job before 2026-11-25
SAME_LINE = {"k562gw": "replogle_k562_essential"}   # source -> the target it may NOT serve
log = RH.log


def canon_sym(s):
    return str(s).strip().upper()


# ================================================================ LINCS (PDGrapher)
def load_pdg(path):
    """PDGrapher's backward file is a pickled list of torch_geometric Data objects.
    It is downloaded data, so it is loaded with weights_only=True and only the
    torch_geometric container classes allow-listed: anything else in the pickle is
    refused instead of executed."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
    return torch.load(path, weights_only=True, map_location="cpu")


def lincs_raw(line, pt, out_dir, seed=0):
    import torch  # noqa: F401
    t0 = time.time()
    log(f"loading {pt}")
    data = load_pdg(pt)
    log(f"  {len(data)} samples in {time.time() - t0:.0f}s")
    genes = [canon_sym(g) for g in data[0].gene_symbols]
    names = np.array([canon_sym(d.perturbagen_name) for d in data], dtype=object)
    T = np.stack([d.treated.numpy().astype(np.float32) for d in data])
    D = np.stack([d.diseased.numpy().astype(np.float32) for d in data])
    del data
    rng = np.random.default_rng(seed)
    # controls: the distinct pre-intervention profiles (PDGrapher cycles them), split in half
    Du = np.unique(D, axis=0)
    rng.shuffle(Du)
    h = Du.shape[0] // 2
    muA, muB, mu = Du[:h].mean(0), Du[h:].mean(0), Du.mean(0)
    uniq, inv = np.unique(names, return_inverse=True)
    Y, Q, owner, nrep, cands = [], [], [], [], []
    for i, g in enumerate(uniq):
        r = np.flatnonzero(inv == i)
        if r.size < 2:          # one sample cannot give both a signature and a held-out query
            continue
        rng.shuffle(r)
        hh = r.size // 2
        k = len(cands)
        cands.append(str(g))
        Y.append(T[r].mean(0) - mu)
        for j in r[hh:]:
            Q.append(T[j] - muB)
            owner.append(k)
        nrep.append(r.size)
    out = os.path.join(out_dir, f"raw_lincs_{line}.npz")
    np.savez(out, name=np.array(f"lincs_{line}"), Y=np.array(Y, np.float32), Q=np.array(Q, np.float32),
             owner=np.array(owner, np.int64), cands=np.array(cands, dtype=str),
             readout=np.array(genes, dtype=str), nrep=np.array(nrep, np.int64),
             self_query=np.array(False))
    log(f"  lincs_{line}: {len(cands)} genes with >=2 samples (of {uniq.size}), {len(Q)} queries, "
        f"{len(genes)} readout, {Du.shape[0]} distinct controls -> {out}")
    return 0


# ================================================================ Replogle genome-wide
def gw_raw(h5, out_dir):
    import anndata as ad
    log(f"loading {h5}")
    A = ad.read_h5ad(h5)
    log(f"  obs {A.shape[0]} x var {A.shape[1]}; obs columns {list(A.obs.columns)[:20]}; "
        f"var columns {list(A.var.columns)[:10]}; obs_names[:3] {list(A.obs_names[:3])}")
    if "gene" in A.obs:
        pert = A.obs["gene"].astype(str).values
    else:                      # e.g. "10000_AAAS_P1P2_ENSG00000094914"
        pert = np.array([s.split("_")[1] if "_" in s else s for s in A.obs_names.astype(str)])
    pert = np.array([canon_sym(p) for p in pert], dtype=object)
    readout = (A.var["gene_name"].astype(str).values if "gene_name" in A.var else A.var_names.astype(str).values)
    X = RH.densify(A.X)
    # the authors' z-scoring leaves NaN where a readout gene has zero control variance in
    # a gem group; a NaN poisons every cosine it touches, so it reads as "no change"
    n_nan = int(np.isnan(X).sum())
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    log(f"  {n_nan} NaN entries ({n_nan / X.size:.2%}) set to 0")
    keep = ~np.isin(pert, ["NON-TARGETING", "NONTARGETING", "CTRL", "CONTROL"])
    uniq, inv = np.unique(pert[keep], return_inverse=True)
    Xk = X[keep]
    Y = np.zeros((uniq.size, X.shape[1]), np.float32)
    np.add.at(Y, inv, Xk)
    cnt = np.bincount(inv, minlength=uniq.size).astype(np.float32)
    Y /= cnt[:, None]          # several transcripts of one gene: their mean
    out = os.path.join(out_dir, "raw_k562gw.npz")
    np.savez(out, name=np.array("k562gw"), Y=Y, Q=Y.copy(), owner=np.arange(uniq.size, dtype=np.int64),
             cands=np.array([str(u) for u in uniq], dtype=str),
             readout=np.array([canon_sym(r) for r in readout], dtype=str), nrep=cnt.astype(np.int64),
             self_query=np.array(True))
    log(f"  k562gw: {uniq.size} genes ({int((cnt > 1).sum())} with >1 transcript), "
        f"{int((~keep).sum())} control rows dropped, {len(readout)} readout -> {out}")
    return 0


# ================================================================ assemble + probe
def split_of(K, seed):
    perm = np.random.default_rng(seed).permutation(K)
    n_rep = max(60, K // 6)
    n_act = min((K - n_rep) // 2, 437)
    return {"rep": np.sort(perm[:n_rep]), "act": np.sort(perm[n_rep:n_rep + n_act]),
            "aux": np.sort(perm[n_rep + n_act:])}


def pair_spearman(A, B, max_pairs=200000, seed=0):
    """Spearman over gene pairs between two gene x gene similarity matrices."""
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    a, b = A[iu], B[iu]
    if a.size > max_pairs:
        s = np.random.default_rng(seed).choice(a.size, max_pairs, replace=False)
        a, b = a[s], b[s]
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def assemble(src_dir, cache_dir, gene2go):
    check = json.load(open(os.path.join(cache_dir, "cache_check.json")))
    tz = {t: np.load(check["datasets"][t]["path"]) for t in TARGETS}
    tc = {t: [canon_sym(c) for c in tz[t]["cands"]] for t in TARGETS}
    universe = set(tc[TARGETS[0]]) | set(tc[TARGETS[1]])
    g2g = RH.load_gene2go(gene2go)
    g2g = {canon_sym(k): v for k, v in g2g.items()}
    # T's own gene-gene cosine, for the transfer diagnostic
    tcc = {}
    for t in TARGETS:
        U = RH.unit(tz[t]["Y"])
        tcc[t] = U @ U.T
    out = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "universe": len(universe), "sources": {}}
    srcs = []
    for p in sorted(glob.glob(os.path.join(src_dir, "raw_*.npz"))):
        z = np.load(p)
        srcs.append((str(z["name"]), p))
    # reference: the two Replogle lines as each other's source (the known result)
    srcs += [(t, check["datasets"][t]["path"]) for t in TARGETS]
    for name, p in srcs:
        z = np.load(p)
        ref = name in TARGETS
        cands = [canon_sym(c) for c in z["cands"]]
        if ref:
            Y, Q, owner, cpath = z["Y"], z["Q"], z["owner"], p
            keepc = np.arange(len(cands))
        else:
            keepc = np.array([i for i, c in enumerate(cands) if c in universe], np.int64)
            kc = [cands[i] for i in keepc]
            targets_ = set(kc)
            mask = np.array([g not in targets_ for g in z["readout"]])
            Y = z["Y"][keepc][:, mask]
            qk = np.isin(z["owner"], keepc)
            remap = -np.ones(len(cands), np.int64)
            remap[keepc] = np.arange(keepc.size)
            Q, owner = z["Q"][qk][:, mask], remap[z["owner"][qk]]
            F = RH.go_features(kc, g2g)
            cpath = os.path.join(src_dir, f"cache_{name}.npz")
            np.savez(cpath, name=np.array(name), Y=Y.astype(np.float32), Q=Q.astype(np.float32),
                     owner=owner.astype(np.int64), F=(F > 0).astype(np.uint8), cands=np.array(kc, dtype=str))
            cands = kc
        ceil = RH.score(Y, Q, owner)
        info = {"path": cpath, "K": len(cands), "readout_kept": int(Y.shape[1]), "n_queries": int(Q.shape[0]),
                "self_query": bool(z["self_query"]) if "self_query" in z.files else False,
                "ceiling_top1": ceil["top1"], "ceiling_mrr": ceil["mrr"], "chance_mrr": ceil["chance_mrr"],
                "targets": {}, "same_line": {}}
        spos = {c: i for i, c in enumerate(cands)}
        U = RH.unit(Y)
        for t in TARGETS:
            if t == name:
                continue
            # a same-line source is never a simulator for that target; its numbers are
            # written anyway, marked, as the sanity check that the source is read right
            same_line = SAME_LINE.get(name) == t
            idx_t = np.array([i for i, c in enumerate(tc[t]) if c in spos], np.int64)
            idx_s = np.array([spos[tc[t][i]] for i in idx_t], np.int64)
            K = len(tc[t])
            cov = {}
            for grp, sps in (("dev", DEV_SPLITS), ("held", HELD_SPLITS), ("new", NEW_SPLITS)):
                fr = [float(np.isin(split_of(K, s)["act"], idx_t).mean()) for s in sps]
                cov[grp] = {"mean": float(np.mean(fr)), "min": float(np.min(fr)), "max": float(np.max(fr))}
            scc = U[idx_s] @ U[idx_s].T
            info["same_line" if same_line else "targets"][t] = {"n_shared": int(idx_t.size), "frac_of_T": float(idx_t.size / K),
                                  "act_coverage": cov,
                                  "pair_spearman_vs_T": pair_spearman(tcc[t][np.ix_(idx_t, idx_t)], scc)}
        out["sources"][name] = info
        log(f"{name}: K={info['K']} ceil mrr={info['ceiling_mrr']:.3f} (chance {info['chance_mrr']:.4f}) "
            + "; ".join(f"{t.split('_')[1]}{' (SAME LINE, sanity)' if t in info['same_line'] else ''}: cov {v['act_coverage']['dev']['mean']:.2f}, "
                        f"pairρ {v['pair_spearman_vs_T']:+.3f}" for t, v in {**info["targets"], **info["same_line"]}.items()))
        with open(os.path.join(src_dir, "src_check.json"), "w") as f:
            json.dump(out, f, indent=2)
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(src_dir, "src_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("lincs", "gw", "assemble"))
    ap.add_argument("--line")
    ap.add_argument("--pt")
    ap.add_argument("--h5")
    ap.add_argument("--out-dir")
    ap.add_argument("--src-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--gene2go")
    a = ap.parse_args()
    if a.mode == "lincs":
        os.makedirs(a.out_dir, exist_ok=True)
        return lincs_raw(a.line, a.pt, a.out_dir)
    if a.mode == "gw":
        os.makedirs(a.out_dir, exist_ok=True)
        return gw_raw(a.h5, a.out_dir)
    return assemble(a.src_dir, a.cache_dir, a.gene2go)


if __name__ == "__main__":
    sys.exit(main())
