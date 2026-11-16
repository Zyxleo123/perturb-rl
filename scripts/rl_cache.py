#!/usr/bin/env python
"""Cache the inputs of the gene-acquisition MDP for scripts/rl_acquire.py.

Builds each Replogle dataset EXACTLY as scripts/rl_headroom.py does -- the same
function, the same seed -- so every RL number downstream is computed on the same
signatures, queries and GO features as GAPS.md "Is the unseen-gene gap RL-shaped?".
It then re-derives part 4's deployment split and its random x100 distribution
through rl_headroom's own code path and checks the median against the value
GAPS.md reports. If the cache does not reproduce that number, nothing built on it
is comparable, and the check says so.

Writes <out>/rl_acquire/cache_<dataset>.npz and <out>/rl_acquire/cache_check.json.
Always exits 0 after writing the check; its per-dataset `ok` is the gate the RL
jobs read.

Usage: python -I rl_cache.py --gears-dir D --gene2go F --out-dir D
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own scripts/ only
import numpy as np  # noqa: E402

import rl_headroom as RH  # noqa: E402

# artifacts/rl_headroom/reward_noise.jsonl, strategy=random, B=100, 100 draws
EXPECTED_RANDOM_MEDIAN = {"replogle_k562_essential": 0.17811616476089143,
                          "replogle_rpe1_essential": 0.1008751323373303}
DATASETS = tuple(EXPECTED_RANDOM_MEDIAN)


def part4_split(K, seed):
    """rl_headroom.reward_noise_headroom's split; seed 4 is the one GAPS.md reports."""
    perm = np.random.default_rng(seed).permutation(K)
    n_rep = max(60, K // 6)
    n_act = min((K - n_rep) // 2, 437)
    return (np.sort(perm[:n_rep]), np.sort(perm[n_rep:n_rep + n_act]),
            np.sort(perm[n_rep + n_act:]))


def random_median(Y, Q, owner, F, rep, act, B=100, n=100, seed=9100):
    """Part 4's random x100, line for line (rl_headroom.py l.618-625)."""
    m = np.isin(owner, rep)
    remap = {o: i for i, o in enumerate(rep)}
    Q_rep, o_rep = Q[m], np.array([remap[o] for o in owner[m]])
    J_rep, Y_act = RH.jaccard(F[rep], F[act]), Y[act]
    srng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        ch = srng.choice(act.size, size=B, replace=False)
        P = RH.go_knn_predict(J_rep[:, ch], Y_act[ch], Y_act[ch].mean(0))
        v.append(RH.score(P, Q_rep, o_rep)["mrr"])
    return float(np.median(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gears-dir", required=True)
    ap.add_argument("--gene2go", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    a = ap.parse_args()

    od = os.path.join(a.out_dir, "rl_acquire")
    os.makedirs(od, exist_ok=True)
    check = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "datasets": {}, "errors": []}

    def dump():
        with open(os.path.join(od, "cache_check.json"), "w") as f:
            json.dump(check, f, indent=2)

    g2g = RH.load_gene2go(a.gene2go)
    RH.log(f"gene2go: {len(g2g)} genes")
    for name in a.datasets:
        try:
            D = RH.build(name, os.path.join(a.gears_dir, name, "perturb_processed.h5ad"),
                         np.random.default_rng(RH.SEED))
            F = RH.go_features(D["cands"], g2g)
            mk = D["mask"]
            Y, YA, Q, owner = D["dF"][:, mk], D["dA"][:, mk], D["Q"][:, mk], D["owner"]
            ceil = RH.score(YA, Q, owner)
            rep, act, aux = part4_split(len(D["cands"]), RH.SEED + 4)
            med = random_median(Y, Q, owner, F, rep, act)
            exp = EXPECTED_RANDOM_MEDIAN.get(name)
            ok = exp is not None and abs(med - exp) < 1e-6
            path = os.path.join(od, f"cache_{name}.npz")
            np.savez(path, name=np.array(name), Y=Y.astype(np.float32), Q=Q.astype(np.float32),
                     owner=owner.astype(np.int64), F=(F > 0).astype(np.uint8),
                     cands=np.array(D["cands"], dtype=str))
            check["datasets"][name] = {
                "ok": bool(ok), "path": path, "K": int(len(D["cands"])), "genes_masked": int(mk.sum()),
                "go_terms": int(F.shape[1]), "n_queries": int(Q.shape[0]),
                "split4_sizes": {"rep": int(rep.size), "act": int(act.size), "aux": int(aux.size)},
                "random_median_reproduced": med, "random_median_gaps": exp,
                "ceiling_top1_allK": ceil["top1"], "ceiling_mrr_allK": ceil["mrr"]}
            RH.log(f"  {name}: random median {med:.6f} vs GAPS {exp} -> {'OK' if ok else 'MISMATCH'}")
            del D, F, Y, YA, Q
        except Exception as e:
            traceback.print_exc()
            check["errors"].append(f"{name}: {type(e).__name__}: {e}")
        dump()
    check["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump()
    RH.log("errors:", check["errors"] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
