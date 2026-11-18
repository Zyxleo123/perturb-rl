#!/usr/bin/env python
"""Published acquisition methods on the RL acquisition task, number for number.

The task is GAPS.md "RL on the acquisition gap" exactly: choose 100 of 437 genes,
build the GO-kNN predictor from the chosen genes, score retrieval MRR on report genes
no method sees -- the same splits (rl_acquire.split_of), the same budget spent in 4
rounds of 25, the same Reporter, and rl_report re-scores every chosen set in the
analysis env beside the RL runs. Only the selection rule changes.

Two information regimes, the RL write-up's own (rl_acquire.py docstring):
  blind  a candidate's signature is revealed only once it is measured. A blind method
         reads GO, prior-knowledge kernels, the visible atlas (aux) and the genes it
         has itself measured. It receives a signature matrix whose unmeasured rows are
         NaN, so reading a hidden signature cannot help it silently: it fails loudly.
  peek   every non-report signature is visible, the information greedy and REINFORCE
         use. Report rows are NaN here, for the same reason.

blind -- IterPert and the benchmark it was published against, through IterPert's OWN
code (iterpert/bmdal, its fork of bmdal_reg), configured line for line as
iterpert/iterpert.py initialize_active_learning_strategy():
  iterpert       IterPert (Huang, Lopez, Hutter, Kudo, Rios, Regev; RECOMB 2024).
                 MaxDist on the mean of max-normalised kernels: its released prior
                 kernels (PoPS, ESM, BioGPT, node2vec on the GEARS GO graph, three
                 optical pooled screens, RPE1 Perturb-seq) plus the model's own kernel.
  iterpert_go    the same algorithm with GO Jaccard as the only prior: separates the
                 selection rule from the priors.
  iterpert_cold  IterPert without the random initial batch (see protocol below).
  al_coreset     Core-Set, k-centre greedy (Sener & Savarese, ICLR 2018)
  al_bald        BALD as max posterior variance (Houlsby et al. 2011)
  al_batchbald   BatchBALD as max-determinant (Kirsch et al., NeurIPS 2019)
  al_badge       BADGE, k-means++ seeding (Ash et al., ICLR 2020)
  al_lcmd        LCMD (Holzmuller et al., JMLR 2023)
  al_acsfw       ACS-FW, Frank-Wolfe coreset (Pinsler et al., NeurIPS 2019)
  al_kmeans      k-means sampling (GeneDisco, Mehrjou et al., ICLR 2022; IterPert's
                 KMeansSampling, ported)
  al_typiclust   TypiClust (Hacohen et al., ICML 2022), IterPert's variant, ported
  fl_iterpert_priors  the ablation that separates IterPert's PRIORS from its RULE: our
                 static floor (greedy facility location, act + aux targets) on the mean
                 of IterPert's prior kernels, each cosine-normalised (facility location
                 needs comparable similarities; IterPert's 'max' normalisation keeps the
                 norms), instead of GO Jaccard.
  kc_iterpert_priors / kc_go  IterPert's own rule (MaxDist = k-centre on the mean of
                 max-normalised kernels, first pick the largest diagonal) on its priors /
                 on GO alone, with NO model kernel and NO random batch (IterPert's
                 use_prior_only), so it is static like facility location. A static
                 k-centre spent in 4 rounds of 25 is the same 100 sequential picks, so it
                 runs as one. Together with go_facility and fl_iterpert_priors this is a
                 clean 2 x 2: {k-centre, facility location} x {GO, IterPert's priors}.
  BAIT is omitted: iterpert.py marks it "there is bug somehow". ProbCover is omitted:
  its radius is set by k-means purity against the number of CLASSES, which this
  regression task does not have, and its soft relative is GO facility location, which
  is already the floor in every table.

  The model. IterPert's kernels are built on the GEARS model's per-perturbation output.
  Here the model is the deployed predictor's own rule: a gene is represented by its
  GO-kNN predicted signature from every MEASURED gene (aux + chosen so far), leave-one-
  out, refitted after every round as measurements come back.

  Protocol. IterPert's benchmark starts every method from a random labelled set and then
  runs acquisition rounds (iterpert.py start(n_init_labeled, n_round, n_query)). Scaled
  to this budget: round 1 = 25 random candidates, rounds 2-4 = 25 by the method. BALD,
  BatchBALD and BADGE cannot run without labelled genes (their posterior transform
  scales by the mean training-kernel diagonal), and bmdal's linear kernel indexes row 0
  of the training set, so this is also the only protocol the reference code supports.
  TypiClust is a cold-start method by design and selects all four rounds itself.
  iterpert_cold measures the cost of the random batch for the headline method; its
  first round uses a numpy replica of IterPert's MaxDist (largest prior-kernel diagonal,
  then farthest point), and every later round runs IterPert's own code with the replica
  checked against it.

peek -- data valuation, the standard answer to "which training points make a
predictor good", on the very reward greedy and REINFORCE optimise (MRR on aux), with
REINFORCE's budget of reward evaluations (600 iterations x 32 sampled sets):
  dv_shapley     TMC Data Shapley (Ghorbani & Zou, ICML 2019), 1% truncation
  dv_banzhaf     Data Banzhaf, maximum-sample-reuse estimator (Wang & Jia, AISTATS 2023),
                 published as the value notion robust to a noisy utility
  dv_datamodels  datamodels (Ilyas et al., ICML 2022): l1 linear surrogate of the reward
                 on random sets of the deployment size, alpha by 5-fold CV
  each keeps the top 100 values. And two static exemplar methods on the signatures:
  rep_facility_sig  facility location / exemplar summarisation on signature cosine
                    (Gomes & Krause 2010; Mirzasoleiman et al. 2013), targets act + aux
  rep_coreset_sig   k-centre greedy on the true signatures (Core-Set with oracle features)

Usage:
  python -I related_work.py prep --zip Z --cache-dir D --out-dir D
  python -I related_work.py run  --cache-dir D --out-dir D --code-dir D --task-index I
  python -I related_work.py grid
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import pickle
import sys
import time
import traceback
import types
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own scripts/ only
import numpy as np  # noqa: E402

import rl_acquire as RA  # noqa: E402
import rl_headroom as RH  # noqa: E402

B = RA.B_DEPLOY
SMOKE = bool(os.environ.get("PJ_REL_SMOKE"))
# REINFORCE (rl_acquire.train_pg_tab) samples 32 sets per iteration for 600 iterations
N_EVAL = int(os.environ.get("PJ_REL_NEVAL") or (300 if SMOKE else 600 * 32))
OUT_SUB = "related_smoke" if SMOKE else "related"

# name: regime, family, deterministic, random initial batch, citation
METHODS = {
    "iterpert":         ("blind", "iterpert", False, True, "Huang et al., RECOMB 2024"),
    "iterpert_go":      ("blind", "iterpert", False, True, "Huang et al. 2024, GO prior only"),
    "iterpert_cold":    ("blind", "iterpert", True, False, "Huang et al. 2024, no random batch"),
    "fl_iterpert_priors": ("blind", "static", True, False, "Huang et al. 2024 priors, facility location"),
    "kc_iterpert_priors": ("blind", "static", True, False, "Huang et al. 2024 rule + priors, static"),
    "kc_go":            ("blind", "static", True, False, "Huang et al. 2024 rule, GO prior, static"),
    "al_coreset":       ("blind", "bmdal", False, True, "Sener & Savarese, ICLR 2018"),
    "al_bald":          ("blind", "bmdal", False, True, "Houlsby et al. 2011"),
    "al_batchbald":     ("blind", "bmdal", False, True, "Kirsch et al., NeurIPS 2019"),
    "al_badge":         ("blind", "bmdal", False, True, "Ash et al., ICLR 2020"),
    "al_lcmd":          ("blind", "bmdal", False, True, "Holzmuller et al., JMLR 2023"),
    "al_acsfw":         ("blind", "bmdal", False, True, "Pinsler et al., NeurIPS 2019"),
    "al_kmeans":        ("blind", "kmeans", False, True, "Mehrjou et al. (GeneDisco), ICLR 2022"),
    "al_typiclust":     ("blind", "typiclust", False, False, "Hacohen et al., ICML 2022"),
    "dv_shapley":       ("peek", "valuation", False, False, "Ghorbani & Zou, ICML 2019"),
    "dv_banzhaf":       ("peek", "valuation", False, False, "Wang & Jia, AISTATS 2023"),
    "dv_datamodels":    ("peek", "valuation", False, False, "Ilyas et al., ICML 2022"),
    "rep_facility_sig": ("peek", "exemplar", True, False, "Gomes & Krause 2010; Mirzasoleiman et al. 2013"),
    "rep_coreset_sig":  ("peek", "exemplar", True, False, "Sener & Savarese 2018, oracle features"),
}

# iterpert/iterpert.py initialize_active_learning_strategy(), verbatim:
#   name: (selection_method, kernel_transforms, sel_with_train)
BMDAL_CFG = {
    "al_bald": ("maxdiag", [("train", [0.1, None])], False),
    "al_batchbald": ("maxdet", [("train", [0.1, None])], False),
    "al_acsfw": ("fw", [("acs-rf", [512, 0.1, None])], False),
    "al_coreset": ("maxdist", [], True),
    "al_badge": ("kmeanspp", [("train", [0.1, None])], False),
    "al_lcmd": ("lcmd", [("rp", [512])], True),
}

# IterPert's prior list for replogle_k562_essential_1000hvg (iterpert.py). RPE1 has
# none released; there the same recipe is applied to the same embeddings (the authors'
# own mean imputation for genes a source lacks), WITHOUT rpe1_kernel -- on RPE1 that is
# the deployment screen's own signatures, i.e. the answer -- and WITH the symmetric
# prior IterPert's K562 run gets from RPE1: the other line's Perturb-seq, built from our
# K562 cache by the same recipe (feat = signature, kernel = feat feat^T).
ITERPERT_PRIORS = ["pops_kernel", "rpe1_kernel", "esm_kernel", "biogpt_kernel", "node2vec_kernel",
                   "ops_A549_kernel", "ops_HeLa_HPLM_kernel", "ops_HeLa_DMEM_kernel"]
K562, RPE1 = "replogle_k562_essential", "replogle_rpe1_essential"


def log(*a):
    RH.log(*a)


# ================================================================= prep (job 44)
class NumpyOnlyUnpickler(pickle.Unpickler):
    """The kernel zip is third-party pickles. Admit numpy array reconstruction and
    nothing else: anything that names another global is refused, not executed."""
    OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
          ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
          ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
          ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.OK:
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def zload(z, member):
    with z.open(member) as f:
        return NumpyOnlyUnpickler(io.BytesIO(f.read())).load()


def aligned_feat(names_src, F_src, cands):
    """IterPert's own recipe (demo/knowledge_kernels_process.ipynb): a gene's row of the
    source embedding, the source's mean embedding when the source lacks the gene."""
    pos = {str(g): i for i, g in enumerate(names_src)}
    F_src = np.asarray(F_src, np.float64)
    mean = F_src.mean(0)
    hit = np.array([str(g) in pos for g in cands])
    F = np.stack([F_src[pos[str(g)]] if h else mean for g, h in zip(cands, hit)])
    return F, float(hit.mean())


def prep(a):
    od = os.path.join(a.out_dir, "rl_acquire", "related")
    os.makedirs(od, exist_ok=True)
    check = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "zip": a.zip, "priors": {}, "datasets": {},
             "errors": []}

    def dump():
        with open(os.path.join(od, "priors_check.json"), "w") as f:
            json.dump(check, f, indent=2)

    cc = json.load(open(os.path.join(a.cache_dir, "cache_check.json")))
    cands = {ds: np.load(cc["datasets"][ds]["path"])["cands"] for ds in (K562, RPE1)}
    z = zipfile.ZipFile(a.zip)
    members = z.namelist()
    log(f"zip: {len(members)} members, e.g. {members[:4]}")
    src = {}
    for k in ITERPERT_PRIORS:
        try:
            base = [m for m in members if m.rstrip("/").endswith(f"{k}/feat.pkl")]
            if len(base) != 1:
                raise FileNotFoundError(f"{k}/feat.pkl: {len(base)} matches")
            d = base[0][:-len("feat.pkl")]
            names = [str(x) for x in zload(z, d + "pert_list.pkl")]
            F = np.asarray(zload(z, d + "feat.pkl"), np.float64)
            Kr = np.asarray(zload(z, d + "kernel.pkl"), np.float64)
            # the released kernel must be feat feat^T, or rebuilding it on our genes from
            # feat is not the same prior
            rel = float(np.abs(F @ F.T - Kr).max() / max(np.abs(Kr).max(), 1e-12))
            src[k] = (names, F)
            check["priors"][k] = {"n_genes": len(names), "dim": int(F.shape[1]),
                                  "kernel_is_feat_featT_max_rel_err": rel}
            log(f"  {k}: {len(names)} genes x {F.shape[1]} dims, |FF^T - K|/|K| = {rel:.2e}")
        except Exception as e:
            traceback.print_exc()
            check["errors"].append(f"{k}: {type(e).__name__}: {e}")
        dump()
    yk = np.load(cc["datasets"][K562]["path"])["Y"]
    for ds in (K562, RPE1):
        names, mats, cov = [], [], {}
        for k in ITERPERT_PRIORS:
            if k not in src or (ds == RPE1 and k == "rpe1_kernel"):
                continue
            F, c = aligned_feat(*src[k], cands[ds])
            names.append(k)
            mats.append((F @ F.T).astype(np.float32))
            cov[k] = c
        if ds == RPE1:
            F, c = aligned_feat(cands[K562], yk, cands[ds])
            names.append("k562_perturbseq_kernel")
            mats.append((F @ F.T).astype(np.float32))
            cov["k562_perturbseq_kernel"] = c
        path = os.path.join(od, f"priors_{ds}.npz")
        np.savez(path, names=np.array(names), kernels=np.stack(mats), cands=cands[ds])
        # IterPert as published needs all 8 priors on K562; RPE1 is our extension
        ok = len(names) == len(ITERPERT_PRIORS) and not check["errors"]
        check["datasets"][ds] = {"ok": bool(ok), "path": path, "priors": names, "coverage": cov}
        log(f"  {ds}: {len(names)} priors, coverage {json.dumps({k: round(v, 3) for k, v in cov.items()})}")
        dump()
    check["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump()
    log("errors:", check["errors"] or "none")
    return 0


# ============================================================ the blind model view
class Measured:
    """Everything a blind method may read: GO, and the signatures of measured genes.
    Unmeasured rows are NaN, so any use of a hidden signature poisons the output."""

    def __init__(self, W, measured):
        self.W = W
        self.M = np.asarray(measured, np.int64)
        self.Yv = np.full(W.Y.shape, np.nan, np.float32)
        self.Yv[self.M] = W.Y[self.M]

    def view(self, genes):
        """GO-kNN predicted signature of each gene from the measured genes, leave-one-
        out: the deployed predictor's own rule is the model, and a measured gene is
        represented exactly as an unmeasured one would be."""
        g = np.asarray(genes, np.int64)
        J = self.W.JJ[np.ix_(g, self.M)].copy()
        J[g[:, None] == self.M[None, :]] = 0.0
        Ym = self.Yv[self.M]
        P = RH.go_knn_predict(J, Ym, Ym.mean(0))
        if not np.isfinite(P).all():
            raise RuntimeError("model view read an unmeasured signature")
        return P.astype(np.float32)


_IP = {}


def iterpert_api(code_dir, threads):
    """IterPert's bmdal fork, imported from the cloned repo. Paths are APPENDED, so
    numpy/torch/scipy always resolve to the environment and never to the clone; only
    the two packages absent from the env (iterpert, bmdal_reg) come from there. Its
    utils import `dill` for save/load helpers this script never calls; a stub stands in
    rather than adding a package to the shared env."""
    if not _IP:
        import torch
        torch.set_num_threads(threads)
        if "dill" not in sys.modules:
            d = types.ModuleType("dill")
            d.Unpickler = pickle.Unpickler

            def _no(*_a, **_k):
                raise RuntimeError("dill stub: serialisation is not used here")
            d.dump = d.load = d.dumps = d.loads = _no
            sys.modules["dill"] = d
        for sub in ("iterative-perturb-seq", "bmdal_reg"):
            p = os.path.join(code_dir, sub)
            if not os.path.isdir(p):
                raise FileNotFoundError(f"{p} (did jobs/14_fetch_related.sbatch run?)")
            sys.path.append(p)
        from iterpert.bmdal import algorithms as alg
        from iterpert.bmdal.feature_data import TensorFeatureData
        _IP.update(torch=torch, alg=alg, TFD=TensorFeatureData)
    return _IP


def bmdal_select(api, method, Xp, Xt, b, seed):
    torch = api["torch"]
    sm, tf, swt = BMDAL_CFG[method]
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)            # kmeanspp's first pick uses np.random
    data = {"train": api["TFD"](torch.from_numpy(Xt)), "pool": api["TFD"](torch.from_numpy(Xp))}
    idx, info = api["alg"].select_batch(batch_size=b, models=[torch.nn.Identity()], data=data, y_train=0,
                                        selection_method=sm, sel_with_train=swt, base_kernel="linear",
                                        kernel_transforms=tf, lamb=2)
    return idx.cpu().numpy().astype(np.int64), info.get("selection_status")


def _maxnorm(K):
    return K / np.max(K)


def maxdist_prior_replica(priors_sub, Xp, Xt, b):
    """IterPert's MaxDistSelectionMethodwithPrior on the mean_new / 'max' aggregate,
    in numpy: used for the cold first round (bmdal cannot take an empty training set)
    and as a check on every round IterPert's own code runs."""
    X = np.concatenate([Xp, Xt]).astype(np.float32)
    ks = [_maxnorm(np.asarray(P, np.float64)) for P in priors_sub] + [_maxnorm((X @ X.T).astype(np.float64))]
    K = np.mean(ks, 0)
    npool = Xp.shape[0]
    d = np.diag(K)
    md = np.full(npool, np.inf)
    for i in range(npool, K.shape[0]):
        md = np.minimum(md, d[i] + d[:npool] - 2 * K[i, :npool])
    sel = []
    for _ in range(b):
        if K.shape[0] == npool and not sel:
            j = int(np.argmax(d[:npool]))
        else:
            s = md.copy()
            s[sel] = -np.inf
            j = int(np.argmax(s))
        sel.append(j)
        md = np.minimum(md, d[j] + d[:npool] - 2 * K[j, :npool])
    return np.array(sel, np.int64)


def iterpert_select(api, priors, pool, train, Xp, Xt, b):
    """kernel_based_active_learning.query() with prior kernels: reindex every prior to
    [pool, train], then BatchSelectorImpl(..., integrate_mode='mean_new',
    normalize_mode='max').select('maxdist_prior', sel_with_train=True)."""
    re = np.concatenate([pool, train])
    sub = [P[np.ix_(re, re)] for P in priors]
    rep = maxdist_prior_replica(sub, Xp, Xt, b)
    if len(train) == 0:
        return rep, {"replica_used": True}
    torch = api["torch"]
    data = {"train": api["TFD"](torch.from_numpy(Xt)), "pool": api["TFD"](torch.from_numpy(Xp))}
    bs = api["alg"].BatchSelectorImpl([torch.nn.Identity()], data, 0, train_gold=None,
                                      prior_kernel_list=[np.array(s, np.float32) for s in sub],
                                      use_prior_only=False, integrate_mode="mean_new", normalize_mode="max",
                                      valid_perts=None, lamb=2, round=None)
    idx, info = bs.select(selection_method="maxdist_prior", sel_with_train=True, base_kernel="linear",
                          kernel_transforms=[], batch_size=b)
    idx = idx.cpu().numpy().astype(np.int64)
    return idx, {"replica_used": False, "replica_overlap": float(np.isin(idx, rep).mean()),
                 "selection_status": info.get("selection_status")}


def kmeans_labels(E, k, seed):
    """TypiClust's (and IterPert's copy of it) clustering, including its switch to
    mini-batch k-means above 50 clusters; seeded here, unseeded there."""
    from sklearn.cluster import KMeans, MiniBatchKMeans
    km = (KMeans(n_clusters=k, random_state=seed) if k <= 50
          else MiniBatchKMeans(n_clusters=k, batch_size=5000, random_state=seed))
    return km.fit_predict(E)


def typiclust_select(E, labeled, b, seed):
    """iterpert/query_strategies/typiclust.py query(): cluster labelled + pool into
    |L| + b clusters, visit clusters by (fewest labelled, largest), take the most typical
    unlabelled member (1 / mean euclidean distance to its min(20, n/2) nearest). IterPert
    drops the original's minimum cluster size of 5, which with 437 genes and 100 clusters
    would discard most clusters; this follows IterPert."""
    from sklearn.neighbors import NearestNeighbors
    k = int(labeled.sum()) + b
    cl = kmeans_labels(E, k, seed).copy()
    sizes = np.bincount(cl, minlength=k)
    existing = np.bincount(cl[labeled], minlength=k)
    order = np.lexsort((-sizes, existing))
    cl[labeled] = -1
    sel, i = [], 0
    while len(sel) < b:
        c = order[i % k]
        idx = np.flatnonzero(cl == c)
        if idx.size:
            nn = min(20, idx.size // 2)
            if nn == 0:
                j = int(idx[0])
            else:
                d, _ = NearestNeighbors(n_neighbors=nn + 1).fit(E[idx]).kneighbors(E[idx])
                j = int(idx[np.argmax(1.0 / (d[:, 1:].mean(1) + 1e-5))])
            sel.append(j)
            cl[j] = -1
        i += 1
    return np.array(sel, np.int64)


def kmeans_select(Ep, b, seed, rng):
    """iterpert/query_strategies/kmeans_sampling.py (mode 'kmeans'): b clusters of the
    pool, the member nearest each centre, empty clusters filled at random."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=b, random_state=seed).fit(Ep)
    lab = km.predict(Ep)
    dis = ((Ep - km.cluster_centers_[lab]) ** 2).sum(1)
    q = [int(np.flatnonzero(lab == i)[np.argmin(dis[lab == i])]) for i in range(b) if (lab == i).any()]
    if len(q) < b:
        rest = np.setdiff1d(np.arange(Ep.shape[0]), q)
        q += [int(x) for x in rng.choice(rest, size=b - len(q), replace=False)]
    return np.array(q, np.int64)


def facility_prior(priors, dep):
    """Greedy facility location (rl_acquire.facility's rule) on the mean of the prior
    kernels, each turned into a cosine; negatives clipped, a gene not its own neighbour."""
    S = np.zeros(priors[0].shape, np.float64)
    for P in priors:
        P = np.asarray(P, np.float64)
        d = np.sqrt(np.maximum(np.diag(P), 1e-24))
        S += P / d[:, None] / d[None, :]
    S /= len(priors)
    A, T = dep["act"], np.concatenate([dep["act"], dep["aux"]])
    M = np.maximum(S[np.ix_(T, A)], 0.0)
    M[T[:, None] == A[None, :]] = 0.0
    s, tm = np.zeros(T.size), np.ones(T.size)
    avail = np.ones(A.size, bool)
    chosen = []
    for _ in range(B):
        av = np.flatnonzero(avail)
        k = int(av[np.argmax((tm[:, None] * np.maximum(M[:, av] - s[:, None], 0)).sum(0))])
        avail[k] = False
        tm[T == A[k]] = 0.0
        s = np.maximum(s, M[:, k])
        chosen.append(int(A[k]))
    return chosen


def kcenter_prior(priors, dep):
    """maxdist_prior_replica with use_prior_only: no data-driven kernel, pool = act."""
    K = np.mean([_maxnorm(np.asarray(P, np.float64)) for P in priors], 0)
    A = dep["act"]
    K = K[np.ix_(A, A)]
    d = np.diag(K)
    md = np.full(A.size, np.inf)
    sel = [int(np.argmax(d))]
    for _ in range(B - 1):
        j = sel[-1]
        md = np.minimum(md, d[j] + d - 2 * K[j])
        s = md.copy()
        s[sel] = -np.inf
        sel.append(int(np.argmax(s)))
    return [int(A[i]) for i in sel]


def run_blind(W, dep, method, seed, priors, api):
    regime, fam, _det, init_random, _ = METHODS[method]
    rng = np.random.default_rng(10_000 + seed)
    act, aux = dep["act"], dep["aux"]
    chosen, rounds = [], []
    for rd, b in enumerate(RA.round_sizes(B)):
        pool = np.setdiff1d(act, chosen)
        train = np.asarray(chosen, np.int64)
        info = {"round": rd, "by": "random" if (rd == 0 and init_random) else method}
        if rd == 0 and init_random:
            pick = rng.choice(pool, size=b, replace=False)
        else:
            mv = Measured(W, np.concatenate([aux, train]))
            if fam == "typiclust":
                lab = np.isin(act, train)
                loc = typiclust_select(mv.view(act), lab, b, seed * 100 + rd)
                pick = act[loc]
            else:
                Xp = mv.view(pool)
                Xt = mv.view(train) if train.size else np.zeros((0, Xp.shape[1]), np.float32)
                if fam == "bmdal":
                    loc, st = bmdal_select(api, method, Xp, Xt, b, seed * 100 + rd)
                    info["selection_status"] = st
                elif fam == "kmeans":
                    loc = kmeans_select(Xp, b, seed * 100 + rd, rng)
                elif fam == "iterpert":
                    loc, extra = iterpert_select(api, priors, pool, train, Xp, Xt, b)
                    info.update(extra)
                else:
                    raise ValueError(fam)
                pick = pool[loc]
        pick = [int(x) for x in pick]
        if len(set(pick)) != b or set(pick) & set(chosen) or not set(pick) <= set(act.tolist()):
            raise RuntimeError(f"round {rd}: invalid batch (dupes / repeats / outside act)")
        chosen += pick
        rounds.append(info)
    return chosen, {"rounds": rounds}


# ======================================================================== peek
def top_b(values, act, b=B):
    return [int(x) for x in act[np.argsort(-values, kind="stable")[:b]]]


def tmc_shapley(rs, act, n_eval, rng, tol=0.01):
    """Ghorbani & Zou 2019, Algorithm 1: walk random permutations, credit each gene its
    marginal gain, stop scoring a permutation once within `tol` of the full-set value."""
    nA = act.size
    v_full = rs.mrr(act)
    phi, cnt = np.zeros(nA), np.zeros(nA)
    ev, n_perm, n_trunc = 1, 0, 0
    while ev < n_eval:
        perm = rng.permutation(nA)
        prev = rs.chance_mrr                  # the empty predictor scores chance
        n_perm += 1
        for j in range(nA):
            if abs(v_full - prev) < tol * abs(v_full):
                m = 0.0
                n_trunc += 1
            else:
                cur = rs.mrr(act[perm[:j + 1]])
                ev += 1
                m, prev = cur - prev, cur
            phi[perm[j]] += m
            cnt[perm[j]] += 1
            if ev >= n_eval:
                break
    return phi / np.maximum(cnt, 1), {"n_eval": ev, "n_perm": n_perm, "v_full": v_full,
                                      "frac_truncated": n_trunc / max(cnt.sum(), 1),
                                      "min_samples_per_gene": int(cnt.min())}


def banzhaf_msr(rs, act, n_eval, rng):
    """Wang & Jia 2023, MSR: sample sets with every gene in w.p. 1/2; a gene's value is
    mean utility of sets containing it minus mean utility of sets that do not."""
    nA = act.size
    si, so, ni, no = np.zeros(nA), np.zeros(nA), np.zeros(nA), np.zeros(nA)
    for _ in range(n_eval):
        m = rng.random(nA) < 0.5
        u = rs.mrr(act[m]) if m.any() else rs.chance_mrr
        si += m * u
        ni += m
        so += (~m) * u
        no += ~m
    return si / np.maximum(ni, 1) - so / np.maximum(no, 1), {"n_eval": n_eval}


def datamodels(rs, act, n_eval, rng, seed):
    """Ilyas et al. 2022: fit an l1-regularised linear model of the reward on the
    inclusion indicators of random sets of the deployment size."""
    from sklearn.linear_model import LassoCV
    nA = act.size
    X = np.zeros((n_eval, nA), np.float32)
    y = np.zeros(n_eval)
    for t in range(n_eval):
        s = rng.choice(nA, size=B, replace=False)
        X[t, s] = 1.0
        y[t] = rs.mrr(act[s])
    # sklearn >= 1.9 takes the NUMBER of alphas as `alphas` (n_alphas was removed)
    m = LassoCV(cv=5, alphas=40, random_state=seed, max_iter=5000).fit(X, y)
    cv_mse = float(m.mse_path_.mean(1).min())
    return m.coef_, {"n_eval": n_eval, "alpha": float(m.alpha_), "n_nonzero": int((m.coef_ != 0).sum()),
                     "cv_r2": 1.0 - cv_mse / float(y.var())}


def facility_sig(Ypk, dep):
    """Greedy facility location on signature cosine (negatives clipped: a dissimilar
    exemplar represents nothing), targets act + aux, candidates act."""
    A, T = dep["act"], np.concatenate([dep["act"], dep["aux"]])
    U = RH.unit(Ypk[T].astype(np.float64))
    M = np.maximum(U @ U[:A.size].T, 0.0)          # T x A; T starts with A
    M[np.arange(A.size), np.arange(A.size)] = 0.0  # a gene is not its own exemplar
    s, tm = np.zeros(T.size), np.ones(T.size)
    avail = np.ones(A.size, bool)
    chosen = []
    for _ in range(B):
        av = np.flatnonzero(avail)
        k = int(av[np.argmax((tm[:, None] * np.maximum(M[:, av] - s[:, None], 0)).sum(0))])
        avail[k] = False
        tm[k] = 0.0
        s = np.maximum(s, M[:, k])
        chosen.append(int(A[k]))
    return chosen


def coreset_sig(Ypk, dep):
    """k-centre greedy on the true signatures, bmdal's convention for an empty labelled
    set: start from the largest norm, then always the farthest point."""
    A = dep["act"]
    X = Ypk[A].astype(np.float64)
    sq = (X ** 2).sum(1)
    md = np.full(A.size, np.inf)
    sel = [int(np.argmax(sq))]
    for _ in range(B - 1):
        md = np.minimum(md, sq + sq[sel[-1]] - 2 * X @ X[sel[-1]])
        s = md.copy()
        s[sel] = -np.inf
        sel.append(int(np.argmax(s)))
    return [int(A[i]) for i in sel]


def run_peek(W, dep, method, seed):
    rng = np.random.default_rng(20_000 + seed)
    act = dep["act"]
    Ypk = W.Y.copy()
    Ypk[dep["rep"]] = np.nan                     # report genes are never visible
    if method == "rep_facility_sig":
        return facility_sig(Ypk, dep), {}
    if method == "rep_coreset_sig":
        return coreset_sig(Ypk, dep), {}
    rs = RA.RewardSet(W, dep["aux"])             # the reward greedy and REINFORCE optimise
    t0 = time.time()
    if method == "dv_shapley":
        v, d = tmc_shapley(rs, act, N_EVAL, rng)
    elif method == "dv_banzhaf":
        v, d = banzhaf_msr(rs, act, N_EVAL, rng)
    elif method == "dv_datamodels":
        v, d = datamodels(rs, act, N_EVAL, rng, seed)
    else:
        raise ValueError(method)
    ch = top_b(v, act)
    d.update(own_reward=rs.mrr(ch), valuation_secs=round(time.time() - t0, 1))
    return ch, d


# ======================================================================== grid
def grid():
    ds = RA.env_list("PJ_RLA_DATASETS", RA.DATASETS)
    sp = RA.env_list("PJ_RLA_SPLITS", (4, 104, 204), int)
    me = RA.env_list("PJ_REL_METHODS", METHODS)
    se = RA.env_list("PJ_REL_SEEDS", (0, 1, 2), int)
    return [(d, s, m, z) for d in ds for s in sp for m in me for z in se]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prep", "run", "grid"))
    ap.add_argument("--zip", default="")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--code-dir", default="")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    a = ap.parse_args()
    if a.mode == "grid":
        g = grid()
        print(json.dumps({"n": len(g), "grid": g}))
        return 0
    if a.mode == "prep":
        return prep(a)

    g = grid()
    ds, sp, me, seed = g[a.task_index]
    if me not in METHODS:
        print(f"GATE NOT MET: unknown method {me}")
        return 0
    regime, fam, det, init_random, cite = METHODS[me]
    if det and seed != RA.env_list("PJ_REL_SEEDS", (0, 1, 2), int)[0]:
        log(f"{me} is deterministic; the first seed covers it, nothing to do for seed {seed}")
        return 0
    od = os.path.join(a.out_dir, "rl_acquire", OUT_SUB, "runs")
    os.makedirs(od, exist_ok=True)
    tag = f"{me}__{ds}__s{sp}__seed{seed}"
    out = os.path.join(od, tag + ".json")
    if os.path.exists(out) and not os.environ.get("PJ_REL_FORCE"):
        log(f"already done, skipping (PJ_REL_FORCE=1 to redo): {out}")
        return 0
    check = json.load(open(os.path.join(a.cache_dir, "cache_check.json")))
    info = check["datasets"].get(ds)
    if not info or not info.get("ok"):
        print(f"GATE NOT MET: cache for {ds} missing or did not reproduce GAPS's random median: {info}")
        return 0
    priors = None
    if fam == "iterpert" or me in ("fl_iterpert_priors", "kc_iterpert_priors", "kc_go"):
        if me in ("iterpert_go", "kc_go"):
            priors = "go"
        else:
            pc = os.path.join(a.out_dir, "rl_acquire", "related", "priors_check.json")
            pinfo = (json.load(open(pc)) if os.path.exists(pc) else {}).get("datasets", {}).get(ds)
            if not pinfo or not pinfo.get("ok") or not os.path.exists(pinfo.get("path", "")):
                # a failed gate, not a crash: say why and exit 0 (CLAUDE.md). `ok` needs
                # every prior IterPert's config names, so it never runs on a partial set.
                print(f"GATE NOT MET: IterPert priors for {ds} missing or incomplete ({pc}: {pinfo}); "
                      f"see jobs/44_related_prep.sbatch")
                return 0
            priors = pinfo["path"]
    W = RA.World(info["path"])
    dep = RA.split_of(W.K, sp)
    log(f"related {tag}: {regime}/{fam} ({cite}) act={dep['act'].size} aux={dep['aux'].size} "
        f"rep={dep['rep'].size} n_eval={N_EVAL if fam == 'valuation' else '-'}")
    base = {"dataset": ds, "split": sp, "seed": seed, "budget": B, "K_report": int(dep["rep"].size),
            "method": me, "regime": regime, "family": fam, "cite": cite, "random_init_batch": init_random,
            "smoke": SMOKE}
    t0 = time.time()
    try:
        if regime == "blind":
            api = iterpert_api(a.code_dir, a.threads) if fam in ("bmdal", "iterpert") else None
            P = None
            if priors == "go":
                P = [W.JJ]
            elif priors is not None:
                z = np.load(priors, allow_pickle=False)
                if [str(c) for c in z["cands"]] != [str(c) for c in W.cands]:
                    raise RuntimeError("prior kernels are not aligned to this cache's candidates")
                P = list(z["kernels"])
                base["priors"] = [str(x) for x in z["names"]]
            if me == "fl_iterpert_priors":
                chosen, diag = facility_prior(P, dep), {}
            elif me in ("kc_iterpert_priors", "kc_go"):
                chosen, diag = kcenter_prior(P, dep), {}
            else:
                chosen, diag = run_blind(W, dep, me, seed, P, api)
        else:
            chosen, diag = run_peek(W, dep, me, seed)
        r = RA.Reporter(W, dep["rep"], dep["act"])(chosen)
        res = dict(base, chosen=[int(c) for c in chosen], diag=diag, **{f"rep_{k}": v for k, v in r.items()})
    except Exception as e:
        traceback.print_exc()
        res = dict(base, error=f"{type(e).__name__}: {e}")
    res["secs"] = round(time.time() - t0, 1)
    if "error" in res:
        # never write the result file on a crash, or the resume check skips it forever
        with open(os.path.join(od, tag + ".error.json"), "w") as f:
            json.dump([res], f)
        return 1
    log(f"RESULT {me:16s} rep mrr={res['rep_mrr']:.4f} top1={res['rep_top1']:.4f} ({res['secs']}s)")
    with open(out + ".tmp", "w") as f:
        json.dump([res], f, default=float)
    os.replace(out + ".tmp", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
