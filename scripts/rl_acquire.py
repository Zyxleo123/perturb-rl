#!/usr/bin/env python
"""RL for the gene-acquisition gap (GAPS.md, "Is the unseen-gene gap RL-shaped?").

The measured failure. Choose 100 of 437 genes to measure; reward = retrieval MRR
of a GO-kNN predictor on held-out genes. Greedy on a reward computed over a few
hundred genes chases that reward's noise: it reaches ~1.2x random on disjoint
report genes, while the same greedy selecting on the report genes themselves (a
cheat) shows ~1.65-2.1x exists. Three RL approaches, each a different hypothesis
about why greedy falls short, all evaluated on the identical deployment split,
predictor and report genes as GAPS part 4:

  pg_tab        REINFORCE with a leave-one-out baseline on a Plackett-Luce policy
                holding one logit per candidate gene. Same information and same
                objective as greedy; only the optimiser changes.
                pg_tab_boot resamples half the reward genes every episode.
                Hypothesis tested: "it is the search".
  ppo_peek /    PPO with group-relative advantages (GRPO-style: G rollouts share one
  ppo_blind     sub-problem, so how lucky that sub-problem's reward genes were cancels
                out) on a permutation-equivariant policy that scores genes by
                FEATURES. Trained across thousands of randomly re-split sub-problems,
                it learns a rule, not a list, so it cannot memorise one reward set.
                Hypothesis tested: "it is generalisation".
  critic        value-based: a learned critic of the one-step acquisition gain
                (fitted Q, gamma = 0, i.e. a contextual-bandit value) trained on
                held-out reward across re-split sub-problems. Used greedily on its
                own (critic_only), and as a control variate that shrinks greedy's
                noisy empirical gain toward it (critic_shrink).
                Hypothesis tested: "it is the reward estimate".

Added 2026-11-17, after the first round left the blind regime open (PPO there only
matched a static GO facility-location heuristic). Three suspects, one variant each
and the combination, all deployed on the identical splits (see BLIND_VARIANTS):
  ppo_blind_match       deployment-scale training sub-problems (437 candidates,
                        budget 100) instead of proportionally shrunk ones (~190/43).
  ppo_blind_xline       trained on the OTHER cell line's whole atlas: deployment
                        scale, and nothing of the deployment line enters training or
                        model selection.
  ppo_blind_rich        richer prior view of a gene than GO Jaccard (IDF-weighted
                        term cosine + a term co-occurrence embedding). The PREDICTOR
                        is unchanged, so every MRR stays comparable.
  ppo_blind_xline_rich  both.

Information regimes, i.e. what a policy may look at:
  peek   every non-report signature, including a candidate's before it is chosen.
         GAPS part 4's greedy used exactly this (it scores a candidate by inserting
         its real signature), so it is the fair comparison for greedy.
  blind  a candidate's signature is revealed only once measured, in 4 batched
         rounds: genuine sequential experimental design. Its training sub-problems
         are drawn from the already-measured atlas (aux) only, never from the
         candidates, so the policy has never seen a single candidate signature.

Splits per dataset (split seed 4 = GAPS part 4's split, number for number):
  act  437 genes a policy chooses 100 of
  aux  remaining non-report genes (part 4's "rew_pool"): the visible atlas
  rep  report genes -- seen ONLY by the explicitly cheating bound and by evaluation.
Every learned method trains on sub-problems re-split three ways (candidates /
visible / hidden-reward), so it is always rewarded on genes its features could not
see -- the deployment condition. Checkpoints are selected on validation
sub-problems, never on rep. Rep is logged along the learning curve for diagnosis
only.

Final numbers go through rl_headroom's own go_knn_predict + score (class Reporter);
the fast Gram-matrix scorer used inside training is checked against it every run.

Usage:
  python -I rl_acquire.py baselines --cache-dir D --out-dir D --task-index I
  python -I rl_acquire.py train     --cache-dir D --out-dir D --task-index I
  python -I rl_acquire.py grid      (prints the task grid and its size)
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

KNN = 10                 # rl_headroom.go_knn_predict's k
R_ROUNDS = 4             # GAPS part 4: budget spent in 4 rounds
B_DEPLOY = 100           # GAPS part 4: budget 100
N_RAND = 100             # random baseline is a distribution, never one draw
N_GREEDY_SEEDS = 8       # ... and so is a greedy run (see run_baselines)
N_GREEDY_ALL_SEEDS = 3
DATASETS = ("replogle_k562_essential", "replogle_rpe1_essential")
METHODS = ("pg_tab", "pg_tab_boot", "ppo_peek", "ppo_blind", "critic",
           "ppo_blind_match", "ppo_blind_rich", "ppo_blind_xline", "ppo_blind_xline_rich")
OTHER = {"replogle_k562_essential": "replogle_rpe1_essential",
         "replogle_rpe1_essential": "replogle_k562_essential"}

# The blind regime was the open problem: PPO there only matched static GO facility
# location. GAPS.md named three suspects -- sub-problem scale, where the policy is
# trained, and a GO-only view of a gene. One variant each, plus the combination.
#   pool  "aux"   the visible atlas of THIS line (the honest default; pool-limited)
#         "all"   act + aux of this line: big enough to match deployment scale, at
#                 the cost that candidate signatures enter the training REWARD (never
#                 the features). Isolates scale with no cross-line shift.
#         "other" the other cell line's entire atlas: deployment scale AND nothing of
#                 this line is touched by training or model selection.
BLIND_VARIANTS = {
    "ppo_blind": {"pool": "aux", "match": False, "rich": False},
    "ppo_blind_match": {"pool": "all", "match": True, "rich": False},
    "ppo_blind_rich": {"pool": "aux", "match": False, "rich": True},
    "ppo_blind_xline": {"pool": "other", "match": True, "rich": False},
    "ppo_blind_xline_rich": {"pool": "other", "match": True, "rich": True},
}


def env_list(name, default, typ=str):
    v = os.environ.get(name, "").split()
    return [typ(x) for x in v] if v else list(default)


def env_num(name, default, typ=int):
    v = os.environ.get(name, "")
    return typ(v) if v else default


SMOKE = bool(os.environ.get("PJ_RLA_SMOKE"))
CFG = {
    "ppo_iters": env_num("PJ_RLA_PPO_ITERS", 3 if SMOKE else 120),
    "ppo_univ": env_num("PJ_RLA_PPO_UNIV", 2 if SMOKE else 8),
    "ppo_group": env_num("PJ_RLA_PPO_GROUP", 3 if SMOKE else 8),
    "ppo_epochs": env_num("PJ_RLA_PPO_EPOCHS", 4),
    "ppo_lr": env_num("PJ_RLA_PPO_LR", 1e-3, float),
    "ppo_clip": 0.2,
    "ppo_ent": env_num("PJ_RLA_PPO_ENT", 3e-3, float),
    "ppo_hidden": 32,
    "t_max": 400,            # proxy targets sub-sampled per TRAINING episode (speed)
    "n_val": env_num("PJ_RLA_NVAL", 3 if SMOKE else 16),
    "val_every": env_num("PJ_RLA_VAL_EVERY", 1 if SMOKE else 10),
    "pg_iters": env_num("PJ_RLA_PG_ITERS", 5 if SMOKE else 600),
    "pg_samples": 32,
    "pg_lr": 0.05,
    "critic_univ": env_num("PJ_RLA_CRITIC_UNIV", 4 if SMOKE else 300),
    "critic_states": 6,
    "critic_cands": 48,
    "critic_epochs": env_num("PJ_RLA_CRITIC_EPOCHS", 5 if SMOKE else 300),
}


def log(*a):
    RH.log(*a)


# ======================================================================= world
class World:
    """One dataset plus the Gram matrices that make one reward evaluation a small
    (queries x B x |R|) product instead of a fresh kNN + cosine over ~4,500 genes."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.name = str(z["name"])
        self.Y = z["Y"].astype(np.float32)          # K x G, masked, all cells
        self.Q = z["Q"].astype(np.float32)          # nq x G, held-out-cell queries
        self.owner = z["owner"].astype(np.int64)
        self.F = z["F"].astype(np.float32)          # K x terms, binary GO
        self.cands = z["cands"]
        self.K = self.Y.shape[0]
        self.JJ = RH.jaccard(self.F, self.F)        # K x K, exact (0/1 sums)
        Yd = self.Y.astype(np.float64)
        self.YY = Yd @ Yd.T
        nrm = np.sqrt(np.maximum(np.diag(self.YY), 1e-24))
        self.CC = (self.YY / nrm[:, None] / nrm[None, :]).astype(np.float32)
        self.lognorm = np.log(nrm).astype(np.float32)
        self.ynorm = nrm
        self.QY = (RH.unit(self.Q) @ self.Y.T).astype(np.float32)   # nq x K
        self.deg = np.log1p(self.F.sum(1)).astype(np.float32)
        order = np.argsort(self.owner, kind="stable")
        b = np.searchsorted(self.owner[order], np.arange(self.K + 1))
        self.q_of = [order[b[i]:b[i + 1]] for i in range(self.K)]
        self._rich = None

    def rich(self):
        """Two prior-only similarity views richer than GO Jaccard, built once.

        Jaccard weights every term equally and sees only DIRECT overlap, so two
        genes annotated to different terms of the same process are as far apart as
        two unrelated genes. These two views fix one limitation each:
          idf   cosine on IDF-weighted term vectors -- a rare term counts for more
                than a term half the genome carries.
          emb   cosine in the top-32 SVD subspace of that weighted matrix -- term
                CO-OCCURRENCE, so related-but-non-overlapping annotations are near.
        Both are functions of F alone, so they are legal in the blind regime: they
        say nothing about any candidate's signature. The PREDICTOR still uses plain
        Jaccard kNN, exactly as every number in GAPS.md does; only the policy's view
        of the candidates is richer. That is the point -- a policy that can see
        structure the predictor cannot is what lets it pick genes the predictor will
        be able to use."""
        if self._rich is None:
            F = self.F
            idf = np.log(self.K / np.maximum(F.sum(0), 1.0)).astype(np.float32)
            Fw = F * idf
            U = RH.unit(Fw)
            Sidf = (U @ U.T).astype(np.float32)
            Fc = (Fw - Fw.mean(0)).astype(np.float64)
            r = max(1, min(32, min(Fc.shape) - 1))
            Us, sv, _ = np.linalg.svd(Fc, full_matrices=False)
            E = RH.unit((Us[:, :r] * sv[:r]).astype(np.float32))
            self._rich = {"idf": Sidf, "emb": (E @ E.T).astype(np.float32),
                          "spec": ((F @ idf) / np.maximum(F.sum(1), 1.0)).astype(np.float32)}
            log(f"  [{self.name}] rich views: idf + emb(r={r}) over {F.shape[1]} GO terms")
        return self._rich


def split_of(K, seed):
    """GAPS part 4's split (rl_headroom.reward_noise_headroom); seed 4 is GAPS's."""
    perm = np.random.default_rng(seed).permutation(K)
    n_rep = max(60, K // 6)
    n_act = min((K - n_rep) // 2, 437)
    return {"rep": np.sort(perm[:n_rep]), "act": np.sort(perm[n_rep:n_rep + n_act]),
            "aux": np.sort(perm[n_rep + n_act:])}


class RewardSet:
    """Retrieval of the GO-kNN predictor built from `chosen`, scored on genes R.

    Algebraically identical to rl_headroom.score(rl_headroom.go_knn_predict(...)):
    the same argpartition picks the same neighbours, and cos(q, sum_c w_c Y_c) is
    assembled from the precomputed unit(Q) Y^T and Y Y^T. Same tie-aware metric."""

    def __init__(self, W, R):
        self.W = W
        self.R = np.asarray(R, np.int64)
        self.n = self.R.size
        qi = np.concatenate([W.q_of[g] for g in self.R])
        pos = np.full(W.K, -1, np.int64)
        pos[self.R] = np.arange(self.n)
        self.o = pos[W.owner[qi]]
        self.QY = W.QY[qi]
        self.JR = W.JJ[self.R]
        self.cnt = np.bincount(self.o, minlength=self.n)
        self.m = self.cnt > 0
        self.chance_mrr = float(RH.harmonic(self.n)[self.n] / self.n)

    def S(self, chosen):
        c = np.asarray(chosen, np.int64)
        B = c.size
        J = self.JR[:, c]
        kk = min(KNN, B)
        top = np.argpartition(-J, kk - 1, axis=1)[:, :kk]
        w = np.take_along_axis(J, top, 1)
        ws = w.sum(1)
        ok = ws > 0
        Wm = np.zeros((self.n, B), np.float32)
        np.put_along_axis(Wm, top, (w / np.where(ok, ws, 1.0)[:, None]).astype(np.float32), 1)
        Wm[~ok] = 1.0 / B                                   # go_knn_predict's `base`
        num = self.QY[:, c] @ Wm.T
        n2 = np.einsum("ij,ij->i", Wm.astype(np.float64) @ self.W.YY[np.ix_(c, c)], Wm)
        return np.round(num / np.sqrt(np.maximum(n2, 1e-24)).astype(np.float32)[None, :], 6)

    def metrics(self, chosen, full=False):
        if len(chosen) == 0:
            return self.chance_mrr if not full else {"mrr": self.chance_mrr, "top1": 1.0 / self.n}
        S = self.S(chosen)
        t = S[np.arange(S.shape[0]), self.o]
        g = (S > t[:, None]).sum(1)
        e = (S == t[:, None]).sum(1) - 1
        H = RH.harmonic(self.n)
        per = (H[g + e + 1] - H[g]) / (e + 1)
        mrr = float((np.bincount(self.o, weights=per, minlength=self.n)[self.m] / self.cnt[self.m]).mean())
        if not full:
            return mrr
        t1 = np.clip((1 - g) / (e + 1), 0, 1)
        top1 = float((np.bincount(self.o, weights=t1, minlength=self.n)[self.m] / self.cnt[self.m]).mean())
        return {"mrr": mrr, "top1": top1}

    def mrr(self, chosen):
        return self.metrics(chosen)


class Reporter:
    """Final numbers through rl_headroom's own functions, so a reported number is
    the same measurement GAPS.md reports, not this file's fast re-implementation."""

    def __init__(self, W, rep, act):
        m = np.isin(W.owner, rep)
        remap = {o: i for i, o in enumerate(rep)}
        self.Q, self.o = W.Q[m], np.array([remap[o] for o in W.owner[m]])
        self.J = RH.jaccard(W.F[rep], W.F[act])
        self.Yact = W.Y[act]
        self.pos = {int(g): i for i, g in enumerate(act)}

    def __call__(self, chosen):
        c = np.array([self.pos[int(g)] for g in chosen])
        P = RH.go_knn_predict(self.J[:, c], self.Yact[c], self.Yact[c].mean(0))
        r = RH.score(P, self.Q, self.o)
        return {k: r[k] for k in ("mrr", "top1", "top5", "top10", "median_rank")}


def round_sizes(B, R=R_ROUNDS):
    return [B // R + (1 if r < B % R else 0) for r in range(R)]


# ==================================================================== features
GO_FEATS = ["deg", "centrality", "fl_gain", "knn_entry", "uncovered", "go_redund"]
PEEK_FEATS = ["lognorm", "coherence", "typicality", "sig_redund"]
NB_FEATS = ["nb_lognorm", "nb_agree", "nb_mass"]
BLIND_FEATS = ["pred_redund"]
# prior-only, blind-legal: the same coverage/redundancy questions asked under the two
# richer similarity views of World.rich(), plus how specific the gene's annotation is
RICH_FEATS = ["idf_fl_gain", "idf_redund", "emb_fl_gain", "emb_redund", "spec"]
CTX_FEATS = ["spent", "frac_uncov", "mean_cov", "round"]


def feature_names(regime, rich=False):
    body = GO_FEATS + (PEEK_FEATS + NB_FEATS if regime == "peek" else NB_FEATS + BLIND_FEATS)
    return body + (RICH_FEATS if rich else []) + CTX_FEATS


class FeatureState:
    """Per-candidate features of one acquisition episode, updated after every pick.

    GO-only features use prior information only; they are legal in both regimes.
      fl_gain     facility-location gain: mean over proxy targets t of
                  max(0, J(t,i) - max_{c chosen} J(t,c))
      knn_entry   fraction of targets for which i would enter their 10 GO neighbours
      uncovered   fraction of targets with no GO overlap to the chosen set that i covers
      go_redund   max GO Jaccard of i to the chosen set
    Proxy targets are the candidates plus the visible atlas, never the reward genes,
    mirroring deployment where they are act + aux and never rep.

    peek adds the candidate's own signature: its norm, `coherence` (do its GO
    neighbours among visible genes have its signature?), typicality and signature
    redundancy with the chosen set. blind never touches a candidate's signature: its
    neighbourhood features read only the visible atlas and genes already measured,
    refreshed at the start of each round (a batch of measurements comes back)."""

    def __init__(self, W, act, vis, regime, B, rng=None, t_max=None, rich=False):
        self.W, self.A, self.vis, self.regime, self.B = W, np.asarray(act), np.asarray(vis), regime, B
        self.rich = rich
        T = np.concatenate([self.A, self.vis])
        if t_max is not None and T.size > t_max:
            T = rng.choice(T, size=t_max, replace=False)
        self.T = T
        JTA = W.JJ[np.ix_(T, self.A)].copy()
        JTA[T[:, None] == self.A[None, :]] = 0.0            # a gene is not its own neighbour
        self.JTA = JTA
        self.trow = {int(g): i for i, g in enumerate(T)}
        self.tm = np.ones(T.size, np.float32)
        self.s = np.zeros(T.size, np.float32)
        self.topk = np.zeros((T.size, KNN), np.float32)     # ascending: [:, 0] = k-th best
        nA = self.A.size
        self.avail = np.ones(nA, bool)
        self.go_red = np.zeros(nA, np.float32)
        self.chosen = []
        self.round = 0
        self.static = {"deg": W.deg[self.A], "centrality": JTA.mean(0)}
        if rich:
            R = W.rich()
            self._R = R
            self.rich_TA, self.rich_s, self.rich_red = {}, {}, {}
            for v in ("idf", "emb"):
                M = R[v][np.ix_(T, self.A)].copy()
                M[T[:, None] == self.A[None, :]] = 0.0      # a gene is not its own neighbour
                self.rich_TA[v] = M
                self.rich_s[v] = np.zeros(T.size, np.float32)
                self.rich_red[v] = np.zeros(nA, np.float32)
            self.static["spec"] = R["spec"][self.A]
        if regime == "peek":
            V = np.concatenate([self.A, self.vis])
            self.static["lognorm"] = W.lognorm[self.A]
            JAV = W.JJ[np.ix_(self.A, V)].copy()
            JAV[self.A[:, None] == V[None, :]] = 0.0
            top = np.argpartition(-JAV, KNN - 1, axis=1)[:, :KNN]
            w = np.take_along_axis(JAV, top, 1)
            cs = W.CC[self.A[:, None], V[top]]
            self.static["coherence"] = ((w * cs).sum(1) / np.maximum(w.sum(1), 1e-9)).astype(np.float32)
            self.static["typicality"] = (RH.unit(W.Y[self.A]) @ RH.unit(W.Y[V].mean(0))[0]).astype(np.float32)
            self.sig_red = np.zeros(nA, np.float32)
            self._neighbourhood(V)

    def _neighbourhood(self, M):
        """Features of each candidate's GO neighbourhood among signature-visible genes M:
        how strong and how self-consistent the signature a kNN would predict for it is."""
        W = self.W
        M = np.asarray(M)
        JAM = W.JJ[np.ix_(self.A, M)].copy()
        JAM[self.A[:, None] == M[None, :]] = 0.0
        kk = min(KNN, M.size)
        top = np.argpartition(-JAM, kk - 1, axis=1)[:, :kk]
        w = np.take_along_axis(JAM, top, 1)
        ws = w.sum(1)
        wn = w / np.maximum(ws, 1e-9)[:, None]
        nb = M[top]
        n2 = np.einsum("ik,ikl,il->i", wn, W.YY[nb[:, :, None], nb[:, None, :]], wn)
        ag = np.einsum("ik,ikl,il->i", wn, W.CC[nb[:, :, None], nb[:, None, :]].astype(np.float64), wn)
        none = ws <= 0
        if none.any():                       # no GO overlap: the predictor falls back to the mean
            n2[none] = W.YY[np.ix_(M, M)].mean()
            ag[none] = 0.0
        pn = np.sqrt(np.maximum(n2, 1e-24))
        self.nb = {"nb_lognorm": np.log(pn).astype(np.float32),
                   "nb_agree": np.sqrt(np.maximum(ag, 0)).astype(np.float32),
                   "nb_mass": ws.astype(np.float32)}
        if self.regime == "blind":
            Cm = np.asarray(self.chosen, np.int64)
            if Cm.size:
                dots = np.einsum("ik,ikc->ic", wn, W.YY[nb[:, :, None], Cm[None, None, :]])
                cos = dots / pn[:, None] / W.ynorm[Cm][None, :]
                cos[none] = 0.0
                self.nb["pred_redund"] = cos.max(1).astype(np.float32)
            else:
                self.nb["pred_redund"] = np.zeros(self.A.size, np.float32)

    def begin_round(self, rd):
        self.round = rd
        if self.regime == "blind":   # everything chosen so far has now been measured
            self._neighbourhood(np.concatenate([self.vis, np.asarray(self.chosen, np.int64)]))

    def raw_gain(self):
        av = np.flatnonzero(self.avail)
        G = self.JTA[:, av]
        return (self.tm[:, None] * np.maximum(G - self.s[:, None], 0)).sum(0), av

    def features(self):
        av = np.flatnonzero(self.avail)
        G = self.JTA[:, av]
        tm = self.tm
        den = max(float(tm.sum()), 1.0)
        unc = tm * (self.s <= 0)
        cols = [self.static["deg"][av], self.static["centrality"][av],
                (tm[:, None] * np.maximum(G - self.s[:, None], 0)).sum(0) / den,
                (tm[:, None] * (G > self.topk[:, :1])).sum(0) / den,
                (unc[:, None] * (G > 0)).sum(0) / den,
                self.go_red[av]]
        if self.regime == "peek":
            cols += [self.static["lognorm"][av], self.static["coherence"][av],
                     self.static["typicality"][av], self.sig_red[av]]
            cols += [self.nb[k][av] for k in NB_FEATS]
        else:
            cols += [self.nb[k][av] for k in NB_FEATS + BLIND_FEATS]
        if self.rich:
            for v in ("idf", "emb"):
                Gv = self.rich_TA[v][:, av]
                cols += [(tm[:, None] * np.maximum(Gv - self.rich_s[v][:, None], 0)).sum(0) / den,
                         self.rich_red[v][av]]
            cols.append(self.static["spec"][av])
        X = np.stack(cols, 1).astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        ctx = np.array([len(self.chosen) / self.B, unc.sum() / den, (tm * self.s).sum() / den,
                        self.round / R_ROUNDS], np.float32)
        return np.concatenate([X, np.broadcast_to(ctx, (av.size, ctx.size))], 1), av

    def add(self, a):
        """a = position in self.A."""
        g = int(self.A[a])
        self.avail[a] = False
        self.chosen.append(g)
        r = self.trow.get(g)
        if r is not None:
            self.tm[r] = 0.0
        col = self.JTA[:, a]
        self.s = np.maximum(self.s, col)
        tk = np.concatenate([self.topk, col[:, None]], 1)
        tk.sort(1)
        self.topk = tk[:, 1:]
        self.go_red = np.maximum(self.go_red, self.W.JJ[self.A, g])
        if self.rich:
            for v in ("idf", "emb"):
                self.rich_s[v] = np.maximum(self.rich_s[v], self.rich_TA[v][:, a])
                self.rich_red[v] = np.maximum(self.rich_red[v], self._R[v][self.A, g])
        if self.regime == "peek":
            self.sig_red = np.maximum(self.sig_red, self.W.CC[self.A, g])


# =================================================================== episodes
def make_sampler(dep, regime, pool=None, na_target=None):
    """Training sub-problems in the deployment's proportions (act : vis : hidden
    reward), each fraction jittered +/-15%, budget scaled with the candidate count.
    peek draws from act+aux (the same signatures greedy may read); blind from aux.

    `na_target` turns on SIZE MATCHING, the lever GAPS.md names for the blind regime.
    With it None (the default, and what every result before 2026-11-17 used), a
    sub-problem is a scaled-down copy of deployment: its candidate count is a
    fraction of the pool, so on K562 the blind policy trains on ~190 candidates with
    budget ~43 and is then deployed on 437 with budget 100. The ratio matches; the
    absolute scale does not, and k in the kNN is fixed at 10, so the redundancy a
    gene has to avoid is a different quantity at the two scales. With na_target set,
    the candidate slate and the budget are deployment's own and the remaining genes
    keep deployment's reward : visible ratio -- which only fits if the pool is big
    enough, hence `pool` (the other cell line's whole atlas is the clean way to get
    one; see BLIND_VARIANTS)."""
    n = np.array([dep["act"].size, dep["aux"].size, dep["rep"].size], float)
    if pool is None:
        pool = np.concatenate([dep["act"], dep["aux"]]) if regime == "peek" else dep["aux"].copy()
    pool = np.asarray(pool)

    def sample(rng):
        p = rng.permutation(pool)
        fr = rng.uniform(0.85, 1.15, 3) * n
        fr = fr / fr.sum()
        if na_target is None:
            na, nr = int(round(fr[0] * p.size)), int(round(fr[2] * p.size))
        else:
            # never take so many candidates that nothing is left to be rewarded on
            na = int(min(na_target, 0.8 * p.size))
            rest = p.size - na
            nr = int(np.clip(round(rest * n[2] / (n[1] + n[2])), 8, max(8, rest - 8)))
        B = max(8, int(round(B_DEPLOY * na / n[0])))
        return {"act": np.sort(p[:na]), "rew": np.sort(p[na:na + nr]),
                "vis": np.sort(p[na + nr:]), "B": B}
    return sample


def deploy_episode(dep):
    return {"act": dep["act"], "vis": dep["aux"], "rew": None, "B": B_DEPLOY}


def rollout(W, ep, regime, logit_fn, rng, mode="sample", rew=None, record=False, t_max=None,
            rich=False):
    fs = FeatureState(W, ep["act"], ep["vis"], regime, ep["B"], rng, t_max, rich)
    mr, rec = [], []
    for rd, b in enumerate(round_sizes(ep["B"])):
        fs.begin_round(rd)
        for _ in range(b):
            X, av = fs.features()
            z = logit_fn(X)
            k = int(np.argmax(z + rng.gumbel(size=z.size))) if mode == "sample" else int(np.argmax(z))
            if record:
                rec.append((X, k, rd))
            fs.add(av[k])
        if rew is not None:
            mr.append(rew.mrr(fs.chosen))
    return fs.chosen, mr, rec


# ================================================================== baselines
def greedy(W, rewset, act, B, srng, n_cand=None):
    """rl_headroom.reward_noise_headroom.run_greedy line for line, on the fast scorer.
    n_cand=None scores every available candidate instead of a random 120."""
    nA = act.size
    ev = lambda loc: rewset.mrr(act[np.asarray(loc, np.int64)])  # noqa: E731
    chosen = []
    for rd in range(R_ROUNDS):
        b = B // R_ROUNDS + (1 if rd < B % R_ROUNDS else 0)
        avail = np.setdiff1d(np.arange(nA), chosen)
        pick = []
        for _ in range(b):
            rest = np.setdiff1d(avail, pick)
            cand = rest if n_cand is None else srng.choice(rest, size=min(n_cand, avail.size - len(pick)),
                                                           replace=False)
            base = chosen + list(pick)
            sc = [ev(base + [int(c)]) if base else
                  ev([int(c)] + list(srng.choice(avail, size=2, replace=False))) for c in cand]
            pick.append(int(cand[int(np.argmax(sc))]))
        chosen += pick
    return [int(act[i]) for i in chosen]


def go_coverage(W, act, B):
    """Greedy max-coverage of GO terms (GAPS part 2/3's `coverage`)."""
    Fp = W.F[act] > 0
    covered = np.zeros(Fp.shape[1], bool)
    chosen = []
    for _ in range(B):
        gain = (Fp & ~covered).sum(1).astype(float)
        gain[chosen] = -1
        i = int(np.argmax(gain))
        chosen.append(i)
        covered |= Fp[i]
    return [int(act[i]) for i in chosen]


def go_facility(W, dep, B):
    """Facility location on GO Jaccard over targets act+aux: the obvious static,
    prior-only heuristic for 'make every gene have a close measured neighbour'.
    Blind-legal. It is also the GO part of the RL policies' features, so an RL
    policy must beat it to have learned anything beyond it."""
    fs = FeatureState(W, dep["act"], dep["aux"], "blind", B)
    for _ in range(B):
        g, av = fs.raw_gain()
        fs.add(av[int(np.argmax(g))])
    return fs.chosen


def facility(W, dep, B, view=None):
    """Greedy facility location on a prior similarity matrix: choose genes so that
    every gene (candidates + visible atlas) has a close CHOSEN neighbour. Static,
    reads no signature, so it is legal in the blind regime.

    `view=None` is GO Jaccard -- the floor GAPS.md already reports, and this path is
    checked against go_facility()'s own result. The point of the other two views is
    fairness: a policy handed the IDF and embedding views as features (RICH_FEATS)
    must be compared against the best STATIC use of those same views, not only
    against the Jaccard one, or 'richer priors help' would just mean 'the floor was
    computed on a weaker similarity'."""
    A = dep["act"]
    T = np.concatenate([dep["act"], dep["aux"]])
    S = W.JJ if view is None else W.rich()[view]
    M = S[np.ix_(T, A)].copy()
    M[T[:, None] == A[None, :]] = 0.0           # a gene is not its own neighbour
    s = np.zeros(T.size, np.float32)
    tm = np.ones(T.size, np.float32)
    avail = np.ones(A.size, bool)
    chosen = []
    for _ in range(B):
        av = np.flatnonzero(avail)
        g = (tm[:, None] * np.maximum(M[:, av] - s[:, None], 0)).sum(0)
        k = int(av[int(np.argmax(g))])
        avail[k] = False
        chosen.append(int(A[k]))
        r = np.flatnonzero(T == A[k])
        tm[r] = 0.0
        s = np.maximum(s, M[:, k])
    return chosen


def random_dist(rep_fn, act, B, seed=9100, n=N_RAND):
    srng = np.random.default_rng(seed)
    v = [rep_fn(act[srng.choice(act.size, size=B, replace=False)]) for _ in range(n)]
    m = np.array([x["mrr"] for x in v])
    t = np.array([x["top1"] for x in v])
    return {"rep_mrr": float(np.median(m)), "rep_mrr_p05": float(np.percentile(m, 5)),
            "rep_mrr_p95": float(np.percentile(m, 95)), "rep_mrr_max": float(m.max()),
            "rep_mrr_mean": float(m.mean()), "rep_mrr_sd": float(m.std()),
            "rep_top1": float(np.median(t)), "n_draws": n}


def scorer_check(W, dep, rep_fn, n=30, seed=31):
    """Fast scorer vs rl_headroom's own path on random subsets. Must agree."""
    rs = RewardSet(W, dep["rep"])
    rng = np.random.default_rng(seed)
    d_m, d_t = [], []
    for i in range(n):
        B = (10, 25, 100)[i % 3]
        ch = dep["act"][rng.choice(dep["act"].size, size=B, replace=False)]
        a, b = rs.metrics(ch, full=True), rep_fn(ch)
        d_m.append(abs(a["mrr"] - b["mrr"]))
        d_t.append(abs(a["top1"] - b["top1"]))
    return {"max_abs_diff_mrr": float(max(d_m)), "max_abs_diff_top1": float(max(d_t)), "n": n}


def run_baselines(W, split_seed, out):
    dep = split_of(W.K, split_seed)
    rep_fn = Reporter(W, dep["rep"], dep["act"])
    base = {"dataset": W.name, "split": split_seed, "budget": B_DEPLOY, "act": int(dep["act"].size),
            "aux": int(dep["aux"].size), "K_report": int(dep["rep"].size)}
    rows = []

    def put(strategy, chosen=None, extra=None, t0=None):
        r = dict(base, strategy=strategy)
        if chosen is not None:
            r.update({f"rep_{k}": v for k, v in rep_fn(chosen).items()})
            r["chosen"] = [int(x) for x in chosen]
        r.update(extra or {})
        if t0 is not None:
            r["secs"] = round(time.time() - t0, 1)
        rows.append(r)
        with open(out, "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        log(f"  [{W.name} s{split_seed}] {strategy:18s} rep mrr="
            f"{r.get('rep_mrr', float('nan')):.4f} top1={r.get('rep_top1', float('nan')):.4f}")

    t0 = time.time()
    chk = scorer_check(W, dep, rep_fn)
    put("scorer_check", extra=chk, t0=t0)
    t0 = time.time()
    put("random", extra=random_dist(rep_fn, dep["act"], B_DEPLOY), t0=t0)
    t0 = time.time()
    put("go_coverage", go_coverage(W, dep["act"], B_DEPLOY), t0=t0)
    t0 = time.time()
    ch_fac = go_facility(W, dep, B_DEPLOY)
    put("go_facility", ch_fac, t0=t0)
    # The same static heuristic under the two richer prior views. A policy handed
    # those views as features (RICH_FEATS) has to beat their best STATIC use, not
    # just the Jaccard one, or "richer priors help" would only mean "the floor was
    # measured on a weaker similarity".
    for v in ("idf", "emb"):
        t0 = time.time()
        put(f"go_facility_{v}", facility(W, dep, B_DEPLOY, v), t0=t0)
    put("facility_path_check",
        extra={"jaccard_view_matches_go_facility": facility(W, dep, B_DEPLOY) == ch_fac})
    rs_aux, rs_rep = RewardSet(W, dep["aux"]), RewardSet(W, dep["rep"])
    # A single greedy run is itself ONE DRAW: its 120 sampled candidates, its random
    # first pick, and the kNN's arbitrary tie-breaking among equal GO Jaccards all
    # move its path. (Measured: the identical protocol gave 0.221 under numpy 2.4 and
    # 0.190 under numpy 1.26, whose argpartition breaks ties differently.) So greedy
    # and the cheat are run over several seeds and read as distributions. Seed 0 of
    # each is GAPS part 4's own seed.
    if dep["aux"].size >= 400:     # GAPS part 4's exact protocol: 400 reward genes, 120 candidates
        sub = np.sort(np.random.default_rng(555).choice(dep["aux"], size=400, replace=False))
        rs_sub = RewardSet(W, sub)
        for k in range(N_GREEDY_SEEDS):
            t0 = time.time()
            put("greedy_rew400_c120", greedy(W, rs_sub, dep["act"], B_DEPLOY,
                                             np.random.default_rng(9200 + 400 + 1000 * k), 120),
                extra={"n_reward_genes": 400, "n_cand": 120, "seed": k}, t0=t0)
    for k in range(N_GREEDY_SEEDS):
        t0 = time.time()
        put("cheat_rep_c120", greedy(W, rs_rep, dep["act"], B_DEPLOY,
                                     np.random.default_rng(9300 + 1000 * k), 120),
            extra={"cheats": True, "n_cand": 120, "seed": k}, t0=t0)
    # its strongest honest form: every aux gene as reward, every candidate scored
    # (only the first pick is random, so fewer seeds are needed)
    for k in range(N_GREEDY_ALL_SEEDS):
        t0 = time.time()
        put("greedy_aux_all", greedy(W, rs_aux, dep["act"], B_DEPLOY, np.random.default_rng(9201 + 1000 * k)),
            extra={"n_reward_genes": int(dep["aux"].size), "n_cand": "all", "seed": k}, t0=t0)
    for k in range(N_GREEDY_ALL_SEEDS):
        t0 = time.time()
        put("cheat_rep_all", greedy(W, rs_rep, dep["act"], B_DEPLOY, np.random.default_rng(9301 + 1000 * k)),
            extra={"cheats": True, "n_cand": "all", "seed": k}, t0=t0)
    put("done")
    return rows


# ======================================================================= torch
def _torch(n_threads):
    import torch
    torch.set_num_threads(n_threads)
    return torch


def make_mlp(torch, d, h):
    net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, h),
                              torch.nn.Tanh(), torch.nn.Linear(h, 1))
    with torch.no_grad():
        net[-1].weight.mul_(0.01)
        net[-1].bias.zero_()
    return net


def export_np(net):
    """The torch MLP as a numpy function, for fast single-decision rollouts."""
    P = [p.detach().cpu().numpy().astype(np.float32) for p in net.parameters()]
    W1, b1, W2, b2, W3, b3 = P

    def f(X):
        h = np.tanh(X @ W1.T + b1)
        h = np.tanh(h @ W2.T + b2)
        return (h @ W3.T + b3)[:, 0]
    return f


def seg_logp(torch, net, X, seg, nseg, act_rows):
    """Per-decision log pi(a|s) and entropy, softmax over each decision's candidates."""
    z = net(X).squeeze(-1)
    m = torch.full((nseg,), -1e30, dtype=z.dtype).scatter_reduce(0, seg, z.detach(), "amax")
    ex = torch.exp(z - m[seg])
    s = torch.zeros(nseg, dtype=z.dtype).index_add(0, seg, ex)
    lse = m + torch.log(s)
    p = ex / s[seg]
    ent = -torch.zeros(nseg, dtype=z.dtype).index_add(0, seg, p * (z - lse[seg]))
    return z[act_rows] - lse, ent


# ========================================================================= PPO
def train_ppo(W, dep, regime, seed, n_threads, curve_path, train_world=None, pool=None,
              na_target=None, rich=False):
    """Train on `train_world` (default: the deployment world W), deploy on W.

    Everything that selects -- the sub-problem sampler, the validation sub-problems,
    the checkpoint -- lives entirely in the training world. W is touched only to roll
    the chosen policy out on the real deployment split and to score it, which is the
    reported number. So with train_world set to the OTHER cell line, no part of
    training or model selection has seen this line's candidates, their signatures or
    its report genes -- the cleanest form of the generalisation claim, and the only
    way to get deployment-scale training sub-problems on K562, whose own visible
    atlas (473 genes) cannot spare 437 of them as candidates."""
    torch = _torch(n_threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    C = CFG
    WT = train_world if train_world is not None else W
    d = len(feature_names(regime, rich))
    net = make_mlp(torch, d, C["ppo_hidden"])
    opt = torch.optim.Adam(net.parameters(), lr=C["ppo_lr"])
    sampler = make_sampler(dep, regime, pool, na_target)
    # validation sub-problems: fixed across seeds, drawn the same way as training ones
    val = [sampler(np.random.default_rng(770_000 + i)) for i in range(C["n_val"])]
    val_rs = [RewardSet(WT, ep["rew"]) for ep in val]
    rep_fn = Reporter(W, dep["rep"], dep["act"])
    dep_ep = deploy_episode(dep)
    probe = sampler(np.random.default_rng(12345))      # a throwaway draw, to record the scale
    scale = {"train_dataset": WT.name, "pool": int(np.asarray(
        pool if pool is not None else (dep["aux"] if regime == "blind" else
                                       np.concatenate([dep["act"], dep["aux"]]))).size),
        "n_cand": int(probe["act"].size), "budget": int(probe["B"]),
        "n_reward": int(probe["rew"].size), "n_visible": int(probe["vis"].size),
        "deploy_n_cand": int(dep["act"].size), "deploy_budget": B_DEPLOY,
        "size_matched": na_target is not None, "rich": bool(rich)}
    log(f"  train scale: {scale}")

    def fl_logits(X):                     # facility-location reference on the same episodes
        return X[:, GO_FEATS.index("fl_gain")]

    def val_score(fn):
        return float(np.mean([rollout(WT, ep, regime, fn, np.random.default_rng(0), "greedy", rs,
                                      rich=rich)[1][-1]
                              for ep, rs in zip(val, val_rs)]))

    vr = np.random.default_rng(1)
    val_ref = {"facility": val_score(fl_logits),
               "random": float(np.mean([rs.mrr(ep["act"][vr.choice(ep["act"].size, ep["B"], replace=False)])
                                        for ep, rs in zip(val, val_rs) for _ in range(5)]))}
    log(f"  val reference: facility={val_ref['facility']:.4f} random={val_ref['random']:.4f}")

    best = {"val": -1.0}
    curve = []

    def checkpoint(it):
        fn = export_np(net)
        v = val_score(fn)
        ch = rollout(W, dep_ep, regime, fn, np.random.default_rng(0), "greedy", rich=rich)[0]
        r = rep_fn(ch)
        row = {"iter": it, "val_mrr": v, "rep_mrr": r["mrr"], "rep_top1": r["top1"], **val_ref}
        curve.append(row)
        with open(curve_path, "w") as f:
            for x in curve:
                f.write(json.dumps(x) + "\n")
        if v > best["val"]:
            best.update(val=v, iter=it, chosen=ch, rep=r,
                        state={k: t.detach().clone() for k, t in net.state_dict().items()})
        log(f"  it={it:4d} val={v:.4f} (fac {val_ref['facility']:.4f}) deploy rep mrr={r['mrr']:.4f}")

    checkpoint(0)
    for it in range(1, C["ppo_iters"] + 1):
        t0 = time.time()
        fn = export_np(net)
        Xs, ks, advs, finals, ep_dec = [], [], [], [], []
        for _ in range(C["ppo_univ"]):
            ep = sampler(rng)
            rs = RewardSet(WT, ep["rew"])
            M, recs = [], []
            for _g in range(C["ppo_group"]):
                _, mr, rec = rollout(WT, ep, regime, fn, rng, "sample", rs, True, C["t_max"], rich)
                M.append(mr)
                recs.append(rec)
            M = np.array(M)
            prev = np.concatenate([np.zeros((M.shape[0], 1)), M[:, :-1]], 1)
            rtg = M[:, -1:] - prev            # return-to-go of each round's picks
            A = (rtg - rtg.mean(0)) / np.maximum(rtg.std(0), 1e-3)
            finals += list(M[:, -1])
            for g, rec in enumerate(recs):
                d0 = len(Xs)
                for X, k, rd in rec:
                    Xs.append(X)
                    ks.append(k)
                    advs.append(A[g, rd])
                ep_dec.append((d0, len(Xs)))
        sizes = np.array([x.shape[0] for x in Xs])
        offs = np.concatenate([[0], np.cumsum(sizes)])
        ks = np.array(ks)
        Xall = np.concatenate(Xs)
        adv_all = np.array(advs, np.float32)
        nseg = len(Xs)
        with torch.no_grad():
            old_all, _ = seg_logp(torch, net, torch.from_numpy(Xall),
                                  torch.from_numpy(np.repeat(np.arange(nseg), sizes)), nseg,
                                  torch.from_numpy(offs[:-1] + ks))
        old_all = old_all.numpy()
        # PPO epochs over minibatches of whole episodes (an episode's rows are contiguous)
        n_mb = max(1, min(8, len(ep_dec)))
        for _e in range(C["ppo_epochs"]):
            for mb in np.array_split(rng.permutation(len(ep_dec)), n_mb):
                decs = np.concatenate([np.arange(*ep_dec[i]) for i in mb])
                rows = np.concatenate([np.arange(offs[ep_dec[i][0]], offs[ep_dec[i][1]]) for i in mb])
                sz = sizes[decs]
                lo = np.concatenate([[0], np.cumsum(sz)])[:-1]
                X = torch.from_numpy(Xall[rows])
                seg = torch.from_numpy(np.repeat(np.arange(decs.size), sz))
                lp, ent = seg_logp(torch, net, X, seg, decs.size, torch.from_numpy(lo + ks[decs]))
                adv = torch.from_numpy(adv_all[decs])
                ratio = torch.exp(lp - torch.from_numpy(old_all[decs]))
                pg = -torch.min(ratio * adv,
                                torch.clamp(ratio, 1 - C["ppo_clip"], 1 + C["ppo_clip"]) * adv).mean()
                loss = pg - C["ppo_ent"] * ent.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        if it % 5 == 0 or SMOKE:
            log(f"  it={it:4d} train final mrr={np.mean(finals):.4f} ent={ent.mean().item():.3f} "
                f"decisions={nseg} rows={Xall.shape[0]} ({time.time() - t0:.1f}s)")
        if it % C["val_every"] == 0 or it == C["ppo_iters"]:
            checkpoint(it)
    last = curve[-1]
    np.savez(curve_path.replace(".curve.jsonl", ".policy.npz"),
             **{k: v.numpy() for k, v in best["state"].items()},
             features=np.array(feature_names(regime, rich)))
    return {"chosen": best["chosen"], "rep": best["rep"], "best_iter": best["iter"],
            "best_val": best["val"], "val_ref": val_ref, "last_rep_mrr": last["rep_mrr"],
            "last_val_mrr": last["val_mrr"], "features": feature_names(regime, rich),
            "train_scale": scale}


# ================================================================ tabular PG
def train_pg_tab(W, dep, seed, boot, curve_path):
    """REINFORCE, leave-one-out baseline, Adam, on Plackett-Luce logits theta.

    log PL(a_1..a_B) = sum_t theta_{a_t} - log Z_t, Z_t = sum over still-available.
    d/dtheta_j = [j chosen] - u_j * sum_{t <= tau(j)} 1/Z_t, u = exp(theta), tau(j)
    the step j was taken (or B-1 if never). O(nA + B) per sample."""
    C = CFG
    rng = np.random.default_rng(seed)
    act, aux = dep["act"], dep["aux"]
    nA, B = act.size, B_DEPLOY
    theta = np.zeros(nA)
    m1, m2 = np.zeros(nA), np.zeros(nA)
    rs_full = RewardSet(W, aux)
    rep_fn = Reporter(W, dep["rep"], act)
    curve = []
    for it in range(1, C["pg_iters"] + 1):
        orders, R = [], []
        for _ in range(C["pg_samples"]):
            o = np.argsort(-(theta + rng.gumbel(size=nA)))[:B]
            rs = RewardSet(W, np.sort(rng.choice(aux, size=aux.size // 2, replace=False))) if boot else rs_full
            orders.append(o)
            R.append(rs.mrr(act[o]))
        R = np.array(R)
        n = R.size
        adv = R - (R.sum() - R) / (n - 1)
        u = np.exp(theta - theta.max())
        grad = np.zeros(nA)
        for o, a in zip(orders, adv):
            Z = u.sum() - np.concatenate([[0.0], np.cumsum(u[o])[:-1]])
            cz = np.cumsum(1.0 / Z)
            tau = np.full(nA, B - 1)
            tau[o] = np.arange(B)
            g = -u * cz[tau]
            g[o] += 1.0
            grad += a * g
        grad /= n
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad ** 2
        theta += C["pg_lr"] * (m1 / (1 - 0.9 ** it)) / (np.sqrt(m2 / (1 - 0.999 ** it)) + 1e-8)
        if it % 50 == 0 or it == C["pg_iters"] or SMOKE:
            mode = act[np.argsort(-theta)[:B]]
            r = rep_fn(mode)
            row = {"iter": it, "sample_reward_mean": float(R.mean()), "mode_own_reward": rs_full.mrr(mode),
                   "rep_mrr": r["mrr"], "rep_top1": r["top1"]}
            curve.append(row)
            with open(curve_path, "w") as f:
                for x in curve:
                    f.write(json.dumps(x) + "\n")
            log(f"  it={it:4d} sample R={R.mean():.4f} mode own={row['mode_own_reward']:.4f} "
                f"rep mrr={r['mrr']:.4f}")
    mode = act[np.argsort(-theta)[:B]]
    return {"chosen": [int(x) for x in mode], "rep": rep_fn(mode), "own_reward": rs_full.mrr(mode),
            "boot": bool(boot)}


# ==================================================================== critic
def train_critic(W, dep, seed, n_threads, curve_path):
    """Fitted Q at gamma = 0: regress the HELD-OUT one-step gain of adding a gene on
    its features, pooled over many re-split sub-problems. Then deploy (a) greedily on
    the critic alone -- no reward evaluated at deployment at all -- and (b) on a
    shrinkage of greedy's noisy empirical gain toward the critic, weights fitted on
    validation sub-problems."""
    torch = _torch(n_threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    C = CFG
    regime = "peek"
    sampler = make_sampler(dep, regime)
    rows_X, y_rew, y_emp, sid, uid = [], [], [], [], []
    t0 = time.time()
    state = 0
    for u in range(C["critic_univ"]):
        ep = sampler(rng)
        rs_rew, rs_emp = RewardSet(W, ep["rew"]), RewardSet(W, ep["vis"])
        fs = FeatureState(W, ep["act"], ep["vis"], regime, ep["B"], rng, C["t_max"])
        stops = set(rng.choice(np.arange(3, ep["B"]), size=min(C["critic_states"], ep["B"] - 3),
                               replace=False).tolist())
        sizes = round_sizes(ep["B"])
        bounds = np.cumsum([0] + sizes)
        for step in range(ep["B"]):
            rd = int(np.searchsorted(bounds, step, side="right") - 1)
            if step == bounds[rd]:
                fs.begin_round(rd)
            if step in stops:
                X, av = fs.features()
                pick = rng.choice(av.size, size=min(C["critic_cands"], av.size), replace=False)
                b_r, b_e = rs_rew.mrr(fs.chosen), rs_emp.mrr(fs.chosen)
                for k in pick:
                    c = fs.chosen + [int(fs.A[av[k]])]
                    rows_X.append(X[k])
                    y_rew.append(rs_rew.mrr(c) - b_r)
                    y_emp.append(rs_emp.mrr(c) - b_e)
                    sid.append(state)
                    uid.append(u)
                state += 1
            # behaviour policy: half random, half facility location -> realistic states
            g, av = fs.raw_gain()
            k = int(np.argmax(g)) if rng.random() < 0.5 else int(rng.integers(av.size))
            fs.add(av[k])
        if u % 25 == 0:
            log(f"  critic data: universe {u}/{C['critic_univ']} rows={len(y_rew)} ({time.time()-t0:.0f}s)")
    X = np.array(rows_X, np.float32)
    y_rew, y_emp = np.array(y_rew), np.array(y_emp)
    sid, uid = np.array(sid), np.array(uid)

    def centre(v):                         # only the ranking WITHIN a state matters to a policy
        mu = np.bincount(sid, weights=v) / np.maximum(np.bincount(sid), 1)
        return v - mu[sid]
    yr, ye = centre(y_rew), centre(y_emp)
    scale = float(yr.std() + 1e-12)
    tr = (uid % 5) != 0                    # whole sub-problems held out
    va = ~tr
    net = make_mlp(torch, X.shape[1], C["ppo_hidden"])
    with torch.no_grad():
        net[-1].weight.mul_(100.0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(X), torch.from_numpy((yr / scale).astype(np.float32))
    best = (np.inf, None)
    curve = []
    for e in range(C["critic_epochs"]):
        net.train()
        loss = ((net(Xt[tr]).squeeze(-1) - yt[tr]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            vl = float(((net(Xt[va]).squeeze(-1) - yt[va]) ** 2).mean())
        if vl < best[0]:
            best = (vl, {k: t.detach().clone() for k, t in net.state_dict().items()}, e)
        if e % 25 == 0 or e == C["critic_epochs"] - 1:
            curve.append({"epoch": e, "train_mse": float(loss), "val_mse": vl})
    net.load_state_dict(best[1])
    with open(curve_path, "w") as f:
        for x in curve:
            f.write(json.dumps(x) + "\n")
    with torch.no_grad():
        qhat = centre(net(Xt).squeeze(-1).numpy().astype(np.float64) * scale)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    # control variate: yr ~ a*qhat + b*ye on held-out sub-problems
    Z = np.stack([qhat[va], ye[va]], 1)
    coef = np.linalg.lstsq(Z, yr[va], rcond=None)[0]
    diag = {"n_rows": int(len(yr)), "n_states": int(state), "best_epoch": int(best[2]),
            "corr_emp_vs_heldout_gain": corr(ye[va], yr[va]),
            "corr_critic_vs_heldout_gain": corr(qhat[va], yr[va]),
            "corr_shrink_vs_heldout_gain": corr(Z @ coef, yr[va]),
            "shrink_coef_critic": float(coef[0]), "shrink_coef_emp": float(coef[1])}
    log(f"  critic diag: {diag}")
    qfn = export_np(net)
    rep_fn = Reporter(W, dep["rep"], dep["act"])
    # (a) greedy on the critic alone
    ch_a = rollout(W, deploy_episode(dep), regime, qfn, np.random.default_rng(0), "greedy")[0]
    # (b) shrinkage greedy: every candidate's empirical gain on aux, blended with the critic
    rs_aux = RewardSet(W, dep["aux"])
    fs = FeatureState(W, dep["act"], dep["aux"], regime, B_DEPLOY)
    bounds = np.cumsum([0] + round_sizes(B_DEPLOY))
    for step in range(B_DEPLOY):
        rd = int(np.searchsorted(bounds, step, side="right") - 1)
        if step == bounds[rd]:
            fs.begin_round(rd)
        Xd, av = fs.features()
        q = qfn(Xd).astype(np.float64) * scale
        if fs.chosen:
            b0 = rs_aux.mrr(fs.chosen)
            emp = np.array([rs_aux.mrr(fs.chosen + [int(fs.A[i])]) for i in av]) - b0
        else:
            emp = np.zeros(av.size)
        sc = coef[0] * (q - q.mean()) + coef[1] * (emp - emp.mean())
        fs.add(av[int(np.argmax(sc))])
    ch_b = fs.chosen
    return {"critic_only": {"chosen": ch_a, "rep": rep_fn(ch_a)},
            "critic_shrink": {"chosen": ch_b, "rep": rep_fn(ch_b)}, "diag": diag}


# ======================================================================== grid
def grid():
    ds = env_list("PJ_RLA_DATASETS", DATASETS)
    sp = env_list("PJ_RLA_SPLITS", (4, 104, 204), int)
    me = env_list("PJ_RLA_METHODS", METHODS)
    se = env_list("PJ_RLA_SEEDS", (0, 1, 2), int)
    return ([(d, s, m, z) for d in ds for s in sp for m in me for z in se],
            [(d, s) for d in ds for s in sp])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("baselines", "train", "grid"))
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()
    train_grid, base_grid = grid()
    if a.mode == "grid":
        print(json.dumps({"n_train": len(train_grid), "n_baselines": len(base_grid),
                          "train": train_grid, "baselines": base_grid}))
        return 0

    od = os.path.join(a.out_dir, "rl_acquire_smoke" if SMOKE else "rl_acquire")
    os.makedirs(os.path.join(od, "runs"), exist_ok=True)
    os.makedirs(os.path.join(od, "baselines"), exist_ok=True)
    check = json.load(open(os.path.join(a.cache_dir, "cache_check.json")))

    def world(name):
        info = check["datasets"].get(name)
        if not info or not info.get("ok"):
            # A failed gate, not a crash: say why, exit 0 (see CLAUDE.md).
            print(f"GATE NOT MET: cache for {name} missing or did not reproduce GAPS's random median: {info}")
            return None
        return World(info["path"])

    if a.mode == "baselines":
        ds, sp = base_grid[a.task_index]
        out = os.path.join(od, "baselines", f"{ds}__s{sp}.jsonl")
        if os.path.exists(out) and not os.environ.get("PJ_RLA_FORCE"):
            done = [json.loads(x)["strategy"] for x in open(out)]
            if "done" in done:                   # written last
                log(f"baselines already complete, skipping: {out}")
                return 0
        W = world(ds)
        if W is None:
            return 0
        log(f"baselines: {ds} split {sp} -> {out}")
        run_baselines(W, sp, out)
        return 0

    ds, sp, me, seed = train_grid[a.task_index]
    tag = f"{me}__{ds}__s{sp}__seed{seed}"
    out = os.path.join(od, "runs", tag + ".json")
    curve = os.path.join(od, "runs", tag + ".curve.jsonl")
    if os.path.exists(out) and not os.environ.get("PJ_RLA_FORCE"):
        log(f"already done, skipping (PJ_RLA_FORCE=1 to redo): {out}")
        return 0
    W = world(ds)
    if W is None:
        return 0
    dep = split_of(W.K, sp)
    log(f"train {tag}: act={dep['act'].size} aux={dep['aux'].size} rep={dep['rep'].size} cfg={CFG}")
    t0 = time.time()
    base = {"dataset": ds, "split": sp, "seed": seed, "budget": B_DEPLOY, "K_report": int(dep["rep"].size),
            "cfg": CFG, "smoke": SMOKE}
    try:
        if me in ("pg_tab", "pg_tab_boot"):
            r = train_pg_tab(W, dep, seed, me == "pg_tab_boot", curve)
            res = [dict(base, method=me, regime="peek", **{f"rep_{k}": v for k, v in r.pop("rep").items()}, **r)]
        elif me == "ppo_peek" or me in BLIND_VARIANTS:
            regime = "peek" if me == "ppo_peek" else "blind"
            kw = {}
            if me in BLIND_VARIANTS:
                v = BLIND_VARIANTS[me]
                kw["rich"] = v["rich"]
                if v["pool"] == "all":
                    kw["pool"] = np.concatenate([dep["act"], dep["aux"]])
                elif v["pool"] == "other":
                    WO = world(OTHER[ds])
                    if WO is None:
                        return 0          # a failed gate, not a crash (see CLAUDE.md)
                    kw["train_world"] = WO
                    kw["pool"] = np.arange(WO.K)
                if v["match"]:
                    kw["na_target"] = int(dep["act"].size)
            r = train_ppo(W, dep, regime, seed, a.threads, curve, **kw)
            res = [dict(base, method=me, regime=regime, **{f"rep_{k}": v for k, v in r.pop("rep").items()}, **r)]
        elif me == "critic":
            r = train_critic(W, dep, seed, a.threads, curve)
            res = [dict(base, method=k, regime="peek", diag=r["diag"],
                        **{f"rep_{kk}": v for kk, v in r[k]["rep"].items()}, chosen=r[k]["chosen"])
                   for k in ("critic_only", "critic_shrink")]
        else:
            raise ValueError(me)
    except Exception as e:
        traceback.print_exc()
        res = [dict(base, method=me, error=f"{type(e).__name__}: {e}")]
    for x in res:
        x["secs"] = round(time.time() - t0, 1)
        x["chosen"] = [int(c) for c in x.get("chosen", [])]
        log(f"RESULT {x['method']:14s} rep mrr={x.get('rep_mrr', float('nan')):.4f} "
            f"top1={x.get('rep_top1', float('nan')):.4f}")
    if any("error" in x for x in res):
        # Never write the result file on a crash: the resume check would then skip
        # this task forever. A real crash exits 1 so the array task shows FAILED.
        with open(os.path.join(od, "runs", tag + ".error.json"), "w") as f:
            json.dump(res, f)
        return 1
    with open(out + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(out + ".tmp", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
