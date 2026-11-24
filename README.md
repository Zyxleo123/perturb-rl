# Inverse design of genetic perturbations — baseline acquisition & feasibility probe

Status as of **2026-11-13**. Covers the proposal's first milestone ("Now ~ Oct 31:
download baseline code, dataset(s) and baseline forward/inverse model weights and a
working environment") plus a feasibility probe of the dataset assumptions.

Code lives here (small, versionable). Everything large lives on scratch:

```
/zfsauton2/home/yixiz/10703_pj/        this repo: jobs, scripts, env
/zfsauton/scratch/yixiz/10703_pj/
  code/        cloned baselines (untrusted third-party)
  data/        downloaded datasets (untrusted)
  weights/     downloaded checkpoints (untrusted)
  artifacts/   OUR computed results: json / jsonl / npz
  figures/     OUR plots, redrawable from artifacts alone
  logs/        slurm .out
  envfiles/    one env file per submission
```

## Quick start

```bash
cd /zfsauton2/home/yixiz/10703_pj
./submit.sh              # full DAG
./submit.sh fetch        # downloads + env build only
./submit.sh analyze      # profile + oracle probe + figures
./submit.sh figures      # redraw figures from existing artifacts (cheap)
./submit.sh gen          # generalization headroom (unseen gene/pair/context) + its figures
./submit.sh genfigs      # redraw only fig13-fig14 (cheap)
./submit.sh rl           # is the unseen-gene gap RL-shaped? (PJ_RL_PARTS="1 2" selects parts)
./submit.sh rlacq        # RL on the acquisition gap: cache -> baselines + training array -> report
./submit.sh rlacq_base   # baselines array only (PJ_RLA_FORCE=1 to add a new strategy)
./submit.sh rlacq_runs   # training array + report only (baselines already there)
./submit.sh rlacq_report # redraw the RL table/figures from saved runs (cheap)
PJ_RLA_SMOKE=1 ./submit.sh rlacq_smoke   # 3-iteration code-path check, no report
./submit.sh related      # published methods on the RL task: fetch IterPert + kernels -> priors -> array -> report
./submit.sh related_runs # array + report only (code, kernels and priors already there)
PJ_REL_SMOKE=1 PJ_RLA_SPLITS=4 PJ_REL_SEEDS=0 ./submit.sh related_smoke   # every method once, 300 reward evals
./submit.sh story        # gap / improvement case studies: per-gene data (CPU, ~11 min) -> fig18-fig21
./submit.sh storyfigs    # redraw fig18-fig21 from artifacts/story/ only (cheap)
./submit.sh paperfigs    # re-render fig13/15/16/17 in the fig18-21 journal style (cheap)
./submit.sh repr         # why the unseen-gene predictor fails: probe -> pred -> sel -> {curve, calib, gram} -> fig22 (CPU, ~1.5 h)
python3 scripts/repr_splice.py   # then regenerate GAPS.md's generated repr block from those artifacts (login node, no compute)
./submit.sh recon        # re-check a per-gene reconciliation failure (diagnostic; writes nothing)
./submit.sh sim          # sim-to-real acquisition: RL trained inside the other cell line, deployed here; baselines + runs -> report (CPU)
./submit.sh sim_report   # redraw the sim table from saved runs (cheap)
./submit.sh sim_readout  # readout gene names behind the RL caches, verified (needed once, before the _timp simulator)
python3 scripts/sim_splice.py    # then regenerate GAPS.md's sim-to-real section + summary paragraph from the reports (login node, no compute)
squeue -u $USER -o '%.8i %.9P %.16j %.8T %.10M %R'
```

Config is passed as **one timestamped env file as a positional argument**, never via
`sbatch --export` (an explicit export list sets `SLURM_GET_USER_ENV=1`; the login-env
rebuild fails on the compute node and the job is requeued+held). Defaults live once in
[env/common.sh](env/common.sh).

## What is downloaded

### Code (`scratch/code/`, SHAs in `artifacts/code_provenance.json`)

| repo | role | ref / date |
|---|---|---|
| [snap-stanford/GEARS](https://github.com/snap-stanford/GEARS) | forward oracle (reward + cycle consistency) | `f374e43` 2025-02-01 |
| [mims-harvard/PDGrapher](https://github.com/mims-harvard/PDGrapher) | supervised inverse baseline, Nature BME 2025 | `41285f9` 2025-10-02 |
| [theislab/LabCompass](https://github.com/theislab/LabCompass) | **SOTA inverse design** (flow matching + guidance) | `cc3d73b` 2026-10-28 |
| [theislab/cellflow](https://github.com/theislab/cellflow) | flow-matching perturbation backbone | `446ed60` 2026-07-12 |
| [ArcInstitute/state](https://github.com/ArcInstitute/state) | SOTA forward oracle (Arc virtual cell) | `9bbfe78` 2026-07-23 |

LabCompass is the paper the proposal calls "LabCompass's per-target differentiable
guidance of a frozen prior" — Paulas-Condori et al., *Generative inverse design steers
human hematopoietic cell fate in vitro*, bioRxiv `2026.10.07.756550`. Code was published
2026-10-28, after the proposal was written; it is the direct competitor and is read
closely in [GAPS.md](GAPS.md).

### Datasets (`scratch/data/`)

GEARS-processed perturb-seq, from Harvard Dataverse (ids read out of
`GEARS@master gears/pertdata.py`):

| dataset | cells | genes | perturbed conditions | file |
|---|---|---|---|---|
| norman | 91,205 | 5,045 | 283 (152 single + 131 combo) | 2.1 G |
| adamson | 68,603 | 5,060 | 86 | 579 M |
| dixit | 44,735 | 5,012 | 19 | 677 M |
| replogle_k562_essential | 162,751 | 5,000 | 1,092 | 2.7 G |
| replogle_rpe1_essential | 162,733 | 5,000 | 1,543 | 2.7 G |

Plus `gene2go_all.pkl` (9.1 M), `essential_all_data_pert_genes.pkl` (546 K),
`go_essential_all.csv` (58 M).

PDGrapher processed data from Zenodo [15375990](https://zenodo.org/records/15375990):
`splits.tar.gz` (6.6 M) + `torch_data_genetic.tar.gz` (21.7 G). Genetic splits cover
10 cell lines — A375, A549, AGS, BICR6, ES2, HT29, MCF7, PC3, U251MG, YAPC — as
`genetic/<line>/random/5fold/splits.pt`, with `torch_data/real_lognorm/data_{forward,
backward}_<LINE>.pt` (4–6 G each). The 37.3 G chemical tarball
([15390483](https://zenodo.org/records/15390483)) is **not** fetched by default;
`FETCH_PDG_CHEMICAL=1 ./submit.sh fetch` gets it.

> **Known issue:** the first extraction exited 1 partway through and left
> `data_backward_HT29.pt` at **0 bytes** while the other nine cell lines extracted
> fine. `jobs/12b_repair_pdgrapher.sbatch` re-extracts and then verifies every member,
> so a partial extraction cannot look like success. If HT29 is still empty afterwards,
> check the Zenodo record itself — but note the paper reports HT29 results, so an empty
> file there would be suspicious.

### Weights (`scratch/weights/`)

`gears_norman_official/` — the **official** GEARS Norman checkpoint
(`model_ckpt/model.pt` 8.9 M + `config.pkl`) and its matching dataloader
`norman_umi_go/` (3.2 G h5ad + 3.7 G cell-graph pickle), from Dataverse ids
`10457098` and `6979957` as used by `GEARS/demo/tutorial_inference_Norman.ipynb`.

> The checkpoint is `gears_misc_umi_no_test`, i.e. **split='no_test'** — trained on
> every Norman perturbation. It cannot give held-out oracle numbers. See
> [GAPS.md](GAPS.md) gap 6 and `jobs/22_retrain_oracle.sbatch`.

## Environments

| env | python | contents | used by |
|---|---|---|---|
| `/zfsauton/scratch/yixiz/envs/cellflow` | 3.12 | scanpy 1.12, anndata 0.13, numpy 2.4 | profiling, figures |
| `/zfsauton/scratch/yixiz/envs/pertrl` | 3.11 | torch 2.4.1+cu121, PyG 2.6.1, scanpy 1.10, **numpy 1.26**, cell-gears 0.1.2 | GEARS oracle |

`pertrl` pins `numpy<2` deliberately: `essential_all_data_pert_genes.pkl` is a pickle
referencing `numpy.core.multiarray`, a path removed in numpy 2.x, so GEARS's own data
files cannot be loaded under the numpy the analysis env uses. cu121 wheels cover every
GPU in this pool (sm_70 v100 → sm_90 h200).

## Jobs

| job | partition | what |
|---|---|---|
| `10_fetch_code` | cpu | clone the 5 baselines, record SHAs |
| `11_fetch_gears_data` | cpu | perturb-seq datasets + aux + official oracle |
| `12_fetch_pdgrapher_data` | cpu | PDGrapher splits + genetic torch_data |
| `13_build_torch_env` | cpu | build `pertrl` via uv; always exits 0, writes a status file |
| `14_fetch_related` | cpu | clone IterPert + bmdal_reg (SHAs in `artifacts/code_provenance_related.json`), IterPert's prior kernels (MD5-checked) |
| `12b_repair_pdgrapher` | cpu | re-extract + verify every `.pt` member |
| `20_profile` | cpu | dataset stats + identifiability → artifacts |
| `21_oracle_probe` | legacy (1 GPU) | GEARS vs trivial baselines, top-20 DE genes |
| `22_retrain_oracle` | legacy (1 GPU) | retrain under `split='simulation'`, re-probe held out |
| `23_inverse_gap` | legacy (1 GPU) | the ladder: ceiling / oracle / non-conditional / chance |
| `24_trivial_rule` | cpu | is identification just reading the knocked-down gene? |
| `25_generalization` | cpu | headroom on unseen gene / unseen pair / unseen context |
| `26_rl_headroom` | cpu | is the unseen-gene gap RL-shaped? objective / acquisition / adaptivity / reward noise |
| `40_rl_cache` | cpu | cache the acquisition-MDP inputs; gate: reproduces GAPS part 4's random median |
| `41_rl_baselines` | cpu, array | random x100, GO coverage, facility location (3 prior views), greedy and cheat over seeds |
| `42_rl_train` | cpu, array | RL runs: REINFORCE (per-gene logits), PPO feature policy (peek/blind + 4 blind variants), learned critic |
| `43_rl_report` | cpu | re-score every chosen set in GAPS's env, table, fig15-fig17 |
| `44_related_prep` | cpu | IterPert's priors re-expressed on our candidates (numpy-only unpickler, kernel = feat featᵀ checked) |
| `45_related_run` | cpu, array | IterPert + its AL benchmark through IterPert's own code (blind); data valuation + exemplars (peek) |
| `46_story_data` | cpu | case-study data: per-gene ranks, kNN neighbourhoods, greedy's per-step gains → `artifacts/story/`; reconciles with `records.jsonl` |
| `47_story_figs` | cpu | fig18-fig21 from `artifacts/story/` only |
| `48_paper_figs` | cpu | fig13/15/16/17 re-rendered in the fig18-21 journal style, from saved artifacts |
| `50_repr_probe` | cpu | why GO-kNN fails on unseen genes: predictor-class ceilings, GO's neighbour ordering vs the true one, CSLS / hubness |
| `51_repr_predict` | cpu | the predictor's similarity swapped for IterPert's 8 prior kernels, two richer GO views and a learned combination |
| `52_repr_select` | cpu | ranking vs combining neighbours: kNN average, ridge over the atlas Gram, MMR; hyper-parameters out of fold |
| `53_repr_curve` | cpu | similarity quality swept continuously -> retrieval; the real representations placed on the same axes |
| `54_repr_figs` | cpu | fig22 + paired Wilcoxon over genes, from `artifacts/repr_*/` only |
| `55_repr_calib` | cpu | calibration ablation: the oracle similarity raw vs standardised (same ordering, different magnitudes) |
| `56_repr_gram` | cpu | estimate the combiner's inner products instead of plugging them in: neighbourhood x b 2x2, then a learned b |
| `57_recon_check` | cpu | bisect a per-gene reconciliation failure and test whether it is a tie-boundary flip; writes nothing |
| `60_sim_base` | cpu, array | sim-to-real baselines per (line, split, simulator): reconciliation with `41`'s random median and GO facility set, greedy on the simulator reward, facility on source signatures, transfer probe |
| `61_sim_train` | cpu, array | REINFORCE / critic / PPO trained inside the other line's screen on the deployment candidates; four simulators (`all`, `_tpop`, `_tpred`, `_timp`); combined reward (simulator + the target's GO facility objective) and its all-candidates hybrid, each with greedy on the identical reward |
| `62_sim_report` | cpu | re-score every chosen set in GAPS's env, paired tests vs the floors → `artifacts/rl_acquire/sim/{report,summary}{,_confirm,_confirm2,_confirm3}.*` (`PJ_SIM_REPORT_SPLITS` / `PJ_SIM_REPORT_TAG` select the held-out sets) |
| `63_sim_readout` | cpu | recover the readout gene names behind the caches' signature columns, verified against the cache (masked counts; signature rows recomputed from the h5ad) |
| `30_figures` | cpu | aggregate figures, drawn from artifacts |
| `31_case_studies` | cpu | named individual perturbations rather than metrics |
| `33_generalization_figures` | cpu | redraw fig13-fig14 from `artifacts/generalization/` |

### Figures

| file | shows |
|---|---|
| `fig1_action_space` | perturbable genes, labelled conditions, oracle visibility |
| `fig2_identifiability` | nearest-centroid retrieval vs cells pooled |
| `fig3_effect_and_confusability` | single-cell SNR, signature collinearity, DE breadth |
| `fig4_data_budget` | cells per condition, control pool |
| `fig5_signature_space` | PCA of Δ-signatures, effective dimension |
| `fig6_oracle_quality` | GEARS vs control-mean vs mean-perturbed, top-20 DE |
| `fig7_published_gap` | every published baseline vs chance (CDN Table 1) |
| `fig8_inverse_ladder` | ceiling / oracle / non-conditional / chance |
| `fig9_trivial_rule` | where the perturbed gene's own \|Δ\| ranks |
| **`fig10_barcode_cases`** | **one representative perturbation per dataset, target marked** |
| **`fig11_oracle_cases`** | **named perturbations: measured vs GEARS vs average effect** |
| **`fig12_retrieval_cases`** | **what the oracle ranks above the right answer** |
| **`fig13_headroom_by_axis`** | **ceiling vs best cheap predictor: unseen gene / pair / context** |
| **`fig14_context_transfer`** | **few-shot cross-line transfer + is it even the same effect** |
| **`fig15_rl_acquisition`** | **RL vs greedy vs facility location on gene acquisition, per split x seed; the five blind variants sit on the static floor** |
| **`fig16_rl_curves`** | **PPO learning curves: validation vs facility location, report set (diagnostic)** |
| **`fig17_related_work`** | **published acquisition methods (IterPert, Core-Set, BALD, BADGE, LCMD, TypiClust, Data Shapley, ...) beside our RL, same task** |
| **`fig18_gap_crowd`** | **the gap, one gene at a time: a held-out target vs all 1,092 measured signatures, then vs their GO-kNN predictions** |
| **`fig19_rescue_flow`** | **where the RL gain lands: every report gene's rank under a typical random atlas → REINFORCE's atlas** |
| **`fig20_rescue_anatomy`** | **one rescue dissected: the 10 neighbours each atlas lends ATP6V0C, and the readout they predict** |
| **`fig21_greedy_noise`** | **one greedy decision: gain on the genes it is scored on vs gain on held-out genes, for every candidate** |

`fig10`–`fig12` are case studies: named perturbations, with the selection rule printed on
every panel and representative cases picked by **median** rank rather than cherry-picked.
`fig18`–`fig21` tell the gap / improvement story the same way, in journal format (7-inch
width, STIX serif, panel letters, no titles; captions live in GAPS.md, and a vector PDF sits
beside each PNG). Their selection rules are deterministic and live in
`scripts/story_draw.py`: the 25th/50th/75th-percentile genes (fig18), the largest rescue
among genes REINFORCE ranks #1 (fig20), and the decision whose correlation is the median
(fig21).

### Two traps worth knowing about

**`PertData.load()` unpickles a 3.7 GB cell-graph cache unconditionally**
(`pertdata.py` ~l.200), which took **over 30 minutes** on this NFS mount under load and
is never read by `predict()`. `PJ_ORACLE_FAST=1` / `PJ_INVGAP_FAST=1` intercept that one
filename and set `node_map`/`gene_names` exactly as `get_dataloader()` would
(`pertdata.py` l.429-430), cutting startup to **11 seconds**. `set_pert_genes()` and
`prepare_split()` still run on the real `PertData`, so the perturbation indexing stays
GEARS's own.

**`uns['top_non_dropout_de_20']` holds Ensembl IDs, not gene names**, in the
`norman_umi_go` release — while `adata.var` carries both `gene_id` and `gene_name`.
Matching the wrong one silently empties every DE set, and the probe then reports
`status: ok` with zero rows. `scripts/oracle_probe.py` now indexes both and counts
unresolved conditions so this cannot pass as success.

Conventions followed throughout: GPU work only where a GPU is needed and everything
that just reads saved artefacts runs on `cpu`; figures are redrawable without
recomputation; every job prints `hostname`, `nvidia-smi -L` and its resource request
(there is no `sacct` here, so the node is otherwise unrecoverable after the fact);
`set -euo pipefail` sits in each job body, not only the preamble; and a job whose gate
fails prints why and **exits 0** rather than stranding dependents in
`DependencyNeverSatisfied`.

Two environment facts measured here, worth keeping: `unzip` is **not installed** on the
compute nodes (extraction goes through `pj_unzip`, a python `zipfile` helper), and `du`
misreports individual file sizes on this NFS mount (the summaries use `stat -c%s`).

A third, measured 2026-11-23: `/zfsauton/scratch` is NFS v4 mounted with
`acregmin=acdirmin=600`, so a file a compute node has just written can be **invisible on
the login node for up to ten minutes** -- to `ls`, `stat` and any script alike. Reading a
job's outputs right after it finishes can silently see nothing. Anything that consumes job
artifacts on the login node must fail when an expected file is absent rather than fall
back (`scripts/repr_splice.py` does), or run as a dependent job instead.

## Findings

**[GAPS.md](GAPS.md)** — seven gaps ordered by headroom, each with its figure inline, plus
the novelty re-scoping, recommendations and the evaluation traps worth knowing about.

Plots live in two places: [figures/](figures/) in this repo (checked in, 1.8 MB, so the
markdown renders anywhere) and `$PJ_FIGURES` on scratch, which is what
`jobs/30_figures.sbatch` and `jobs/31_case_studies.sbatch` write. After re-running either,
refresh the copies here:

```bash
cp /zfsauton/scratch/yixiz/10703_pj/figures/*.png figures/
```

The headline: the measured ceiling on Norman is **86.6%** top-1 from a 128-cell target,
the GEARS-driven oracle rung is **8.3%**, and GEARS beats a predictor that ignores the
perturbation's identity on only **44.8%** of conditions. The bottleneck is the reward
model, not the generator.

Gap 0 then asks *where* that headroom actually lives, since a perturbation the atlas has
already measured needs no model at all. Of the three ways to be unseen, only **unseen
gene** is a gap: unseen pairs are closed outright by adding two single-gene signatures
(**71.6%** vs a **71.1%** ceiling, while GEARS trained on those very combos gets 16.1%),
and cross-cell-line transfer is partly irreducible (disattenuated cross-context cosine
median **0.42**). On the unseen-gene axis every cheap predictor — GO kNN, co-response kNN,
the published linear baseline — sits at **0.8–1.9%** top-1 against a **32.5–88.9%**
ceiling.

The RL work on that axis -- choosing which genes to measure -- is in GAPS.md's "RL on the
acquisition gap", "The blind regime" and "Related work on the same task".

**Why the unseen-gene predictor fails** (GAPS.md "The unseen-gene gap is a combiner
problem", fig22, `./submit.sh repr`). Not the predictor class: the same weighted average
of measured signatures reaches **44.4% / 48.3%** top-1 (K562 / RPE1) given the true
similarity and a ridge combiner, against GO-kNN's 2.9% / 1.5%. Not prior knowledge: of 20
prior-only predictors, none beats GO-kNN on both top-1 and median rank on both lines. Not
scoring geometry: CSLS, the standard hubness correction, does not help. The shortfall is
estimating the inner products the combiner reads -- handed the true ones, GO's own
neighbourhood reaches 85% of the oracle on both lines. The one realistic input that moves
it is a measurement: a gene unseen in this line but measured in the other is identifiable
**1.7–2.9×** better than GO manages, on the genes the other line covers.

**RL with the other line's screen as a simulator** (GAPS.md "Sim-to-real", `./submit.sh sim`).
Choosing 100 of 437 genes to measure in one line, with no signature of a candidate read there,
RL trained inside the other line's screen picks better genes than GO facility location *among
the genes that screen covers*: K562 +0.0145 MRR (13/15 splits, p 0.005), RPE1 +0.0070 (12/15,
p 0.01), read only on 15 held-out splits that chose nothing. It also out-optimises greedy on
the identical simulator reward (K562 15/15 splits, p 5e-6; RPE1 11/15, p 0.006). Over all 437
candidates it ties the best prior floor (p 0.98 / 0.36): genes the source line never screened
cannot be simulated, and losing them costs about what RL gains. Two apparent floor wins found
on development splits reversed on fresh ones; GAPS.md reports both. The section is generated
from the reports by `python3 scripts/sim_splice.py`.
