#!/usr/bin/env bash
# Shared preamble for all 10703_pj jobs.
#
# Sourced by every .sbatch body. NOT self-executing.
#
# Contract (see CLAUDE.md):
#   - Jobs are configured by environment variable; defaults live HERE, once.
#   - The submitter writes ONE timestamped env file per submission and passes its
#     path as a POSITIONAL ARGUMENT to the batch script. We never use
#     `sbatch --export` (it triggers SLURM_GET_USER_ENV and the job gets held).
#   - Any argument that is an existing file is sourced; any VAR=value argument is
#     exported. So both `job.sbatch /path/to/env.sh` and `job.sbatch K=16` work.
#   - This file must be self-sufficient under `env -i`.

# ---------------------------------------------------------------- roots
: "${PJ_HOME:=/zfsauton2/home/yixiz/10703_pj}"     # code we write (git-able, small)
: "${PJ_SCRATCH:=/zfsauton/scratch/yixiz/10703_pj}" # everything large

: "${PJ_CODE:=$PJ_SCRATCH/code}"           # cloned third-party baselines (UNTRUSTED)
: "${PJ_DATA:=$PJ_SCRATCH/data}"           # downloaded datasets (UNTRUSTED)
: "${PJ_WEIGHTS:=$PJ_SCRATCH/weights}"     # downloaded checkpoints (UNTRUSTED)
: "${PJ_ARTIFACTS:=$PJ_SCRATCH/artifacts}" # our computed results: jsonl/npz/json
: "${PJ_FIGURES:=$PJ_SCRATCH/figures}"     # our plots, redrawable from artifacts
: "${PJ_LOGS:=$PJ_SCRATCH/logs}"           # slurm .out files
: "${PJ_ENVFILES:=$PJ_SCRATCH/envfiles}"   # one env file per submission

# ---------------------------------------------------------------- python
# Analysis/plotting env: scanpy 1.12 / anndata 0.13 / numpy 2.4 / matplotlib 3.11.
# Reused rather than rebuilt; it has no torch, which is fine for profiling+plots.
: "${PJ_PY_ANALYSIS:=/zfsauton/scratch/yixiz/envs/cellflow/bin/python}"
# Torch env for running the GEARS oracle. Built by jobs/13_build_torch_env.sbatch.
: "${PJ_PY_TORCH:=/zfsauton/scratch/yixiz/envs/pertrl/bin/python}"
: "${PJ_UV:=/zfsauton2/home/yixiz/.local/bin/uv}"

# ---------------------------------------------------------------- download manifest
# GEARS processed perturb-seq datasets (Harvard Dataverse file ids, read out of
# snap-stanford/GEARS@master gears/pertdata.py). Each is a zip that unpacks to
# <name>/perturb_processed.h5ad.
: "${GEARS_DATASETS:=norman adamson dixit replogle_k562_essential replogle_rpe1_essential}"
: "${DV_norman:=6154020}"
: "${DV_adamson:=6154417}"
: "${DV_dixit:=6154416}"
: "${DV_replogle_k562_essential:=7458695}"
: "${DV_replogle_rpe1_essential:=7458694}"

# GEARS auxiliary files (GO graph + perturbable gene universe)
: "${DV_gene2go_all:=6153417}"              # gene2go_all.pkl
: "${DV_essential_pert_genes:=6934320}"     # essential_all_data_pert_genes.pkl
: "${DV_go_essential_all:=6934319}"         # go_essential_all.csv (GO similarity net)

# GEARS OFFICIAL pretrained forward oracle (from demo/tutorial_inference_Norman.ipynb)
: "${DV_gears_ckpt:=10457098}"              # model.zip -> model_ckpt/{model.pt,config.pkl}
: "${DV_gears_dataloader:=6979957}"         # norman_umi_go.tar.gz (matching dataloader)

# Third-party HF mirror of a Norman-trained GEARS (UNVERIFIED, convenience only)
: "${HF_GEARS_MIRROR:=matthewshu/gears-norman}"

# PDGrapher (Nature BME 2025) processed data, Zenodo
: "${ZENODO_PDG_GENETIC:=15375990}"         # splits.tar.gz + torch_data_genetic.tar.gz (21.7 GB)
: "${ZENODO_PDG_CHEMICAL:=15390483}"        # torch_data_chemical.tar.gz (37.3 GB)
: "${FETCH_PDG_CHEMICAL:=0}"                # off by default: our project is genetic

# Baseline code repos: "name|url|ref"
: "${CODE_REPOS:=\
GEARS|https://github.com/snap-stanford/GEARS|master \
PDGrapher|https://github.com/mims-harvard/PDGrapher|main \
LabCompass|https://github.com/theislab/LabCompass|main \
cellflow|https://github.com/theislab/cellflow|main \
state|https://github.com/ArcInstitute/state|main}"

# ---------------------------------------------------------------- RL acquisition (jobs 40-43)
# Task grid for jobs/42_rl_train.sbatch is datasets x splits x methods x seeds.
# Split 4 is GAPS part 4's split, number for number; 104 and 204 are there so a
# result cannot be one split's luck. The submitter writes only the overrides you
# set into the env file, so these stay the single source of defaults.
: "${PJ_RLA_DATASETS:=replogle_k562_essential replogle_rpe1_essential}"
: "${PJ_RLA_SPLITS:=4 104 204}"
# The last four attack the blind regime, the one the first round left open (see
# scripts/rl_acquire.py BLIND_VARIANTS): deployment-scale training sub-problems,
# richer gene priors than GO Jaccard, training on the other cell line, and both.
: "${PJ_RLA_METHODS:=pg_tab pg_tab_boot ppo_peek ppo_blind critic ppo_blind_match ppo_blind_rich ppo_blind_xline ppo_blind_xline_rich}"
: "${PJ_RLA_SEEDS:=0 1 2}"
: "${PJ_RLA_PER_TASK:=6}"  # grid entries run in parallel in one array task (qos_cpu: 50 jobs, 352 CPUs per user)
: "${PJ_RLA_MAXPAR:=30}"   # concurrent array tasks
: "${PJ_RLA_SMOKE:=}"      # 1 = tiny training settings, results to artifacts/rl_acquire_smoke/
: "${PJ_RLA_FORCE:=}"      # 1 = redo tasks whose result file already exists
export PJ_RLA_DATASETS PJ_RLA_SPLITS PJ_RLA_METHODS PJ_RLA_SEEDS PJ_RLA_PER_TASK PJ_RLA_MAXPAR PJ_RLA_SMOKE PJ_RLA_FORCE

# ---------------------------------------------------------------- sim-to-real acquisition (jobs 60-62)
# scripts/rl_sim.py: the peek RL methods trained inside the OTHER cell line's screen on
# the deployment candidates, deployed in this line. Datasets, splits and seeds are the
# RL grid's own (PJ_RLA_*), so every run pairs with the existing floors.
# Two simulators (scripts/rl_sim.py POPS): `all` rewards on every source gene outside the
# candidates and T's report genes; `_tpop` methods reward only on the source rows of T's
# own measured atlas, i.e. the deployment's gene population; `_tpred` methods additionally
# use T's GO Jaccard inside the simulator, so only the signatures come from the source;
# `_timp` is T's own peek world with each candidate's hidden signature imputed from the
# source (re-centred) or GO-kNN -- needs ./submit.sh sim_readout first.
: "${PJ_SIM_METHODS:=sim_pg_tab sim_pg_tab_boot sim_critic sim_ppo_peek sim_pg_tab_tpop sim_pg_tab_boot_tpop sim_critic_tpop sim_ppo_peek_tpop sim_pg_tab_tpred sim_pg_tab_boot_tpred sim_critic_tpred sim_ppo_peek_tpred sim_pg_tab_timp sim_pg_tab_boot_timp sim_critic_timp sim_ppo_peek_timp}"
: "${PJ_SIM_POPS:=all tpop tpred timp}"   # which simulators the baselines array covers
: "${PJ_SIM_PER_TASK:=12}"     # grid entries run in parallel in one array task
: "${PJ_SIM_MAXPAR:=6}"
: "${PJ_SIM_FORCE:=}"          # 1 = redo runs whose result file exists
export PJ_SIM_METHODS PJ_SIM_POPS PJ_SIM_PER_TASK PJ_SIM_MAXPAR PJ_SIM_FORCE

# ---------------------------------------------------------------- related work (jobs 14, 44-45)
# Published acquisition methods re-run on the RL task's exact splits, predictor and
# report genes (scripts/related_work.py). The datasets and splits are the RL grid's
# own (PJ_RLA_DATASETS / PJ_RLA_SPLITS), so every number pairs with an RL run.
# Code is cloned, never pip-installed, into $PJ_CODE (UNTRUSTED, like every baseline).
: "${RELATED_REPOS:=\
iterative-perturb-seq|https://github.com/Genentech/iterative-perturb-seq|master \
bmdal_reg|https://github.com/dholzmueller/bmdal_reg|main}"
# IterPert's released prior-knowledge kernels for Replogle K562 essential (Harvard
# Dataverse, the URL hard-coded in iterpert/iterpert.py), MD5 from the Dataverse API.
: "${DV_iterpert_kernels:=7377301}"
: "${MD5_iterpert_kernels:=25cb5ef91e1b53c33414b4eb42ea6213}"
: "${PJ_REL_METHODS:=iterpert iterpert_go iterpert_cold kc_go kc_iterpert_priors fl_iterpert_priors al_coreset al_bald al_batchbald al_badge al_lcmd al_acsfw al_kmeans al_typiclust dv_shapley dv_banzhaf dv_datamodels rep_facility_sig rep_coreset_sig}"
: "${PJ_REL_SEEDS:=0 1 2}"     # deterministic methods run seed 0 only (see related_work.METHODS)
: "${PJ_REL_PER_TASK:=8}"      # grid entries run in parallel in one array task
: "${PJ_REL_MAXPAR:=30}"
: "${PJ_REL_NEVAL:=}"          # reward evaluations per data-valuation run; empty = REINFORCE's 600 x 32
: "${PJ_REL_SMOKE:=}"          # 1 = 300 reward evaluations, results to artifacts/rl_acquire/related_smoke/
: "${PJ_REL_FORCE:=}"          # 1 = redo runs whose result file exists
export RELATED_REPOS DV_iterpert_kernels MD5_iterpert_kernels PJ_REL_METHODS PJ_REL_SEEDS \
       PJ_REL_PER_TASK PJ_REL_MAXPAR PJ_REL_NEVAL PJ_REL_SMOKE PJ_REL_FORCE

# ---------------------------------------------------------------- story case studies (jobs 46-47)
# scripts/story_cases.py: named-gene case studies for the gap / improvement story,
# on the RL grid's datasets (PJ_RLA_DATASETS) and GAPS part 4's split.
: "${PJ_STORY_FORCE:=}"        # 1 = recompute datasets whose artifacts/story npz exists
export PJ_STORY_FORCE

# ---------------------------------------------------------------- helpers
pj_banner() {
  echo "=================================================================="
  echo "job      : ${SLURM_JOB_NAME:-<none>} (${SLURM_JOB_ID:-<none>})"
  echo "host     : $(hostname)"
  echo "date     : $(date -Is)"
  echo "partition: ${SLURM_JOB_PARTITION:-<none>}"
  echo "cpus     : ${SLURM_CPUS_ON_NODE:-?}   mem: ${SLURM_MEM_PER_NODE:-?}M"
  echo "nodelist : ${SLURM_JOB_NODELIST:-<none>}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpus     : $(nvidia-smi -L 2>&1 | tr '\n' ' ' | cut -c1-300)"
  else
    echo "gpus     : (no nvidia-smi on PATH)"
  fi
  echo "submitdir: ${SLURM_SUBMIT_DIR:-<none>}"
  echo "=================================================================="
}

# Consume positional args: source env files, export VAR=value pairs.
pj_consume_args() {
  for a in "$@"; do
    if [ -f "$a" ]; then
      echo "[preamble] sourcing env file: $a"
      # shellcheck disable=SC1090
      . "$a"
    elif [[ "$a" == *=* ]]; then
      echo "[preamble] exporting: $a"
      export "${a?}"
    else
      echo "[preamble] WARNING: ignoring unrecognized argument: $a"
    fi
  done
}

pj_mkdirs() {
  mkdir -p "$PJ_CODE" "$PJ_DATA" "$PJ_WEIGHTS" "$PJ_ARTIFACTS" "$PJ_FIGURES" \
           "$PJ_LOGS" "$PJ_ENVFILES"
}

# Download with resume + retries. $1=url $2=outfile
pj_fetch() {
  local url="$1" out="$2"
  if [ -s "$out" ] && [ -f "${out}.done" ]; then
    echo "[fetch] already complete, skipping: $out"
    return 0
  fi
  echo "[fetch] $url -> $out"
  curl -L --fail --retry 8 --retry-delay 15 --retry-all-errors \
       --continue-at - --connect-timeout 30 \
       -o "$out" "$url" || { echo "[fetch] FAILED: $url"; return 1; }
  touch "${out}.done"
  ls -lh "$out"
}

# `unzip` is NOT installed on the compute nodes (measured 2026-11-13, lov6), so
# extraction goes through python's zipfile. $1=archive $2=dest-dir
pj_unzip() {
  local zip="$1" dest="$2"
  echo "[unzip] $zip -> $dest (python zipfile)"
  "$PJ_PY_ANALYSIS" -I -c '
import sys, zipfile
src, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src) as z:
    bad = z.testzip()
    if bad is not None:
        print("[unzip] CORRUPT member:", bad); sys.exit(1)
    names = z.namelist()
    print(f"[unzip] {len(names)} members, first: {names[:3]}")
    z.extractall(dest)
print("[unzip] ok")
' "$zip" "$dest"
}

# A gate that failed must still exit 0, or every afterok dependent is stranded
# in DependencyNeverSatisfied with no explanation.
pj_gate_fail() {
  echo "###################################################################"
  echo "GATE NOT MET: $*"
  echo "Exiting 0 on purpose so downstream jobs run and report the same gap."
  echo "###################################################################"
  exit 0
}

export PJ_HOME PJ_SCRATCH PJ_CODE PJ_DATA PJ_WEIGHTS PJ_ARTIFACTS PJ_FIGURES \
       PJ_LOGS PJ_ENVFILES PJ_PY_ANALYSIS PJ_PY_TORCH PJ_UV
