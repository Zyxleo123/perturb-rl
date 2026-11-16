#!/bin/bash
# Shared preamble for all 10703_pj jobs.
#
# Sourced by every .sbatch in jobs/.  It:
#   1. defines ALL configuration defaults (single source of truth),
#   2. consumes the job's positional arguments: any argument that is an existing
#      file is sourced (the submitter's timestamped env file); any argument of
#      the form VAR=value is exported.
#      -> we never use `sbatch --export`, which triggers SLURM_GET_USER_ENV=1 and
#         gets jobs requeued+held on this cluster.
#   3. prints the provenance banner (hostname / GPUs / resource request) into the
#      .out, because sacct is dead here and there is no accounting record to
#      fall back on.
#
# NOTE: `set -euo pipefail` lives in the job body, NOT here.  If sourcing this
# file ever fails, the job must die loudly rather than exit 0 and release its
# afterok dependents.

# ---------------------------------------------------------------- paths -------
export PROJ_ROOT="${PROJ_ROOT:-/zfsauton2/home/yixiz/10703_pj}"
export WORK_ROOT="${WORK_ROOT:-/zfsauton/scratch/yixiz/10703_pj}"

export CODE_ROOT="${CODE_ROOT:-$WORK_ROOT/code}"       # cloned 3rd-party repos (untrusted)
export DATA_ROOT="${DATA_ROOT:-$WORK_ROOT/data}"       # downloaded datasets    (untrusted)
export WEIGHT_ROOT="${WEIGHT_ROOT:-$WORK_ROOT/weights}"
export RESULTS_ROOT="${RESULTS_ROOT:-$WORK_ROOT/results}"
export FIG_ROOT="${FIG_ROOT:-$WORK_ROOT/figures}"
export LOG_ROOT="${LOG_ROOT:-$WORK_ROOT/logs}"
export SCRIPT_ROOT="${SCRIPT_ROOT:-$PROJ_ROOT/scripts}"  # OUR code, never mixed with downloads

# ---------------------------------------------------------- python envs -------
# Analysis/plotting env: reuse the existing scanpy+anndata env (no torch needed).
export PY_ANALYSIS="${PY_ANALYSIS:-/zfsauton/scratch/yixiz/envs/cellflow/bin/python}"
# Torch env for the forward-oracle (GEARS) work; built by jobs/20_build_env.sbatch.
export PERTRL_ENV="${PERTRL_ENV:-/zfsauton/scratch/yixiz/envs/pertrl}"
export PY_TORCH="${PY_TORCH:-$PERTRL_ENV/bin/python}"

# ------------------------------------------------------------- datasets -------
# GEARS perturb-seq datasets, Harvard Dataverse file ids (from gears/pertdata.py).
export GEARS_DATASETS="${GEARS_DATASETS:-norman adamson dixit replogle_k562_essential replogle_rpe1_essential}"
export GEARS_AUX="${GEARS_AUX:-gene2go_all:6153417 essential_all_data_pert_genes:6934320 gene2go_coexpress:6934319}"

# PDGrapher (Nature BME 2025) processed data, Zenodo.
export PDG_ZENODO_GENETIC="${PDG_ZENODO_GENETIC:-15375990}"
export PDG_ZENODO_CHEMICAL="${PDG_ZENODO_CHEMICAL:-15390483}"
# Chemical tarball is 37 GB and this project is genetic-perturbation only.
export PDG_FETCH_CHEMICAL="${PDG_FETCH_CHEMICAL:-0}"

# ------------------------------------------------------------- analysis -------
export PROFILE_DATASETS="${PROFILE_DATASETS:-norman adamson dixit replogle_k562_essential replogle_rpe1_essential}"
export N_TOP_DE="${N_TOP_DE:-20}"       # DE genes per perturbation, matches GEARS' top-20 metric
export MIN_CELLS_PER_PERT="${MIN_CELLS_PER_PERT:-20}"
export RANDOM_SEED="${RANDOM_SEED:-0}"

# ----------------------------------------------- positional arg handling ------
# Every job calls:  pj_consume_args "$@"
pj_consume_args() {
  local a
  for a in "$@"; do
    if [[ -f "$a" ]]; then
      echo "[preamble] sourcing env file: $a"
      # shellcheck disable=SC1090
      source "$a"
    elif [[ "$a" == *=* ]]; then
      echo "[preamble] exporting override: $a"
      export "${a?}"
    else
      echo "[preamble] WARNING: ignoring unrecognized argument: $a" >&2
    fi
  done
  mkdir -p "$CODE_ROOT" "$DATA_ROOT" "$WEIGHT_ROOT" "$RESULTS_ROOT" "$FIG_ROOT" "$LOG_ROOT"
}

# ------------------------------------------------------------- provenance -----
pj_banner() {
  echo "=================================================================="
  echo "[prov] job        : ${SLURM_JOB_NAME:-<none>} / id ${SLURM_JOB_ID:-<none>}"
  echo "[prov] host       : $(hostname)"
  echo "[prov] date       : $(date -Is)"
  echo "[prov] submit_dir : ${SLURM_SUBMIT_DIR:-<none>}"
  echo "[prov] partition  : ${SLURM_JOB_PARTITION:-<none>}"
  echo "[prov] cpus       : ${SLURM_CPUS_ON_NODE:-?} (req ${SLURM_CPUS_PER_TASK:-?})"
  echo "[prov] mem req    : ${SLURM_MEM_PER_NODE:-?} MB"
  echo "[prov] nodelist   : ${SLURM_JOB_NODELIST:-<none>}"
  echo "[prov] gpus req   : ${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-none}}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[prov] nvidia-smi -L:"; nvidia-smi -L 2>&1 | sed 's/^/           /'
  else
    echo "[prov] nvidia-smi : not present (CPU node)"
  fi
  echo "[prov] CODE_ROOT  : $CODE_ROOT"
  echo "[prov] DATA_ROOT  : $DATA_ROOT"
  echo "[prov] RESULTS    : $RESULTS_ROOT"
  echo "=================================================================="
}

# --------------------------------------------------------------- helpers ------
# Resumable download: skip if a non-empty marker-verified file already exists.
pj_fetch() {   # pj_fetch <url> <dest>
  local url="$1" dest="$2"
  if [[ -s "$dest" && -f "$dest.done" ]]; then
    echo "[fetch] SKIP (already complete): $dest"
    return 0
  fi
  echo "[fetch] -> $dest"
  mkdir -p "$(dirname "$dest")"
  # -c resumes a partial file; --tries for flaky NFS/HTTP
  wget -c -q --show-progress --progress=dot:giga --tries=5 --timeout=60 \
       -O "$dest" "$url"
  touch "$dest.done"
}

# Gate helper: a failed upstream gate must print why and exit 0, so that
# afterok dependents are released rather than stranded in
# DependencyNeverSatisfied with no explanation.
pj_gate() {   # pj_gate <path> <human description>
  if [[ ! -e "$1" ]]; then
    echo "[gate] MISSING: $1"
    echo "[gate] $2"
    echo "[gate] This is a failed decision gate, not a crash. Exiting 0 so"
    echo "[gate] dependent jobs are released and can report the same gate."
    exit 0
  fi
  echo "[gate] ok: $1"
}
