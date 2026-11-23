#!/usr/bin/env bash
#SBATCH --job-name=pj_recon
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Why does a retrieval row fail to reproduce a saved one? (scripts/repr_recon_check.py)
# Builds the GO-kNN row by repr_select's statement sequence AND by repr_gram's in one
# process and compares every intermediate, then both against the artifact on disk --
# so "the two code paths differ" and "the artifact came from another environment" are
# distinguishable instead of guessed at. It found CCT3: one tie-boundary flip, no bug.
# Read-only, CPU, one dataset. Keep it: the next reconciliation failure wants it too.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x4, 16G, 30m, dataset=${PJ_RECON_DS:-replogle_k562_essential}"

GD="$PJ_DATA/gears"
DS="${PJ_RECON_DS:-replogle_k562_essential}"
[ -s "$PJ_ARTIFACTS/rl_acquire/cache_check.json" ] || pj_gate_fail "no RL cache (./submit.sh rlcache)"
[ -s "$PJ_ARTIFACTS/repr_select/result_$DS.npz" ] || pj_gate_fail "no repr_select result for $DS"
[ -s "$GD/$DS/perturb_processed.h5ad" ] || pj_gate_fail "missing $GD/$DS/perturb_processed.h5ad"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" \
       MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/repr_recon_check.py" \
  --art-dir "$PJ_ARTIFACTS" --gears-dir "$GD" --dataset "$DS"
echo "=== DONE ==="
