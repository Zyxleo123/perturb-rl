#!/usr/bin/env bash
#SBATCH --job-name=pj_rlbase
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out

# Baselines for the RL acquisition experiments, one array task per (dataset, split):
# random x100, GO coverage, GO facility location, greedy on GAPS part 4's exact
# protocol and on every aux gene with every candidate scored, and the cheating
# greedy on the report genes. Also checks the fast scorer against rl_headroom's own.
# Resumable: a (dataset, split) whose file exists is skipped.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x4, 16G, 3h, array task ${SLURM_ARRAY_TASK_ID:-<none>}"

CK="$PJ_ARTIFACTS/rl_acquire/cache_check.json"
[ -s "$CK" ] || pj_gate_fail "missing $CK (did jobs/40_rl_cache.sbatch run?)"
[ -x "$PJ_PY_TORCH" ] || pj_gate_fail "RL python not executable: $PJ_PY_TORCH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# baselines do not depend on the smoke settings: always the real output directory
PJ_RLA_SMOKE= "$PJ_PY_TORCH" -I "$PJ_HOME/scripts/rl_acquire.py" baselines \
  --cache-dir "$PJ_ARTIFACTS/rl_acquire" --out-dir "$PJ_ARTIFACTS" \
  --task-index "${SLURM_ARRAY_TASK_ID:?not an array task}"

ls -la "$PJ_ARTIFACTS/rl_acquire/baselines" | sed 's/^/  /'
echo "=== DONE ==="
