#!/usr/bin/env bash
#SBATCH --job-name=pj_simbase
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out

# Baselines for sim-to-real acquisition (scripts/rl_sim.py), one array task per
# (target dataset, split): reconciliation against rl_acquire's baselines (T's random
# median and GO facility set, exactly), GO facility over covered candidates, facility
# on the source line's signatures and GO, greedy on the source-line reward (3 seeds),
# and the transfer probe (does the source rank sets the way the target does?).
# Exits 1 if a reconciliation point fails: that is a bug, not a gate.
# Resumable: a (dataset, split) whose file ends in "done" is skipped (PJ_SIM_FORCE=1 redoes).
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x4, 16G, 5h, array task ${SLURM_ARRAY_TASK_ID:-<none>}"

CK="$PJ_ARTIFACTS/rl_acquire/cache_check.json"
[ -s "$CK" ] || pj_gate_fail "missing $CK (did jobs/40_rl_cache.sbatch run?)"
[ -d "$PJ_ARTIFACTS/rl_acquire/baselines" ] || pj_gate_fail "no rl_acquire baselines: the reconciliation point is missing"
[ -x "$PJ_PY_TORCH" ] || pj_gate_fail "RL python not executable: $PJ_PY_TORCH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# baselines do not depend on the smoke settings: always the real output directory
PJ_RLA_SMOKE= "$PJ_PY_TORCH" -I "$PJ_HOME/scripts/rl_sim.py" baselines \
  --cache-dir "$PJ_ARTIFACTS/rl_acquire" --out-dir "$PJ_ARTIFACTS" \
  --task-index "${SLURM_ARRAY_TASK_ID:?not an array task}"
echo "=== DONE ==="
