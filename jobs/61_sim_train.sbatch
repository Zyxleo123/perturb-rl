#!/usr/bin/env bash
#SBATCH --job-name=pj_simtrain
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out

# Sim-to-real RL runs (scripts/rl_sim.py): the peek-regime RL methods trained inside
# the OTHER cell line's screen on the deployment candidates, deployed in this line.
# Grid = PJ_RLA_DATASETS x PJ_RLA_SPLITS x PJ_SIM_METHODS x PJ_RLA_SEEDS. Each array
# task runs PJ_SIM_PER_TASK grid entries in parallel (array tasks count against the
# per-user QOS submit limit of 50). Wall time is the slowest run in the task: PPO peek
# took ~0.5-1 h per run at this scale in rl_acquire (2026-11-16), so 8 h is generous.
# Resumable: a run whose result file exists is skipped (PJ_SIM_FORCE=1 redoes it).
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
CPUS="${SLURM_CPUS_PER_TASK:-24}"
PER="${PJ_SIM_PER_TASK}"
NTOT=$(( $(wc -w <<<"$PJ_RLA_DATASETS") * $(wc -w <<<"$PJ_RLA_SPLITS") \
       * $(wc -w <<<"$PJ_SIM_METHODS") * $(wc -w <<<"$PJ_RLA_SEEDS") ))
THREADS=$(( CPUS / PER > 0 ? CPUS / PER : 1 ))
TASK="${SLURM_ARRAY_TASK_ID:?not an array task}"
echo "request  : cpu x${CPUS}, 64G, 8h, array task ${TASK}: grid entries $((TASK * PER))..$((TASK * PER + PER - 1)) of ${NTOT}, ${THREADS} threads each, smoke=${PJ_RLA_SMOKE:-0}"

CK="$PJ_ARTIFACTS/rl_acquire/cache_check.json"
[ -s "$CK" ] || pj_gate_fail "missing $CK (did jobs/40_rl_cache.sbatch run?)"
[ -x "$PJ_PY_TORCH" ] || pj_gate_fail "RL python not executable: $PJ_PY_TORCH"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"

pids=()
for j in $(seq 0 $((PER - 1))); do
  idx=$((TASK * PER + j))
  [ "$idx" -lt "$NTOT" ] || continue
  ( "$PJ_PY_TORCH" -I "$PJ_HOME/scripts/rl_sim.py" train \
      --cache-dir "$PJ_ARTIFACTS/rl_acquire" --out-dir "$PJ_ARTIFACTS" \
      --task-index "$idx" --threads "$THREADS" 2>&1 | sed -u "s/^/[g${idx}] /" ) &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "=== DONE (rc=$rc) ==="
exit "$rc"
