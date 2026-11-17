#!/usr/bin/env bash
#SBATCH --job-name=pj_rltrain
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out

# RL runs from the grid in env/common.sh (PJ_RLA_*): datasets x splits x methods x
# seeds. Methods: pg_tab, pg_tab_boot (REINFORCE on per-gene logits), ppo_peek,
# ppo_blind (GRPO-style PPO on a feature policy), critic (fitted one-step value +
# shrinkage greedy). CPU only: the policy is a 2x32 MLP and the reward is
# Gram-matrix algebra, so a GPU would idle.
#
# Each array task runs PJ_RLA_PER_TASK consecutive grid entries IN PARALLEL, one
# process each with an equal share of the CPUs. With the default of 6, one task
# is two (dataset, split, method) cells with their 3 seeds each. This exists
# because one array task per run exceeded the per-user QOS submit limit
# (QOSMaxSubmitJobPerUserLimit, 2026-11-16): array tasks count as jobs, and
# qos_cpu allows 50 submitted jobs and 352 CPUs per user (scontrol show assoc_mgr).
# Resumable: a run whose result file exists is skipped (PJ_RLA_FORCE=1 redoes it).
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
CPUS="${SLURM_CPUS_PER_TASK:-24}"
PER="${PJ_RLA_PER_TASK}"
NTOT=$(( $(wc -w <<<"$PJ_RLA_DATASETS") * $(wc -w <<<"$PJ_RLA_SPLITS") \
       * $(wc -w <<<"$PJ_RLA_METHODS") * $(wc -w <<<"$PJ_RLA_SEEDS") ))
THREADS=$(( CPUS / PER > 0 ? CPUS / PER : 1 ))
TASK="${SLURM_ARRAY_TASK_ID:?not an array task}"
# 8h: a run on deployment-scale sub-problems (ppo_blind_match / _xline, 437 candidates
# and budget 100 instead of ~190/43) costs ~5x a shrunk-scale one -- measured 2026-11-17,
# smoke job 112815: 2.4-5.0s per iteration at ppo_univ x ppo_group = 6 rollouts, so ~50s
# at the real 64, x120 iterations ~= 2h. The 6 runs in a task are parallel, so the task's
# wall time is the slowest single run, not their sum.
echo "request  : cpu x${CPUS}, 64G, 8h, array task ${TASK}: grid entries $((TASK * PER))..$((TASK * PER + PER - 1)) of ${NTOT}, ${THREADS} threads each, smoke=${PJ_RLA_SMOKE:-0}"

CK="$PJ_ARTIFACTS/rl_acquire/cache_check.json"
[ -s "$CK" ] || pj_gate_fail "missing $CK (did jobs/40_rl_cache.sbatch run?)"
[ -x "$PJ_PY_TORCH" ] || pj_gate_fail "RL python not executable: $PJ_PY_TORCH"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"

pids=()
for j in $(seq 0 $((PER - 1))); do
  idx=$((TASK * PER + j))
  [ "$idx" -lt "$NTOT" ] || continue
  # each run's lines prefixed with its grid index; pipefail makes `wait` see python's status
  ( "$PJ_PY_TORCH" -I "$PJ_HOME/scripts/rl_acquire.py" train \
      --cache-dir "$PJ_ARTIFACTS/rl_acquire" --out-dir "$PJ_ARTIFACTS" \
      --task-index "$idx" --threads "$THREADS" 2>&1 | sed -u "s/^/[g${idx}] /" ) &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "=== DONE (rc=$rc) ==="
exit "$rc"
