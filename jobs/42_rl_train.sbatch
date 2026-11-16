#!/usr/bin/env bash
#SBATCH --job-name=pj_rltrain
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out

# One RL run per array task: (dataset, split, method, seed) from the grid in
# env/common.sh (PJ_RLA_*). Methods: pg_tab, pg_tab_boot (REINFORCE on per-gene
# logits), ppo_peek, ppo_blind (GRPO-style PPO on a feature policy), critic
# (fitted one-step value + shrinkage greedy). CPU only: the policy is a 2x32 MLP
# and the reward is Gram-matrix algebra, so a GPU would idle.
# Resumable: a task whose result file exists is skipped (PJ_RLA_FORCE=1 redoes it).
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x4, 24G, 4h, array task ${SLURM_ARRAY_TASK_ID:-<none>}, smoke=${PJ_RLA_SMOKE:-0}"

CK="$PJ_ARTIFACTS/rl_acquire/cache_check.json"
[ -s "$CK" ] || pj_gate_fail "missing $CK (did jobs/40_rl_cache.sbatch run?)"
[ -x "$PJ_PY_TORCH" ] || pj_gate_fail "RL python not executable: $PJ_PY_TORCH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"$PJ_PY_TORCH" -I "$PJ_HOME/scripts/rl_acquire.py" train \
  --cache-dir "$PJ_ARTIFACTS/rl_acquire" --out-dir "$PJ_ARTIFACTS" \
  --task-index "${SLURM_ARRAY_TASK_ID:?not an array task}" --threads "${SLURM_CPUS_PER_TASK:-4}"

echo "=== DONE ==="
