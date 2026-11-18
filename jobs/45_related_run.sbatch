#!/usr/bin/env bash
#SBATCH --job-name=pj_relrun
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out

# Published acquisition methods on the RL acquisition task (scripts/related_work.py):
# IterPert and its benchmark through IterPert's own code (blind), data valuation and
# exemplar selection (peek). Grid = PJ_RLA_DATASETS x PJ_RLA_SPLITS x PJ_REL_METHODS x
# PJ_REL_SEEDS, the RL grid's own datasets and splits. Each array task runs
# PJ_REL_PER_TASK consecutive entries IN PARALLEL (array tasks count against the
# per-user QOS submit limit; see jobs/42_rl_train.sbatch). A deterministic method's
# later seeds are no-ops. Resumable: a run whose result file exists is skipped.
# CPU only: the largest object is a 1,543 x 1,543 kernel.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
CPUS="${SLURM_CPUS_PER_TASK:-16}"
PER="${PJ_REL_PER_TASK}"
NTOT=$(( $(wc -w <<<"$PJ_RLA_DATASETS") * $(wc -w <<<"$PJ_RLA_SPLITS") \
       * $(wc -w <<<"$PJ_REL_METHODS") * $(wc -w <<<"$PJ_REL_SEEDS") ))
THREADS=$(( CPUS / PER > 0 ? CPUS / PER : 1 ))
TASK="${SLURM_ARRAY_TASK_ID:?not an array task}"
# 4h: the slowest entry is TMC Shapley, 19,200 reward evaluations at prefix sizes up to
# 437 genes; the entries of one task run in parallel, so wall time is the slowest one.
echo "request  : cpu x${CPUS}, 48G, 4h, array task ${TASK}: grid entries $((TASK * PER))..$((TASK * PER + PER - 1)) of ${NTOT}, ${THREADS} threads each, smoke=${PJ_REL_SMOKE:-0}"

CK="$PJ_ARTIFACTS/rl_acquire/cache_check.json"
[ -s "$CK" ] || pj_gate_fail "missing $CK (did jobs/40_rl_cache.sbatch run?)"
[ -x "$PJ_PY_TORCH" ] || pj_gate_fail "RL python not executable: $PJ_PY_TORCH"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"

pids=()
for j in $(seq 0 $((PER - 1))); do
  idx=$((TASK * PER + j))
  [ "$idx" -lt "$NTOT" ] || continue
  ( "$PJ_PY_TORCH" -I "$PJ_HOME/scripts/related_work.py" run \
      --cache-dir "$PJ_ARTIFACTS/rl_acquire" --out-dir "$PJ_ARTIFACTS" --code-dir "$PJ_CODE" \
      --task-index "$idx" --threads "$THREADS" 2>&1 | sed -u "s/^/[g${idx}] /" ) &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "=== DONE (rc=$rc) ==="
exit "$rc"
