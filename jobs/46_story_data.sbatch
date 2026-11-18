#!/usr/bin/env bash
#SBATCH --job-name=pj_storydata
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Case-study DATA for the gap / improvement story figures (fig18-fig21): named genes,
# their ranks, neighbourhoods and greedy's per-step gains, computed once from the RL
# cache + saved chosen sets (+ one rl_headroom.build per dataset for leak-free
# half-A signatures). Writes artifacts/story/; jobs/47_story_figs.sbatch draws from it.
# Resumable per dataset: a finished dataset is skipped unless PJ_STORY_FORCE=1.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x8, 48G, 2h, force=${PJ_STORY_FORCE:-0}"

GD="$PJ_DATA/gears"
[ -s "$PJ_ARTIFACTS/rl_acquire/cache_check.json" ] || pj_gate_fail "no RL cache (run ./submit.sh rlcache first)"
[ -s "$PJ_ARTIFACTS/rl_acquire/records.jsonl" ] || pj_gate_fail "no re-scored records (run ./submit.sh rlacq_report first)"
[ -s "$GD/gene2go_all.pkl" ] || pj_gate_fail "missing $GD/gene2go_all.pkl"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
       MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_$$"

# shellcheck disable=SC2086
"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/story_cases.py" compute \
  --art-dir "$PJ_ARTIFACTS" --gears-dir "$GD" --datasets $PJ_RLA_DATASETS \
  ${PJ_STORY_FORCE:+--force}

ls -la "$PJ_ARTIFACTS/story" | sed 's/^/  /'
echo "=== DONE ==="
