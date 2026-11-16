#!/usr/bin/env bash
#SBATCH --job-name=pj_rlreport
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Table + figures for the RL acquisition runs, from saved artifacts only
# (artifacts/rl_acquire/{baselines,runs}). Cheap; rerun any time with
# ./submit.sh rlacq_report. Runs afterany, so it reports partial results and
# lists failed tasks rather than waiting for a perfect array.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x2, 8G, 30m, smoke=${PJ_RLA_SMOKE:-0}"

[ -d "$PJ_ARTIFACTS/rl_acquire/baselines" ] || pj_gate_fail "no baselines in $PJ_ARTIFACTS/rl_acquire (upstream never ran)"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/rl_report.py" \
  --art-dir "$PJ_ARTIFACTS" --fig-dir "$PJ_FIGURES" ${PJ_RLA_SMOKE:+--smoke}

echo "=== DONE ==="
