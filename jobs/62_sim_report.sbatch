#!/usr/bin/env bash
#SBATCH --job-name=pj_simreport
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Table for the sim-to-real acquisition runs, from saved artifacts only
# (artifacts/rl_acquire/sim/{baselines,runs} + the existing rl_acquire floors), every
# chosen set re-scored in the analysis env. Cheap; runs afterany so it reports partial
# results and lists failures.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x2, 8G, 30m"

[ -d "$PJ_ARTIFACTS/rl_acquire/sim/baselines" ] || pj_gate_fail "no sim baselines (upstream never ran)"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/rl_sim_report.py" --art-dir "$PJ_ARTIFACTS"
echo "=== DONE ==="
