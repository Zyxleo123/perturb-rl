#!/usr/bin/env bash
#SBATCH --job-name=pj_optreport
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# REINFORCE vs classical optimisers (simulated annealing, greedy + swap) at the same
# reward-evaluation budget, peek regime, from saved artifacts only (rl_acquire/runs +
# baselines), every chosen set re-scored in the analysis env -> rl_acquire/opt/.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x2, 16G, 30m"

[ -d "$PJ_ARTIFACTS/rl_acquire/runs" ] || pj_gate_fail "no rl_acquire runs (upstream never ran)"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/opt_report.py" --art-dir "$PJ_ARTIFACTS"
echo "=== DONE ==="
