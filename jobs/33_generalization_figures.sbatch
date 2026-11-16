#!/usr/bin/env bash
#SBATCH --job-name=pj_genfig
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Redraw the generalization-headroom figures from artifacts/generalization/ only.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x2, 8G, 30m"

R="$PJ_ARTIFACTS/generalization/results.jsonl"
[ -f "$R" ] || pj_gate_fail "upstream never ran: $R absent (jobs/25_generalization.sbatch)"
[ -s "$R" ] || pj_gate_fail "upstream ran but wrote no rows: $R is empty -- read its .out"

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/generalization_figures.py" \
  --artifacts "$PJ_ARTIFACTS" --figures "$PJ_FIGURES"
echo "=== DONE ==="
