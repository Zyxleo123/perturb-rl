#!/usr/bin/env bash
#SBATCH --job-name=pj_paperfigs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Re-render fig13/fig15/fig16/fig17 in the journal style of the case studies
# (fig18-fig21): serif STIX, panel letters, no on-figure titles, vector PDF + PNG.
# Reads only saved artifacts (generalization/summary.json, rl_acquire/records.jsonl,
# rl_acquire/runs/ppo_*.curve.jsonl) -- no recomputation. Cheap; rerun any time.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x2, 8G, 20m"

[ -s "$PJ_ARTIFACTS/rl_acquire/records.jsonl" ] \
  || pj_gate_fail "no rl_acquire/records.jsonl (jobs/43_rl_report never ran); nothing to draw for fig15/17"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/paper_figs.py" \
  --art-dir "$PJ_ARTIFACTS" --fig-dir "$PJ_FIGURES"

ls -la "$PJ_FIGURES"/fig13_* "$PJ_FIGURES"/fig15_* "$PJ_FIGURES"/fig16_* "$PJ_FIGURES"/fig17_* 2>/dev/null | sed 's/^/  /'
echo "=== DONE ==="
