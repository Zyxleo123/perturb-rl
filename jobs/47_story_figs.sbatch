#!/usr/bin/env bash
#SBATCH --job-name=pj_storyfigs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Gap / improvement case-study figures (fig18-fig21), drawn from artifacts/story/
# only -- no recomputation. Cheap; rerun any time with ./submit.sh storyfigs.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x2, 8G, 20m"

ls "$PJ_ARTIFACTS"/story/story_*.npz >/dev/null 2>&1 \
  || pj_gate_fail "no artifacts/story/story_*.npz (upstream jobs/46_story_data never ran or failed)"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/story_cases.py" draw \
  --art-dir "$PJ_ARTIFACTS" --fig-dir "$PJ_FIGURES"

ls -la "$PJ_FIGURES"/fig1[89]_* "$PJ_FIGURES"/fig2[0-9]_* 2>/dev/null | sed 's/^/  /'
echo "=== DONE ==="
