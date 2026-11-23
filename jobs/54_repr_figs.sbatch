#!/usr/bin/env bash
#SBATCH --job-name=pj_reprfigs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# fig22 (representation anatomy) + paired significance tests, from artifacts/repr_*/ only.
# No model, no h5ad, no recomputation: the figure re-renders cheaply and cannot drift from
# the numbers in GAPS.md. Journal style, matching fig18-fig21. CPU, tiny.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x2, 8G, 30m"

[ -s "$PJ_ARTIFACTS/repr_probe/summary.json" ]  || pj_gate_fail "no repr_probe (./submit.sh reprprobe)"
[ -s "$PJ_ARTIFACTS/repr_select/summary.json" ] || pj_gate_fail "no repr_select (./submit.sh reprsel)"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$$"
# shellcheck disable=SC2086
"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/repr_figs.py" \
  --art-dir "$PJ_ARTIFACTS" --fig-dir "$PJ_FIGURES" --datasets $PJ_RLA_DATASETS

ls -la "$PJ_FIGURES" | grep -i fig22 | sed 's/^/  /' || true
echo "=== DONE ==="
