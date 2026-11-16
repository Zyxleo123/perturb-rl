#!/usr/bin/env bash
#SBATCH --job-name=pj_profile
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# CPU-only: reads the downloaded h5ads, writes JSON/JSONL/NPZ artifacts.
# No GPU -- asking for one would just queue behind real training work.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

GD="$PJ_DATA/gears"
# Gate: need at least one extracted dataset.
n_found=0
for d in $GEARS_DATASETS; do
  [ -s "$GD/$d/perturb_processed.h5ad" ] && n_found=$((n_found+1))
done
echo "[gate] $n_found / $(echo $GEARS_DATASETS | wc -w) GEARS datasets present under $GD"
[ "$n_found" -gt 0 ] || pj_gate_fail "no extracted GEARS h5ad found; run jobs/11_fetch_gears_data.sbatch"

[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"
echo "[py] $PJ_PY_ANALYSIS -> $("$PJ_PY_ANALYSIS" --version 2>&1)"

# -I: isolated mode. The data dirs are untrusted; we never cd into them and we
# pass every path as an argument.
"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/profile_datasets.py" \
  --gears-dir "$GD" \
  --pdgrapher-dir "$PJ_DATA/pdgrapher" \
  --out-dir "$PJ_ARTIFACTS" \
  --max-cells "${PJ_MAX_CELLS:-250000}"

echo ""
echo "=== artifacts ==="
ls -la "$PJ_ARTIFACTS" | sed 's/^/  /'
ls -la "$PJ_ARTIFACTS/npz" 2>/dev/null | sed 's/^/  /'
echo "=== DONE ==="
