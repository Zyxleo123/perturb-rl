#!/usr/bin/env bash
#SBATCH --job-name=pj_srcbuild
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-10
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%A_%a.out

# Extra source screens for sim-to-real (scripts/src_cache.py), one array task each:
#   0     fetch Replogle 2022 genome-wide K562 normalised pseudobulk -> raw_k562gw.npz
#   1-10  one LINCS CRISPR line from PDGrapher's genetic data -> raw_lincs_<L>.npz
# Resumable: a raw file that exists is skipped (PJ_SRC_FORCE=1 rebuilds).
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x4, 64G, 3h, array task ${SLURM_ARRAY_TASK_ID:-<none>}"
[ -x "$PJ_PY_TORCH" ] || pj_gate_fail "RL python not executable: $PJ_PY_TORCH"

SRC="$PJ_ARTIFACTS/rl_acquire/src"
mkdir -p "$SRC"
LINES=(A549 A375 AGS BICR6 ES2 HT29 MCF7 PC3 U251MG YAPC)
I="${SLURM_ARRAY_TASK_ID:?not an array task}"

if [ "$I" -eq 0 ]; then
  OUT="$SRC/raw_k562gw.npz"
  if [ -s "$OUT" ] && [ -z "${PJ_SRC_FORCE:-}" ]; then echo "exists, skipping: $OUT"; echo "=== DONE ==="; exit 0; fi
  D="$PJ_DATA/replogle_gw"; mkdir -p "$D"
  H5="$D/K562_gwps_normalized_bulk_01.h5ad"
  pj_fetch "$PJ_GW_BULK_URL" "$H5"
  "$PJ_PY_TORCH" -I "$PJ_HOME/scripts/src_cache.py" gw --h5 "$H5" --out-dir "$SRC"
else
  L="${LINES[$((I - 1))]}"
  OUT="$SRC/raw_lincs_${L}.npz"
  if [ -s "$OUT" ] && [ -z "${PJ_SRC_FORCE:-}" ]; then echo "exists, skipping: $OUT"; echo "=== DONE ==="; exit 0; fi
  PT="$PJ_DATA/pdgrapher/processed/torch_data/real_lognorm/data_backward_${L}.pt"
  [ -s "$PT" ] || pj_gate_fail "missing $PT"
  "$PJ_PY_TORCH" -I "$PJ_HOME/scripts/src_cache.py" lincs --line "$L" --pt "$PT" --out-dir "$SRC"
fi
echo "=== DONE ==="
