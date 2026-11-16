#!/usr/bin/env bash
#SBATCH --job-name=pj_general
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Where does inverse design have headroom? Ceiling vs the best CHEAP predictor
# along the three axes where the atlas has no entry: unseen gene, unseen
# combination, unseen cell context. CPU only: reads h5ads + the saved GEARS
# predictions (npz/norman_oracle_signatures.npz); runs no model.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x8, 96G, 4h"

GD="$PJ_DATA/gears"
UMI="$PJ_WEIGHTS/gears_norman_official/norman_umi_go"
ORC="$PJ_ARTIFACTS/npz/norman_oracle_signatures.npz"
for f in "$GD/replogle_k562_essential/perturb_processed.h5ad" \
         "$GD/replogle_rpe1_essential/perturb_processed.h5ad" \
         "$UMI/perturb_processed.h5ad" "$ORC"; do
  [ -s "$f" ] || pj_gate_fail "missing input $f"
done
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/generalization_gaps.py" \
  --gears-dir "$GD" --umi-dir "$UMI" --oracle-npz "$ORC" \
  --gene2go "$GD/gene2go_all.pkl" --out-dir "$PJ_ARTIFACTS"

ls -la "$PJ_ARTIFACTS/generalization" | sed 's/^/  /'
echo "=== DONE ==="
