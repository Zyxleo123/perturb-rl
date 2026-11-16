#!/usr/bin/env bash
#SBATCH --job-name=pj_cases
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Case-study figures: named individual perturbations, not aggregate metrics.
# CPU only. Gene labels are recovered from the dataloader h5ad in backed mode
# (var only, X untouched) rather than by re-running the GEARS GPU job.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

ls "$PJ_ARTIFACTS"/npz/*_signatures.npz >/dev/null 2>&1 \
  || pj_gate_fail "no signature npz; run jobs/20_profile.sbatch first"

ONPZ="$PJ_ARTIFACTS/npz/norman_oracle_signatures.npz"
H5="$PJ_WEIGHTS/gears_norman_official/norman_umi_go/perturb_processed.h5ad"
if [ -f "$ONPZ" ] && [ -f "$H5" ]; then
  echo "=== ensuring oracle npz carries gene names ==="
  "$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/add_gene_names.py" --npz "$ONPZ" --h5ad "$H5" \
    || echo "[cases] WARN: could not add gene names; fig11/12 will use positional labels"
fi

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/case_studies.py" \
  --artifacts "$PJ_ARTIFACTS" --figures "$PJ_FIGURES"

echo ""; ls -la "$PJ_FIGURES" | sed 's/^/  /'
echo "=== DONE ==="
