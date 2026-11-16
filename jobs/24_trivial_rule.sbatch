#!/usr/bin/env bash
#SBATCH --job-name=pj_trivial
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Is "identify the perturbation" just reading which gene was knocked down?
# CPU only: reads the signature npz files job 20 already wrote. No GPU, no model.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

ls "$PJ_ARTIFACTS"/npz/*_signatures.npz >/dev/null 2>&1 \
  || pj_gate_fail "no signature npz files; run jobs/20_profile.sbatch first"

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/trivial_rule.py" --artifacts "$PJ_ARTIFACTS"
echo "=== DONE ==="
