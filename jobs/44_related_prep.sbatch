#!/usr/bin/env bash
#SBATCH --job-name=pj_relprep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# IterPert's released prior-knowledge kernels, re-expressed on OUR candidate genes:
# each prior's per-gene embedding (feat.pkl) aligned to the cache's candidates with the
# authors' own mean imputation, kernel = feat feat^T (checked against their kernel.pkl).
# K562 gets IterPert's exact 8-prior list; RPE1 drops rpe1_kernel (the deployment
# screen's own signatures) and gains a K562 Perturb-seq prior built the same way.
# Writes artifacts/rl_acquire/related/{priors_<dataset>.npz,priors_check.json}.
# Torch env: the pickles reference numpy.core.*, which numpy 2.x (analysis env) renamed.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x8, 48G, 1h"

ZIP="$PJ_DATA/iterpert/knowledge_kernels_1k.zip"
CK="$PJ_ARTIFACTS/rl_acquire/cache_check.json"
[ -s "$ZIP" ] && [ -f "$ZIP.done" ] || pj_gate_fail "missing $ZIP (did jobs/14_fetch_related.sbatch run?)"
[ -s "$CK" ] || pj_gate_fail "missing $CK (did jobs/40_rl_cache.sbatch run?)"
[ -x "$PJ_PY_TORCH" ] || pj_gate_fail "RL python not executable: $PJ_PY_TORCH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

"$PJ_PY_TORCH" -I "$PJ_HOME/scripts/related_work.py" prep \
  --zip "$ZIP" --cache-dir "$PJ_ARTIFACTS/rl_acquire" --out-dir "$PJ_ARTIFACTS"

cat "$PJ_ARTIFACTS/rl_acquire/related/priors_check.json"
echo "=== DONE ==="
