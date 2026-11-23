#!/usr/bin/env bash
#SBATCH --job-name=pj_reprcurve
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# How good does a gene representation have to be? (scripts/repr_curve.py)
# repr_select leaves a verdict but no target: the combiner is worth 3.5x given a perfect
# similarity and nothing given GO. This sweeps similarity QUALITY continuously, by
# corrupting the true signature similarity, and reads off retrieval at each level, so the
# negative becomes a specification a future representation can be measured against. The
# real representations are placed on the same axes. CPU only.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x8, 12G, 4h, knn=${PJ_REPR_KNN:-10}"

GD="$PJ_DATA/gears"
REL="$PJ_ARTIFACTS/rl_acquire/related"
[ -s "$PJ_ARTIFACTS/rl_acquire/cache_check.json" ] || pj_gate_fail "no RL cache (./submit.sh rlcache)"
[ -s "$REL/priors_check.json" ] || pj_gate_fail "no aligned prior kernels (./submit.sh related_fetch)"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"
for ds in $PJ_RLA_DATASETS; do
  [ -s "$GD/$ds/perturb_processed.h5ad" ] || pj_gate_fail "missing $GD/$ds/perturb_processed.h5ad"
  [ -s "$REL/priors_$ds.npz" ]            || pj_gate_fail "missing aligned priors for $ds"
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
       MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_$$"

# shellcheck disable=SC2086
"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/repr_curve.py" \
  --art-dir "$PJ_ARTIFACTS" --gears-dir "$GD" --datasets $PJ_RLA_DATASETS \
  --knn "${PJ_REPR_KNN:-10}"

echo "--- report.md ---"
cat "$PJ_ARTIFACTS/repr_curve/report.md" || true
echo "=== DONE ==="
