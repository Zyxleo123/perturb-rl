#!/usr/bin/env bash
#SBATCH --job-name=pj_reprpred
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Does a better gene representation close the unseen-gene gap? (scripts/repr_predict.py)
# Holds the task and the predictor fixed and swaps only the similarity that picks the
# kNN's neighbours: GO Jaccard (what GAPS reports) vs IterPert's 8 released multimodal
# prior kernels vs two richer GO views vs a learned out-of-fold combination, against the
# oracle-similarity and predictor-class ceilings from repr_probe. CPU only.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x8, 24G, 3h, knn=${PJ_REPR_KNN:-10}"

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
"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/repr_predict.py" \
  --art-dir "$PJ_ARTIFACTS" --gears-dir "$GD" --datasets $PJ_RLA_DATASETS \
  --knn "${PJ_REPR_KNN:-10}"

echo "--- report.md ---"
cat "$PJ_ARTIFACTS/repr_predict/report.md" || true
echo "=== DONE ==="
