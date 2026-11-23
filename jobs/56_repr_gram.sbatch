#!/usr/bin/env bash
#SBATCH --job-name=pj_reprgram
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Estimate the combiner's b instead of plugging it in (scripts/repr_gram.py).
#
# repr_select showed the ridge combiner is worth 3.4-3.6x on the oracle similarity and
# only TRADES on real ones; repr_curve's calibration ablation showed it reads magnitudes,
# not just order. This job follows that to its conclusion: the combiner's input is an
# inner product b_j = <y_j,y_g>, the oracle row is simply the combiner handed an exact b,
# and no stage so far ever ESTIMATED b -- every row plugs in s(g,j)||y_j||.
#
# Part 1 is the 2x2 {real, oracle} neighbourhood x {plug-in, measured} b, which bounds
# what fixing b alone can buy before any method is trusted. Part 2 fits the estimate:
# isotonic (ordering preserved, magnitudes only) and a ridge on pair features, both out
# of fold over genes, on prior views alone and with the other line's measured cosine.
# Reported on all genes and split on whether the other line actually measured the gene.
# CPU only; no GPU is needed to solve a 100x100 system.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x8, 24G, 4h, knn=${PJ_REPR_KNN:-10}"

GD="$PJ_DATA/gears"
REL="$PJ_ARTIFACTS/rl_acquire/related"
[ -s "$PJ_ARTIFACTS/rl_acquire/cache_check.json" ] || pj_gate_fail "no RL cache (./submit.sh rlcache)"
[ -s "$REL/priors_check.json" ] || pj_gate_fail "no aligned prior kernels (./submit.sh related_fetch)"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"
for ds in $PJ_RLA_DATASETS; do
  [ -s "$GD/$ds/perturb_processed.h5ad" ] || pj_gate_fail "missing $GD/$ds/perturb_processed.h5ad"
  [ -s "$REL/priors_$ds.npz" ]            || pj_gate_fail "missing aligned priors for $ds"
  # The two kNN rows are asserted against repr_select's saved per-gene ranks before any
  # new number is reported, so the comparison is to the same measurement GAPS.md shows.
  [ -s "$PJ_ARTIFACTS/repr_select/result_$ds.npz" ] \
    || pj_gate_fail "no repr_select result for $ds to reconcile against (./submit.sh reprsel)"
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
       MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_$$"

# shellcheck disable=SC2086
"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/repr_gram.py" \
  --art-dir "$PJ_ARTIFACTS" --gears-dir "$GD" --datasets $PJ_RLA_DATASETS \
  --knn "${PJ_REPR_KNN:-10}"

echo "--- report.md ---"
cat "$PJ_ARTIFACTS/repr_gram/report.md" || true
echo "=== DONE ==="
