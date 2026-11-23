#!/usr/bin/env bash
#SBATCH --job-name=pj_reprprobe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Failure anatomy of the unseen-gene predictor (scripts/repr_probe.py): splits GAPS
# gap 0's "fidelity without discrimination" into predictor-class / similarity /
# scoring-geometry causes, and measures the CEILING of the neighbour-averaging class
# under a perfect gene-gene similarity. Reads only the RL cache + the GEARS h5ad.
# Writes artifacts/repr_probe/.  CPU only: no GPU is needed to rank cosines.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x8, 24G, 2h"

GD="$PJ_DATA/gears"
[ -s "$PJ_ARTIFACTS/rl_acquire/cache_check.json" ] || pj_gate_fail "no RL cache (./submit.sh rlcache)"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"
for ds in $PJ_RLA_DATASETS; do
  [ -s "$GD/$ds/perturb_processed.h5ad" ] || pj_gate_fail "missing $GD/$ds/perturb_processed.h5ad"
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
       MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_$$"

# shellcheck disable=SC2086
"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/repr_probe.py" \
  --art-dir "$PJ_ARTIFACTS" --gears-dir "$GD" --datasets $PJ_RLA_DATASETS

echo "--- report.md ---"
cat "$PJ_ARTIFACTS/repr_probe/report.md" || true
ls -la "$PJ_ARTIFACTS/repr_probe" | sed 's/^/  /'
echo "=== DONE ==="
