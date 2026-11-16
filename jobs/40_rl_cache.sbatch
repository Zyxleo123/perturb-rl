#!/usr/bin/env bash
#SBATCH --job-name=pj_rlcache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Cache the gene-acquisition MDP's inputs for the RL jobs (41-42): signatures,
# queries and GO features built by rl_headroom.build itself, then a check that the
# cache reproduces GAPS part 4's random-x100 median. The RL jobs gate on that check.
# CPU only, analysis env (the h5ads need anndata; the RL env reads the npz).
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x8, 64G, 1h"

GD="$PJ_DATA/gears"
[ -s "$GD/gene2go_all.pkl" ] || pj_gate_fail "missing $GD/gene2go_all.pkl"
for d in $PJ_RLA_DATASETS; do
  [ -s "$GD/$d/perturb_processed.h5ad" ] || pj_gate_fail "missing $GD/$d/perturb_processed.h5ad"
done
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"

# shellcheck disable=SC2086
"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/rl_cache.py" \
  --gears-dir "$GD" --gene2go "$GD/gene2go_all.pkl" --out-dir "$PJ_ARTIFACTS" \
  --datasets $PJ_RLA_DATASETS

cat "$PJ_ARTIFACTS/rl_acquire/cache_check.json"
ls -la "$PJ_ARTIFACTS/rl_acquire" | sed 's/^/  /'
echo "=== DONE ==="
