#!/usr/bin/env bash
#SBATCH --job-name=pj_readout
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Recover the readout gene names behind the RL caches' signature columns
# (scripts/rl_readout.py), verified against the cache: masked count and a few
# signature rows recomputed from the raw h5ad. Needed to put one line's signature
# beside the other's (the imputed-signature simulator in scripts/rl_sim.py).
# Exits 1 if the recovered names do not reproduce the cache.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x4, 32G, 45m"

GD="$PJ_DATA/gears"
CK="$PJ_ARTIFACTS/rl_acquire/cache_check.json"
[ -s "$CK" ] || pj_gate_fail "missing $CK (did jobs/40_rl_cache.sbatch run?)"
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/rl_readout.py" --gears-dir "$GD" \
  --cache-dir "$PJ_ARTIFACTS/rl_acquire" --out-dir "$PJ_ARTIFACTS"
echo "=== DONE ==="
