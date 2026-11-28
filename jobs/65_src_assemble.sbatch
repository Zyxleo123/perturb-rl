#!/usr/bin/env bash
#SBATCH --job-name=pj_srcasm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Every raw source -> a drop-in rl_acquire.World cache, plus the feasibility probe
# (coverage of each target's candidates, signal ceiling, gene-pair transfer vs the
# K562<->RPE1 reference) -> artifacts/rl_acquire/src/src_check.json.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x4, 48G, 1h"
[ -x "$PJ_PY_TORCH" ] || pj_gate_fail "RL python not executable: $PJ_PY_TORCH"
SRC="$PJ_ARTIFACTS/rl_acquire/src"
ls "$SRC"/raw_*.npz >/dev/null 2>&1 || pj_gate_fail "no raw sources in $SRC (did 64_src_build run?)"
ls -la "$SRC"
"$PJ_PY_TORCH" -I "$PJ_HOME/scripts/src_cache.py" assemble --src-dir "$SRC" \
  --cache-dir "$PJ_ARTIFACTS/rl_acquire" --gene2go "$PJ_DATA/gears/gene2go_all.pkl"
echo "=== DONE ==="
