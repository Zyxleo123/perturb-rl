#!/usr/bin/env bash
#SBATCH --job-name=pj_rlhead
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Is there an RL-shaped problem inside the unseen-gene gap (GAPS.md gap 0)?
#   part 1  objective headroom  -- does re-scoring the SAME predictions recover
#           discrimination? (if yes: a ranking/RL objective has something to win)
#   part 2  acquisition headroom -- does it matter WHICH genes were measured?
#           (if yes: choosing them is a budgeted sequential decision problem)
# CPU only: reads h5ads + gene2go, fits ridge/kNN. No GPU, no model.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x8, 96G, 6h"

GD="$PJ_DATA/gears"
[ -s "$GD/gene2go_all.pkl" ] || pj_gate_fail "missing $GD/gene2go_all.pkl"
for d in replogle_k562_essential replogle_rpe1_essential; do
  [ -s "$GD/$d/perturb_processed.h5ad" ] || pj_gate_fail "missing $GD/$d/perturb_processed.h5ad"
done
[ -x "$PJ_PY_ANALYSIS" ] || pj_gate_fail "analysis python not executable: $PJ_PY_ANALYSIS"

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/rl_headroom.py" \
  --gears-dir "$GD" --gene2go "$GD/gene2go_all.pkl" --out-dir "$PJ_ARTIFACTS" \
  ${PJ_RL_PARTS:+--parts $PJ_RL_PARTS}

ls -la "$PJ_ARTIFACTS/rl_headroom" | sed 's/^/  /'
echo "=== DONE ==="
