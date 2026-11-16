#!/usr/bin/env bash
#SBATCH --job-name=pj_oracle
#SBATCH --partition=legacy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Validate the GEARS forward oracle (the project's reward + cycle-consistency map).
#
# Resource note (myfree, 2026-11-13): `general` (a6000) had 0 free GPUs; `legacy`
# had gpu16/17/18 fully idle with 188G host RAM. GEARS inference on Norman is a
# small model over 300-cell batches, so an 11 GB rtx_2080_ti is ample and we do
# not pin a GPU class. The h5ad is 3.2 GB -> 64 G host RAM is comfortable once we
# skip the 3.7 GB cell-graph pickle (predict() does not need it).
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

W="$PJ_WEIGHTS/gears_norman_official"
CKPT="$W/model_ckpt"
DL="$W/norman_umi_go"

# ---- gates: explain and exit 0, never strand the figure job ----
[ -f "$CKPT/model.pt" ] || pj_gate_fail "no GEARS checkpoint at $CKPT (run jobs/11_fetch_gears_data.sbatch)"
[ -f "$CKPT/config.pkl" ] || pj_gate_fail "no config.pkl at $CKPT"
[ -f "$DL/perturb_processed.h5ad" ] || pj_gate_fail "no matching dataloader at $DL"

STATUS="$PJ_ARTIFACTS/torch_env_status.txt"
if [ ! -x "$PJ_PY_TORCH" ]; then
  pj_gate_fail "torch env missing at $PJ_PY_TORCH (status: $(cat "$STATUS" 2>/dev/null || echo none))"
fi
echo "[py] $PJ_PY_TORCH -> $("$PJ_PY_TORCH" --version 2>&1)"
echo "[env status] $(cat "$STATUS" 2>/dev/null || echo 'no status file')"

if ! "$PJ_PY_TORCH" -I -c "import gears, torch, torch_geometric" 2>&1; then
  pj_gate_fail "torch env cannot import gears/torch/torch_geometric"
fi

WORK="$PJ_SCRATCH/oracle_work"
mkdir -p "$WORK"

# PJ_ORACLE_FAST=1 skips the 3.7 G cell-graph unpickle, which took >30 min on a
# contended NFS mount (measured 2026-11-13). predict() never reads it.
FAST_FLAG=""
if [ "${PJ_ORACLE_FAST:-0}" = "1" ]; then
  FAST_FLAG="--skip-cell-graphs"
  echo "[oracle] fast path: skipping cell_graphs.pkl"
fi

"$PJ_PY_TORCH" -I "$PJ_HOME/scripts/oracle_probe.py" \
  --dataloader-dir "$DL" \
  --ckpt-dir "$CKPT" \
  --out-dir "$PJ_ARTIFACTS" \
  --work-dir "$WORK" \
  --split "${PJ_ORACLE_SPLIT:-no_test}" \
  --device cuda:0 \
  --max-conditions "${PJ_ORACLE_MAX_COND:-0}" \
  $FAST_FLAG

echo ""
echo "=== result head ==="
"$PJ_PY_TORCH" -I -c "
import json,sys
d=json.load(open('$PJ_ARTIFACTS/oracle_probe.json'))
print('status:', d['status'])
print('conditions evaluated:', d.get('n_conditions_evaluated'))
for k,v in (d.get('baselines') or {}).items(): print(f'  {k} = {v}')
"
echo "=== DONE ==="
