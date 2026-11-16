#!/usr/bin/env bash
#SBATCH --job-name=pj_invgap
#SBATCH --partition=legacy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Measure the PERFORMANCE GAP in inverse design on one comparable task:
#   ceiling (empirical centroid) vs oracle (GEARS-predicted) vs non-conditional
#   vs chance. The ceiling-minus-oracle gap is the headroom lost to the reward
#   model rather than to the generator.
#
# Resource note (myfree, 2026-11-13): general/a6000 fully booked; legacy had idle
# rtx_2080_ti nodes with ~188 G host RAM. GPU RAM is not the constraint (small
# model, 300-cell batches) -- host RAM is, because PertData.load() unpickles a
# 3.7 G cell-graph cache. 80 G is comfortable; do not drop below 64 G.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

W="$PJ_WEIGHTS/gears_norman_official"
[ -f "$W/model_ckpt/model.pt" ] || pj_gate_fail "no GEARS checkpoint (run jobs/11_fetch_gears_data.sbatch)"
[ -f "$W/norman_umi_go/perturb_processed.h5ad" ] || pj_gate_fail "no dataloader at $W/norman_umi_go"
[ -x "$PJ_PY_TORCH" ] || pj_gate_fail "torch env missing ($(cat "$PJ_ARTIFACTS/torch_env_status.txt" 2>/dev/null || echo none))"

# PJ_INVGAP_FAST=1 skips the 3.7 G cell-graph unpickle, which took >25 min on a
# contended NFS mount (measured 2026-11-13 while a 20 G download ran). predict()
# never reads it; see scripts/inverse_gap.py for why this is safe.
FAST_FLAG=""
if [ "${PJ_INVGAP_FAST:-0}" = "1" ]; then
  FAST_FLAG="--skip-cell-graphs"
  echo "[invgap] fast path: skipping cell_graphs.pkl"
fi

"$PJ_PY_TORCH" -I "$PJ_HOME/scripts/inverse_gap.py" \
  --dataloader-dir "$W/norman_umi_go" \
  --ckpt-dir "$W/model_ckpt" \
  --out-dir "$PJ_ARTIFACTS" \
  --work-dir "$PJ_SCRATCH/invgap_work" \
  --dataset-label norman \
  --device cuda:0 \
  --queries-per-cond "${PJ_INVGAP_QPC:-3}" \
  --max-candidates "${PJ_INVGAP_MAXCAND:-0}" \
  $FAST_FLAG

echo ""
echo "=== headline gap ==="
"$PJ_PY_TORCH" -I -c "
import json
d=json.load(open('$PJ_ARTIFACTS/inverse_gap.json'))
print('status:', d['status'])
print('candidates:', d.get('n_candidates'))
f=d.get('oracle_fidelity') or {}
for k,v in f.items(): print(f'  fidelity.{k} = {v:.4f}' if isinstance(v,float) else f'  {k}={v}')
h=(d.get('headline') or {}).get('per_pool_size') or {}
print(f\"{'pool':>6}{'ceiling':>10}{'oracle':>10}{'gap':>10}{'retained':>10}\")
for n,r in sorted(h.items(), key=lambda kv: int(kv[0])):
    print(f\"{n:>6}{r['ceiling_top1']:>10.3f}{r['oracle_top1']:>10.3f}\"
          f\"{r['absolute_gap']:>10.3f}{r['frac_of_ceiling_retained']:>10.3f}\")
"
echo "=== DONE ==="
