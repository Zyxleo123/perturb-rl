#!/usr/bin/env bash
#SBATCH --job-name=pj_oracle_retrain
#SBATCH --partition=legacy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# WHY THIS JOB EXISTS
# The only published GEARS checkpoint is `gears_misc_umi_no_test`, trained with
# split='no_test' -- i.e. on every perturbation in Norman. jobs/21_oracle_probe
# can therefore only produce TRAIN-SET numbers, an optimistic upper bound.
#
# RL post-training will push the generator toward perturbations the oracle has
# NOT seen, which is exactly the regime those numbers say nothing about. So we
# retrain GEARS ourselves under split='simulation', which holds out whole
# perturbations (and, for Norman, unseen combos), and re-probe on the held-out
# set. Without this the reward signal is unvalidated where it matters most.
#
# Resource note: GEARS on Norman fits an 11 GB rtx_2080_ti; `legacy` had idle
# nodes while `general` (a6000) was fully booked. Training builds the cell-graph
# dataset, so host RAM (80 G) matters more than GPU RAM here.
#
# Resumable: skips a (dataset, split, seed) whose model.pt already exists.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

DS="${PJ_RETRAIN_DATASET:-norman}"
SPLIT="${PJ_RETRAIN_SPLIT:-simulation}"
SEED="${PJ_RETRAIN_SEED:-1}"
EPOCHS="${PJ_RETRAIN_EPOCHS:-20}"
OUT="$PJ_WEIGHTS/gears_${DS}_${SPLIT}_seed${SEED}"

[ -s "$PJ_DATA/gears/$DS/perturb_processed.h5ad" ] \
  || pj_gate_fail "no $DS h5ad under $PJ_DATA/gears (run jobs/11_fetch_gears_data.sbatch)"
[ -x "$PJ_PY_TORCH" ] \
  || pj_gate_fail "torch env missing ($(cat "$PJ_ARTIFACTS/torch_env_status.txt" 2>/dev/null || echo 'no status'))"

if [ -f "$OUT/model.pt" ]; then
  echo "[retrain] $OUT/model.pt already exists; re-probing only."
else
  echo "[retrain] training GEARS dataset=$DS split=$SPLIT seed=$SEED epochs=$EPOCHS -> $OUT"
  "$PJ_PY_TORCH" -I "$PJ_HOME/scripts/retrain_oracle.py" \
    --gears-dir "$PJ_DATA/gears" \
    --dataset "$DS" --split "$SPLIT" --seed "$SEED" --epochs "$EPOCHS" \
    --out-dir "$OUT" --artifacts "$PJ_ARTIFACTS" --device cuda:0
fi

echo ""
echo "=== held-out probe against the SAME trivial baselines as job 21 ==="
"$PJ_PY_TORCH" -I "$PJ_HOME/scripts/oracle_probe.py" \
  --dataloader-dir "$PJ_DATA/gears/$DS" \
  --ckpt-dir "$OUT" \
  --out-dir "$PJ_ARTIFACTS" \
  --work-dir "$PJ_SCRATCH/oracle_work_${DS}_${SPLIT}" \
  --split "$SPLIT" --seed "$SEED" --device cuda:0 \
  --held-out-only \
  --out-name "oracle_probe_heldout_${DS}_${SPLIT}.json"

echo "=== DONE ==="
