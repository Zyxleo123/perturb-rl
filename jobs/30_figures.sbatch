#!/usr/bin/env bash
#SBATCH --job-name=pj_figures
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Draw every figure from artifacts on disk. Cheap and re-runnable:
#   ./submit.sh figures
# Needs no GPU and no model -- that is the point of writing artifacts first.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

# Distinguish "upstream never ran" from "upstream ran and produced nothing".
if [ ! -f "$PJ_ARTIFACTS/dataset_profiles.json" ]; then
  pj_gate_fail "dataset_profiles.json absent -- jobs/20_profile.sbatch has not produced artifacts yet"
fi
echo "[gate] artifacts present:"
for f in dataset_profiles.json condition_stats.jsonl retrieval.jsonl oracle_probe.json; do
  if [ -s "$PJ_ARTIFACTS/$f" ]; then
    printf '  %-26s OK   %s bytes\n' "$f" "$(stat -c%s "$PJ_ARTIFACTS/$f")"
  else
    printf '  %-26s absent (its figure will be skipped)\n' "$f"
  fi
done

"$PJ_PY_ANALYSIS" -I "$PJ_HOME/scripts/make_figures.py" \
  --artifacts "$PJ_ARTIFACTS" \
  --figures "$PJ_FIGURES"

echo ""
echo "=== figures ==="
ls -la "$PJ_FIGURES" | sed 's/^/  /'
echo "=== DONE ==="
