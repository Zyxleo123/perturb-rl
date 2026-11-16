#!/usr/bin/env bash
#SBATCH --job-name=pj_pdg_repair
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Job 12's tar died mid-extract (exit 1, no WARN line) and left
# data_backward_HT29.pt at 0 bytes. Re-extract and then VERIFY every member,
# so a partial extraction can never look like success again.
#
# `set -e` deliberately NOT applied to the tar step: we want the verification to
# run and report even when tar fails again.
set -uo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

PD="$PJ_DATA/pdgrapher"
TARBALL="$PD/_archives/torch_data_genetic.tar.gz"
DEST="$PD/processed/torch_data"

[ -s "$TARBALL" ] || pj_gate_fail "tarball missing: $TARBALL"
echo "[repair] tarball: $(numfmt --to=iec < <(stat -c%s "$TARBALL"))"
echo "[repair] free space: $(df -h "$PJ_SCRATCH" | tail -1)"

echo ""
echo "=== zero-length or missing members BEFORE repair ==="
find "$DEST" -name '*.pt' -size 0 -printf '  ZERO  %p\n' 2>/dev/null || true

echo ""
echo "=== re-extracting (overwrites in place) ==="
tar -xzvf "$TARBALL" -C "$DEST" 2>&1 | tail -40
rc=${PIPESTATUS[0]}
echo "[repair] tar exit code: $rc"

echo ""
echo "=== verification ==="
bad=0
while IFS= read -r f; do
  sz=$(stat -c%s "$f")
  if [ "$sz" -eq 0 ]; then
    printf '  ZERO     %s\n' "$(basename "$f")"; bad=$((bad+1))
  else
    printf '  ok %8s %s\n' "$(numfmt --to=iec <<< "$sz")" "$(basename "$f")"
  fi
done < <(find "$DEST" -name '*.pt' | sort)

echo ""
echo "[repair] total: $(du -sh "$DEST" | cut -f1), zero-length files: $bad"
if [ "$bad" -eq 0 ] && [ "$rc" -eq 0 ]; then
  touch "$DEST/.genetic_extracted"
  echo "[repair] OK -- marked complete"
else
  echo "[repair] INCOMPLETE: tar rc=$rc, $bad zero-length file(s)."
  echo "[repair] PDGrapher's own data may ship HT29 empty; check the Zenodo record"
  echo "[repair] before assuming our download is at fault. Note the paper reports"
  echo "[repair] HT29 results, so an empty file here is suspicious."
fi
echo "=== DONE (exit 0 regardless, so dependents still run) ==="
exit 0
