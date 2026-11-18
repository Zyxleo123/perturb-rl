#!/usr/bin/env bash
#SBATCH --job-name=pj_relfetch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Related-work code and data for scripts/related_work.py:
#   - IterPert (Huang et al., RECOMB 2024; Genentech/iterative-perturb-seq), whose
#     bmdal fork carries both IterPert itself and its whole active-learning benchmark
#     (BALD, BatchBALD, BADGE, Core-Set, LCMD, ACS-FW), and bmdal_reg (Holzmuller et
#     al., JMLR 2023), which that fork imports.
#   - IterPert's released prior-knowledge kernels (Dataverse 7377301, ~1 GB zip).
# Each repo goes in its OWN directory and is treated as untrusted data: nothing here
# runs an interpreter from inside it. The zip is checked against Dataverse's MD5 and
# is NOT extracted: related_work.py reads the members it needs through a restricted
# unpickler that only admits numpy arrays.
# Kept separate from 10_fetch_code on purpose: that job fetches the LATEST commit of
# every repo it knows, which would silently move the five baselines' recorded SHAs.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs
echo "request  : cpu x2, 8G, 2h"

cd "$PJ_CODE"
for spec in $RELATED_REPOS; do
  name="${spec%%|*}"; rest="${spec#*|}"
  url="${rest%%|*}";  ref="${rest##*|}"
  echo ""
  echo "---------------- $name ($url @ $ref) ----------------"
  if [ -d "$name/.git" ]; then
    echo "[code] exists, keeping the recorded commit: $(git -C "$name" rev-parse --short HEAD)"
  else
    rm -rf "$name"
    git clone --depth 1 --branch "$ref" "$url" "$name" || { echo "[code] FAILED: clone $name"; exit 1; }
  fi
  echo "[code] HEAD: $(git -C "$name" rev-parse --short HEAD) $(git -C "$name" log -1 --format=%cI)"
done

prov="$PJ_ARTIFACTS/code_provenance_related.json"
{
  echo "{"
  echo "  \"fetched_at\": \"$(date -Is)\","
  echo "  \"repos\": {"
  first=1
  for spec in $RELATED_REPOS; do
    name="${spec%%|*}"; rest="${spec#*|}"; url="${rest%%|*}"; ref="${rest##*|}"
    [ $first -eq 1 ] || echo ","
    first=0
    printf '    "%s": {"url": "%s", "ref": "%s", "sha": "%s", "commit_date": "%s"}' \
      "$name" "$url" "$ref" "$(git -C "$name" rev-parse HEAD)" "$(git -C "$name" log -1 --format=%cI)"
  done
  echo ""
  echo "  }"
  echo "}"
} > "$prov"
cat "$prov"

echo ""
echo "================ IterPert prior-knowledge kernels ================"
KD="$PJ_DATA/iterpert"
mkdir -p "$KD"
ZIP="$KD/knowledge_kernels_1k.zip"
pj_fetch "https://dataverse.harvard.edu/api/access/datafile/$DV_iterpert_kernels" "$ZIP"
got=$(md5sum "$ZIP" | cut -d' ' -f1)
echo "[md5] got $got, Dataverse says $MD5_iterpert_kernels"
if [ "$got" != "$MD5_iterpert_kernels" ]; then
  rm -f "$ZIP.done"     # so a resubmission downloads it again
  echo "[md5] MISMATCH: the download is corrupt or the record changed"; exit 1
fi
ls -l "$ZIP"
echo "=== DONE ==="
