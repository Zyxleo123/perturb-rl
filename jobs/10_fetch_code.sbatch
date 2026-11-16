#!/usr/bin/env bash
#SBATCH --job-name=pj_code
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Clone the baseline code repos. Each goes in its OWN empty directory and is
# treated as untrusted data: nothing here runs an interpreter from inside them.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

cd "$PJ_CODE"

for spec in $CODE_REPOS; do
  name="${spec%%|*}"; rest="${spec#*|}"
  url="${rest%%|*}";  ref="${rest##*|}"
  echo ""
  echo "---------------- $name ($url @ $ref) ----------------"
  if [ -d "$name/.git" ]; then
    echo "[code] exists; fetching latest on $ref"
    git -C "$name" fetch --depth 1 origin "$ref" && \
      git -C "$name" checkout -q FETCH_HEAD || echo "[code] WARN: update failed for $name"
  else
    rm -rf "$name"
    git clone --depth 1 --branch "$ref" "$url" "$name" || {
      echo "[code] WARN: clone failed for $name"; continue; }
  fi
  echo "[code] HEAD: $(git -C "$name" rev-parse --short HEAD) $(git -C "$name" log -1 --format=%cI)"
  du -sh "$name" | sed 's/^/[code] size: /'
done

echo ""
echo "=== record provenance ==="
prov="$PJ_ARTIFACTS/code_provenance.json"
{
  echo "{"
  echo "  \"fetched_at\": \"$(date -Is)\","
  echo "  \"repos\": {"
  first=1
  for spec in $CODE_REPOS; do
    name="${spec%%|*}"; rest="${spec#*|}"; url="${rest%%|*}"; ref="${rest##*|}"
    [ -d "$name/.git" ] || continue
    sha=$(git -C "$name" rev-parse HEAD)
    cdate=$(git -C "$name" log -1 --format=%cI)
    [ $first -eq 1 ] || echo ","
    first=0
    printf '    "%s": {"url": "%s", "ref": "%s", "sha": "%s", "commit_date": "%s"}' \
      "$name" "$url" "$ref" "$sha" "$cdate"
  done
  echo ""
  echo "  }"
  echo "}"
} > "$prov"
cat "$prov"

echo ""
echo "=== DONE: code in $PJ_CODE ==="
ls -la "$PJ_CODE"
