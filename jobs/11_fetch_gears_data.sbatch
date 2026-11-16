#!/usr/bin/env bash
#SBATCH --job-name=pj_gears_data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# GEARS processed perturb-seq datasets + GO/gene-universe aux files + the
# OFFICIAL pretrained Norman forward oracle (our reward oracle).
# Resumable: each file gets a .done marker and is skipped on resubmit.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

DV=https://dataverse.harvard.edu/api/access/datafile
GD="$PJ_DATA/gears"            # untrusted download target
mkdir -p "$GD/_archives"

# ---------------------------------------------------------------- datasets
for name in $GEARS_DATASETS; do
  eval "fid=\${DV_${name}:-}"
  if [ -z "${fid:-}" ]; then echo "[gears] no dataverse id for $name, skipping"; continue; fi
  zip="$GD/_archives/${name}.zip"
  h5="$GD/${name}/perturb_processed.h5ad"
  if [ -s "$h5" ]; then echo "[gears] $name already extracted -> $h5"; continue; fi
  echo ""
  echo "---------------- $name (dataverse $fid) ----------------"
  pj_fetch "$DV/$fid" "$zip" || { echo "[gears] WARN: $name download failed"; continue; }
  # archive contains <name>/perturb_processed.h5ad
  pj_unzip "$zip" "$GD" || { echo "[gears] WARN: extract failed for $name"; continue; }
  ls -lh "$GD/$name" 2>/dev/null | head
done

# ---------------------------------------------------------------- aux files
echo ""
echo "================ auxiliary files ================"
pj_fetch "$DV/$DV_gene2go_all"           "$GD/gene2go_all.pkl"                   || true
pj_fetch "$DV/$DV_essential_pert_genes"  "$GD/essential_all_data_pert_genes.pkl" || true
pj_fetch "$DV/$DV_go_essential_all"      "$GD/go_essential_all.csv"              || true

# ---------------------------------------------------------------- official oracle
echo ""
echo "================ official GEARS Norman checkpoint ================"
# Source: snap-stanford/GEARS@master demo/tutorial_inference_Norman.ipynb
WD="$PJ_WEIGHTS/gears_norman_official"
mkdir -p "$WD/_archives"
pj_fetch "$DV/$DV_gears_ckpt"       "$WD/_archives/model.zip"            || true
pj_fetch "$DV/$DV_gears_dataloader" "$WD/_archives/norman_umi_go.tar.gz" || true

if [ -s "$WD/_archives/model.zip" ] && [ ! -f "$WD/model_ckpt/model.pt" ]; then
  pj_unzip "$WD/_archives/model.zip" "$WD" || echo "[ckpt] WARN: extract failed"
fi
if [ -s "$WD/_archives/norman_umi_go.tar.gz" ] && [ ! -d "$WD/norman_umi_go" ]; then
  tar -xzf "$WD/_archives/norman_umi_go.tar.gz" -C "$WD" || echo "[ckpt] WARN: untar failed"
fi
echo "[ckpt] contents of $WD:"
find "$WD" -maxdepth 2 -not -path '*/_archives/*' | sed 's/^/  /' | head -30

# ---------------------------------------------------------------- summary
echo ""
echo "================ SUMMARY ================"
for name in $GEARS_DATASETS; do
  h5="$GD/${name}/perturb_processed.h5ad"
  if [ -s "$h5" ]; then printf '  %-32s OK   %s\n' "$name" "$(numfmt --to=iec < <(stat -c%s "$h5"))"
  else                  printf '  %-32s MISSING\n' "$name"; fi
done
for f in gene2go_all.pkl essential_all_data_pert_genes.pkl go_essential_all.csv; do
  if [ -s "$GD/$f" ]; then printf '  %-32s OK   %s\n' "$f" "$(numfmt --to=iec < <(stat -c%s "$GD/$f"))"
  else                     printf '  %-32s MISSING\n' "$f"; fi
done
if [ -f "$WD/model_ckpt/model.pt" ]; then
  printf '  %-32s OK   %s\n' "gears_official_model.pt" "$(numfmt --to=iec < <(stat -c%s "$WD/model_ckpt/model.pt"))"
else
  printf '  %-32s MISSING\n' "gears_official_model.pt"
fi
echo ""
echo "disk: $(df -h "$PJ_SCRATCH" | tail -1)"
echo "=== DONE ==="
