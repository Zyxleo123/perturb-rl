#!/usr/bin/env bash
#SBATCH --job-name=pj_pdg_data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# PDGrapher (Gonzalez et al., Nature BME 2025) processed data from Zenodo.
#   genetic:  splits.tar.gz (12 MB) + torch_data_genetic.tar.gz  (21.7 GB)
#   chemical: torch_data_chemical.tar.gz (37.3 GB)  -- off unless FETCH_PDG_CHEMICAL=1
# Resumable via curl --continue-at plus .done markers.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

PD="$PJ_DATA/pdgrapher"
mkdir -p "$PD/_archives" "$PD/processed/torch_data"

echo "[disk] before: $(df -h "$PJ_SCRATCH" | tail -1)"

zfile() { echo "https://zenodo.org/api/records/$1/files/$2/content"; }

echo ""
echo "================ splits (small) ================"
pj_fetch "$(zfile "$ZENODO_PDG_GENETIC" splits.tar.gz)" "$PD/_archives/splits.tar.gz" || true
if [ -s "$PD/_archives/splits.tar.gz" ] && [ ! -d "$PD/processed/splits" ]; then
  tar -xzf "$PD/_archives/splits.tar.gz" -C "$PD/processed" || echo "[pdg] WARN: splits untar failed"
fi

echo ""
echo "================ genetic torch_data (21.7 GB) ================"
pj_fetch "$(zfile "$ZENODO_PDG_GENETIC" torch_data_genetic.tar.gz)" \
         "$PD/_archives/torch_data_genetic.tar.gz" || true
if [ -s "$PD/_archives/torch_data_genetic.tar.gz" ] && [ ! -f "$PD/processed/torch_data/.genetic_extracted" ]; then
  echo "[pdg] extracting genetic torch_data (this takes a while)"
  tar -xzf "$PD/_archives/torch_data_genetic.tar.gz" -C "$PD/processed/torch_data" \
    && touch "$PD/processed/torch_data/.genetic_extracted" \
    || echo "[pdg] WARN: genetic untar failed"
fi

if [ "${FETCH_PDG_CHEMICAL}" = "1" ]; then
  echo ""
  echo "================ chemical torch_data (37.3 GB) ================"
  pj_fetch "$(zfile "$ZENODO_PDG_CHEMICAL" torch_data_chemical.tar.gz)" \
           "$PD/_archives/torch_data_chemical.tar.gz" || true
  if [ -s "$PD/_archives/torch_data_chemical.tar.gz" ] && [ ! -f "$PD/processed/torch_data/.chemical_extracted" ]; then
    tar -xzf "$PD/_archives/torch_data_chemical.tar.gz" -C "$PD/processed/torch_data" \
      && touch "$PD/processed/torch_data/.chemical_extracted" \
      || echo "[pdg] WARN: chemical untar failed"
  fi
else
  echo ""
  echo "[pdg] skipping chemical data (FETCH_PDG_CHEMICAL=$FETCH_PDG_CHEMICAL)."
  echo "[pdg] our project targets GENETIC perturbation; rerun with FETCH_PDG_CHEMICAL=1 if needed."
fi

echo ""
echo "================ SUMMARY ================"
find "$PD/processed" -maxdepth 3 -type d | head -40 | sed 's/^/  /'
echo ""
echo "sizes:"
du -sh "$PD"/_archives/* 2>/dev/null | sed 's/^/  /'
du -sh "$PD/processed" 2>/dev/null | sed 's/^/  /'
echo "[disk] after: $(df -h "$PJ_SCRATCH" | tail -1)"
echo "=== DONE ==="
