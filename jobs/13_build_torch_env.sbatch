#!/usr/bin/env bash
#SBATCH --job-name=pj_env
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/10703_pj/logs/%x_%j.out

# Build the torch env needed to RUN the GEARS forward oracle (the reward oracle).
# Separate from the analysis env (cellflow) which has scanpy but no torch.
#
# Pins matter: GEARS was last updated 2025-02 and breaks under numpy>=2 /
# pandas>=2.2. cu121 wheels cover sm_70 (v100), sm_75 (2080 Ti), sm_86 (a6000),
# sm_90 (h200), i.e. every GPU in this pool.
#
# Always exits 0 and writes a status file; the consumer job gates on that status
# so a build failure does not strand dependents in DependencyNeverSatisfied.
set -uo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/10703_pj}/env/common.sh"
[ -f "$PREAMBLE" ] || PREAMBLE=/zfsauton2/home/yixiz/10703_pj/env/common.sh
# shellcheck disable=SC1090
. "$PREAMBLE"
pj_consume_args "$@"
pj_banner
pj_mkdirs

ENV_DIR="/zfsauton/scratch/yixiz/envs/pertrl"
STATUS="$PJ_ARTIFACTS/torch_env_status.txt"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/zfsauton/scratch/yixiz/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"

echo "[env] uv: $PJ_UV ($("$PJ_UV" --version 2>&1))"
echo "[env] target: $ENV_DIR"

if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$PJ_UV" venv --python 3.11 "$ENV_DIR" || { echo "FAIL venv" > "$STATUS"; cat "$STATUS"; exit 0; }
fi
PY="$ENV_DIR/bin/python"
echo "[env] python: $($PY --version 2>&1)"

uvpip() { "$PJ_UV" pip install --python "$PY" "$@"; }

echo ""
echo "---------------- core numeric stack (pinned for GEARS) ----------------"
uvpip "numpy<2" "pandas<2.2" "scipy<1.14" "scikit-learn<1.6" tqdm pyyaml \
      "matplotlib<3.10" h5py || { echo "FAIL numeric" > "$STATUS"; cat "$STATUS"; exit 0; }

echo ""
echo "---------------- torch (cu121) ----------------"
uvpip torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 \
  || { echo "FAIL torch" > "$STATUS"; cat "$STATUS"; exit 0; }

echo ""
echo "---------------- scanpy / anndata (numpy<2 compatible) ----------------"
uvpip "anndata<0.11" "scanpy<1.11" || echo "[env] WARN: scanpy install issue"

echo ""
echo "---------------- PyG + GEARS ----------------"
uvpip "torch-geometric<2.7" networkx dcor || echo "[env] WARN: PyG/dcor install issue"
# cell-gears pulls loose deps; --no-deps keeps our pins intact.
uvpip --no-deps cell-gears==0.1.2 || echo "[env] WARN: cell-gears install issue"

echo ""
echo "================ verification ================"
"$PY" -I - <<'PYEOF'
import importlib, sys
ok, bad = [], []
for m in ["numpy","pandas","scipy","sklearn","torch","torch_geometric","anndata",
          "scanpy","networkx","dcor","gears","matplotlib","h5py"]:
    try:
        mod = importlib.import_module(m)
        ok.append(f"{m}=={getattr(mod,'__version__','?')}")
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}: {e}")
print("OK   :", ", ".join(ok))
for b in bad:
    print("BAD  :", b)
import torch
print("torch cuda built:", torch.version.cuda, "| cuda available here:", torch.cuda.is_available())
sys.exit(1 if bad else 0)
PYEOF
rc=$?

if [ $rc -eq 0 ]; then
  echo "OK $(date -Is) $ENV_DIR" > "$STATUS"
else
  echo "PARTIAL $(date -Is) $ENV_DIR (some imports failed, see log)" > "$STATUS"
fi
echo "[env] status -> $(cat "$STATUS")"
echo "[env] NOTE: set PJ_PY_TORCH=$PY"
echo "=== DONE (always exit 0) ==="
exit 0
