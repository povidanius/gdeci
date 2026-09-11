#!/usr/bin/env bash
# Run the O(n log n) graph-Dirichlet direction score on the full benchmark suite.
#
#   ./run_nlogn.sh                 verify, then score all 1899 pairs x 10 seeds
#   ./run_nlogn.sh --quick         verify, then a single seed (faster)
#   ./run_nlogn.sh --scaling       also time n = 1e3 .. 1e6
#
# See run_nlogn.txt for what the numbers mean and how to change the settings.

set -euo pipefail
START=$SECONDS

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python}"
SEEDS="0,1,2,3,4,5,6,7,8,9"
OUT="results_nlogn.csv"
SCALING=0

while [ $# -gt 0 ]; do
  case "$1" in
    --quick)   SEEDS="0" ;;
    --scaling) SCALING=1 ;;
    --seeds)   shift; SEEDS="$1" ;;
    --out)     shift; OUT="$1" ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see run_nlogn.txt)" >&2; exit 2 ;;
  esac
  shift
done

# Only numpy, scipy and pandas are needed; torch and cdt are not.
"$PY" - <<'EOF'
import sys
missing = []
for mod in ('numpy', 'scipy', 'pandas'):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit('missing required packages: ' + ', '.join(missing))
EOF

if [ ! -d "loci/data" ] || [ ! -d "pairs" ]; then
  echo "error: expected loci/data/ and pairs/ next to this script." >&2
  echo "Run it from inside the laplacian_causality directory." >&2
  exit 1
fi

# One thread per worker: the transforms are already parallel across pairs, and
# letting BLAS spawn its own pool inside each worker oversubscribes the cores.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== 1/3  correctness: compare against the reference implementation ==="
"$PY" lap_nlogn.py --verify 60

echo
echo "=== 2/3  benchmark suite (1899 pairs, seeds: $SEEDS) ==="
"$PY" lap_nlogn.py --seeds "$SEEDS" --out "$OUT"

if [ "$SCALING" -eq 1 ]; then
  echo
  echo "=== 3/3  runtime scaling, single pair, kernel cached ==="
  for n in 1000 10000 100000 1000000; do
    "$PY" lap_nlogn.py --bench "$n"
  done
else
  echo
  echo "=== 3/3  scaling skipped (pass --scaling to include it) ==="
fi

ELAPSED=$((SECONDS - START))
echo
echo "done. per-pair scores in $OUT"
printf 'total execution time: %dh %02dm %02ds\n' \
  $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
