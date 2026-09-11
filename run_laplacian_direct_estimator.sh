#!/usr/bin/env bash
# Run the DIRECT (dense) Laplacian estimator on the full benchmark suite, as in
# paper_sp_letters/laplacian_causal_inference/short_version.tex, Table (b).
#
# This is the O(n^2) reference implementation: it forms the Gaussian kernel in
# blocks and evaluates the quadratic form directly.  It produces the four
# columns the paper reports from this repository -- Lap^std_raw, Lap^std_avg,
# Lap^unif and EdgeMass.  For the O(n log n) version of the same score see
# run_nlogn.sh and run_nlogn.txt.
#
#   ./run_laplacian_direct_estimator.sh              compute, then summarise
#   ./run_laplacian_direct_estimator.sh --summary    summarise existing results
#
# Runtime: about 10 minutes on a GPU, substantially longer on CPU, because the
# largest Tuebingen pair has n = 16382 and the kernel is dense.

set -euo pipefail
START=$SECONDS

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python}"
SUMMARY_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --summary) SUMMARY_ONLY=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

"$PY" - <<'EOF'
import sys
missing = [m for m in ('numpy', 'scipy', 'pandas', 'torch')
           if not __import__('importlib').util.find_spec(m)]
if missing:
    sys.exit('missing required packages: ' + ', '.join(missing))
import torch
print('device:', 'cuda' if torch.cuda.is_available() else 'cpu (this will be slow)')
EOF

if [ ! -d "loci/data" ] || [ ! -d "pairs" ]; then
  echo "error: expected loci/data/ and pairs/ next to this script." >&2
  exit 1
fi

if [ "$SUMMARY_ONLY" -eq 0 ]; then
  echo
  echo "=== 1/2  scoring 1899 pairs with the dense estimator ==="
  ( cd loci_dataset_benchmark && "$PY" run_benchmark.py )
else
  echo
  echo "=== 1/2  skipped (--summary): reusing loci_dataset_benchmark/results/per_pair.csv ==="
fi

echo
echo "=== 2/2  results, in the order of Table (b) of short_version.tex ==="
( cd loci_dataset_benchmark && "$PY" summarize_direct.py )

ELAPSED=$((SECONDS - START))
printf '\ntotal execution time: %dh %02dm %02ds\n' \
  $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
