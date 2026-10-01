#!/usr/bin/env bash
# Reproduce the RECI baseline of Table (b) in two columns, on the same 1899 pairs:
#
#   RECI (CDT default)   cdt.causality.pairwise.RECI(degree=3) exactly as shipped
#                        -- the column currently in the table, which scores 0.180
#                        on AN, 0.220 on LS, 0.130 on MN-U and 0.428 macro, i.e.
#                        far below chance on suites every other method solves.
#   RECI_poly       the same toolbox class, decision rule and min-max
#                        rescaling, with the function class RECI's paper
#                        actually specifies: the POLY design sum_i a_i x^i,
#                        degree taken from the paper's family [1,9] and chosen
#                        per pair by cross-validated fit error.  No causal label
#                        enters the choice.
#
# The below-chance column is not a property of RECI.  CDT's b_fit_score builds
# [1, x, x^2, x^3] and then zeroes the linear and quadratic columns, so it fits
# y ~ a + b x^3; both regression errors then collapse onto the variance of the
# min-max scaled target and the score becomes an inverted marginal-variance rule.
# `--diagnose` shows this: the CDT score correlates 0.87 with
# var(minmax x) - var(minmax y) on AN, and agrees with this repository's own
# "smaller min-max variance is the cause" control on only 16% of all pairs --
# that control scores 0.84/0.83/0.97 where RECI-as-run scores 0.18/0.22/0.13.
#
#   ./reproduce_reci.sh             compute both columns, then print the table
#   ./reproduce_reci.sh --summary   print the table from the saved per-pair scores
#   ./reproduce_reci.sh --diagnose  why the toolbox default lands below chance
#   ./reproduce_reci.sh --sweep     every fixed degree k = 1..9 beside the
#                                   cross-validated rule (~6 min)
#
# Runtime: about 2 minutes on CPU, no GPU used.  Needs the Causal Discovery
# Toolbox (pip install cdt), scikit-learn, numpy, scipy and pandas.

set -euo pipefail
START=$SECONDS

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python}"
SUMMARY_ONLY=0
DIAGNOSE=0
SWEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --summary) SUMMARY_ONLY=1 ;;
    --diagnose) DIAGNOSE=1 ;;
    --sweep) SWEEP=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

"$PY" - <<'EOF'
import sys
from importlib import util
missing = [m for m in ('numpy', 'scipy', 'pandas', 'sklearn', 'cdt')
           if not util.find_spec(m)]
if missing:
    sys.exit('missing required packages: ' + ', '.join(missing))
EOF

if [ ! -d "loci/data" ]; then
  echo "error: expected loci/data/ next to this script." >&2
  exit 1
fi

if [ "$DIAGNOSE" -eq 1 ]; then
  echo
  echo "=== why the toolbox default lands below chance ==="
  ( cd loci_dataset_benchmark && "$PY" run_reci_poly.py --diagnose )
  exit 0
fi

if [ "$SWEEP" -eq 1 ]; then
  echo
  echo "=== every fixed degree beside the cross-validated rule ==="
  ( cd loci_dataset_benchmark && "$PY" run_reci_poly.py --sweep )
  exit 0
fi

if [ "$SUMMARY_ONLY" -eq 0 ]; then
  echo
  echo "=== 1/3  scoring 1899 pairs with RECI, Causal Discovery Toolbox default ==="
  ( cd loci_dataset_benchmark && "$PY" run_reci.py )

  echo
  echo "=== 2/3  scoring the same 1899 pairs with RECI_poly ==="
  ( cd loci_dataset_benchmark && "$PY" run_reci_poly.py )
else
  echo
  echo "=== 1-2/3  skipped: reusing loci_dataset_benchmark/results/reci{,_opt}.csv ==="
fi

echo
echo "=== 3/3  RECI, as shipped and as RECI_poly, in the order of Table (b) ==="
( cd loci_dataset_benchmark && "$PY" summarize_reci.py )

ELAPSED=$((SECONDS - START))
printf '\ntotal execution time: %dh %02dm %02ds\n' \
  $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
