#!/usr/bin/env bash
# Run the GDECI orientation-flip test from the repository root.
# PYTHON selects the Python executable; JOBS sets the default worker count.
set -euo pipefail

START_SECONDS=$SECONDS
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
JOBS_VALUE="${JOBS:-1}"
OUTPUT_DIR="$HERE/orientation_flip_test"
REUSE_RESULTS=0

usage() {
  cat <<'USAGE'
Usage: bash reproduce_orientation_flip_test.sh [options]

  --jobs N         Positive worker count for fresh scoring (default: JOBS or 1).
  --reuse-results  Regenerate summary, report and figures from the stored
                   per-pair CSV without loading raw benchmark pairs.
  --output-dir DIR Output directory (default: orientation_flip_test beside
                   this script). Relative paths use the caller's directory.
  -h, --help       Show this help.

Place this wrapper beside run_orientation_flip_test.py in the repository root.
PYTHON may select a Python executable, for example PYTHON=/path/to/venv/bin/python.
USAGE
}

while (($#)); do
  case "$1" in
    --jobs)
      if (($# < 2)); then
        echo "error: --jobs requires a positive integer" >&2
        exit 2
      fi
      JOBS_VALUE="$2"
      shift 2
      ;;
    --output-dir)
      if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
        echo "error: --output-dir requires a directory" >&2
        exit 2
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --reuse-results)
      REUSE_RESULTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$JOBS_VALUE" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --jobs (or JOBS) must be a positive integer" >&2
  exit 2
fi
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$PWD/$OUTPUT_DIR" ;;
esac
cd "$HERE"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: Python executable not found: $PY" >&2
  exit 1
fi
for source_file in run_orientation_flip_test.py run_smoothness_experiment.py \
                   lap_nlogn.py loci_dataset_benchmark/datasets.py; do
  if [[ ! -f "$HERE/$source_file" ]]; then
    echo "error: missing $HERE/$source_file; place the scripts in the GDECI repository root" >&2
    exit 1
  fi
done

# Avoid numerical-library thread oversubscription when using multiple workers.
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-gdeci}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
mkdir -p "$MPLCONFIGDIR"

echo "=== 1/5  checking Python dependencies ==="
"$PY" - <<'PY'
import importlib.util
import sys

required = ("numpy", "scipy", "pandas", "matplotlib")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("missing required packages: " + ", ".join(missing))
PY

if ((REUSE_RESULTS)); then
  echo "=== 2/5  checking cached per-pair orientation scores ==="
  CSV_PATH="$OUTPUT_DIR/results/orientation_flip_per_pair.csv"
  if [[ ! -s "$CSV_PATH" ]]; then
    echo "error: --reuse-results requires a non-empty $CSV_PATH" >&2
    exit 1
  fi
else
  echo "=== 2/5  checking raw benchmark data ==="
  if [[ ! -d loci/data || ! -d pairs ]]; then
    git submodule update --init --recursive
  else
    echo "benchmark-data directories are already present"
  fi
  if [[ ! -d loci/data || ! -d pairs ]]; then
    echo "error: expected loci/data and pairs after submodule initialization" >&2
    exit 1
  fi
fi

echo "=== 3/5  checking the orientation identities on one synthetic pair ==="
"$PY" - <<'PY'
import numpy as np

from lap_nlogn import kernel_for, rank_grid
from run_orientation_flip_test import CONFIGS, FROZEN_M

rng = np.random.default_rng(20260922)
n = 1000
x = rng.standard_normal(n)
y = np.tanh(2.0 * x) + 0.3 * rng.standard_normal(n)
if len(np.unique(x)) != n or len(np.unique(y)) != n:
    raise SystemExit("error: the synthetic self-test sample must be tie-free")

# (i) negation reverses the rank grid exactly, in the absence of ties.
for sample in (x, y):
    np.testing.assert_allclose(rank_grid(-sample), 1.0 - rank_grid(sample),
                               rtol=0, atol=1e-12)

# (ii) the published energy on L = D - W is orientation-invariant; E_sym is not.
kernel = kernel_for(n, FROZEN_M)
symmetric, published = {}, {}
for config, (sign_x, sign_y) in CONFIGS.items():
    u = rank_grid(sign_x * x)
    v = rank_grid(sign_y * y)
    signals = np.vstack([v[np.argsort(u)], u[np.argsort(v)]])
    published[config] = kernel.energy(signals)
    symmetric[config] = kernel.energy_sym(signals)

for config in ("flipX", "flipY", "flipBoth"):
    np.testing.assert_allclose(published[config], published["orig"],
                               rtol=1e-9, atol=1e-9)

# Forward energy moves with Y only, reverse energy with X only.
np.testing.assert_allclose(symmetric["flipX"][0], symmetric["orig"][0],
                           rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(symmetric["flipY"][1], symmetric["orig"][1],
                           rtol=1e-12, atol=1e-12)
moved_forward = abs(symmetric["flipY"][0] - symmetric["orig"][0])
moved_reverse = abs(symmetric["flipX"][1] - symmetric["orig"][1])
if moved_forward <= 1e-6 or moved_reverse <= 1e-6:
    raise SystemExit("error: E_sym did not move under negation; expected it to")
print(f"validated: rank reversal exact; E = v^T (D - W) v invariant to <=1e-9; "
      f"E_sym moved by {moved_forward:.4f} (forward, Y negated) and "
      f"{moved_reverse:.4f} (reverse, X negated)")
PY

RUN_ARGS=(--jobs "$JOBS_VALUE" --output-dir "$OUTPUT_DIR")
if ((REUSE_RESULTS)); then
  RUN_ARGS+=(--reuse-results)
  echo "=== 4/5  regenerating artifacts from the cached per-pair scores ==="
else
  echo "=== 4/5  scoring all benchmark pairs in four orientations ==="
fi
"$PY" "$HERE/run_orientation_flip_test.py" "${RUN_ARGS[@]}"

echo "=== 5/5  validating reproduced artifacts ==="
EXPECTED=(
  ORIENTATION_FLIP_REPORT.md
  figures/orientation_flip_rates.pdf
  figures/orientation_flip_rates.png
  figures/orientation_accuracy.pdf
  figures/orientation_accuracy.png
  results/orientation_flip_per_pair.csv
  results/orientation_flip_summary.json
)
for artifact in "${EXPECTED[@]}"; do
  if [[ ! -s "$OUTPUT_DIR/$artifact" ]]; then
    echo "error: missing or empty artifact: $OUTPUT_DIR/$artifact" >&2
    exit 1
  fi
done

"$PY" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import run_orientation_flip_test as experiment

output = Path(sys.argv[1])
rows = pd.read_csv(output / "results/orientation_flip_per_pair.csv")
with (output / "results/orientation_flip_summary.json").open(encoding="utf-8") as handle:
    summary = json.load(handle)


def require(condition, message):
    if not condition:
        raise SystemExit("error: " + message)


require(len(rows) == 1899, f"expected 1899 CSV rows, found {len(rows)}")
require(rows["benchmark"].nunique() == 13, "expected 13 benchmark suites")
require(not rows.duplicated(["benchmark", "pair_id"]).any(), "duplicate benchmark/pair_id rows")
require(summary["design"]["pairs"] == len(rows), "JSON pair count is inconsistent")
require(summary["design"]["datasets"] == 13, "JSON dataset count is inconsistent")
np.testing.assert_allclose(summary["design"]["frozen_multiplier"], experiment.FROZEN_M,
                           rtol=0, atol=1e-14)

columns = []
for label, margin_col, scale_col in experiment.margin_columns():
    columns += [margin_col, scale_col, f"dec_{label}"]
require(set(columns).issubset(rows.columns), "CSV is missing margin or decision columns")
require(np.isfinite(rows[[c for c in columns if not c.startswith("dec_")]]
                    .to_numpy(dtype=float)).all(), "nonfinite margins in the CSV")

checks = summary["sanity_checks"]
require(checks["all_passed"], "the stored summary records a failed sanity check")
require(round(checks["published_macro12"]["value"], 3) == experiment.PUBLISHED_MACRO12,
        "the summary does not reproduce the published macro-12 accuracy")
require(round(checks["published_tuebingen"]["value"], 3) == experiment.PUBLISHED_TUEBINGEN,
        "the summary does not reproduce the published Tuebingen accuracy")
require(checks["gdeci_invariance"]["passed"],
        "the published energy is not orientation-invariant on tie-free pairs")
require(checks["normalized_variant_moves"]["passed"],
        "the symmetric-normalized energy did not move under negation")

for group in ("macro12", "pooled_1899"):
    for score in experiment.SCORES:
        rates = summary[group]["flip_rates"][score]
        require(set(experiment.FLIPS).issubset(rates),
                f"missing {score} flip rates in {group}")
        for flip in experiment.FLIPS:
            require(0.0 <= rates[flip]["rate"] <= 1.0,
                    f"implausible {score} flip rate in {group}")
for label, _, _ in experiment.margin_columns():
    require(label in summary["macro12"]["accuracy"], f"missing macro accuracy for {label}")
    require(label in summary["tuebingen"]["weighted"]["accuracy"],
            f"missing weighted Tuebingen accuracy for {label}")

print("validated: 1,899 pairs, 13 suites, reproduced published accuracies, "
      "orientation invariance of the published energy on L = D - W, and 7 "
      "non-empty artifacts")
PY

ELAPSED=$((SECONDS - START_SECONDS))
printf 'done in %dh %02dm %02ds; artifacts are in %s\n' \
  $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) $((ELAPSED % 60)) "$OUTPUT_DIR"
