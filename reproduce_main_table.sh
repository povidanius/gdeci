#!/usr/bin/env bash
# Regenerate Table 1 of the paper (both panels) from the per-pair outputs.
# PYTHON selects the Python executable.
set -euo pipefail

START_SECONDS=$SECONDS
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/loci_dataset_benchmark"
PY="${PYTHON:-python3}"
OUT="$BENCH/results/main_table.tex"
MACRO_CI="bootstrap"
SKIP_LATEX=0

usage() {
  cat <<'USAGE'
Usage: bash reproduce_main_table.sh [options]

  --out FILE       Where to write the table (default:
                   loci_dataset_benchmark/results/main_table.tex).
  --macro-ci WHICH bootstrap (default, as in the paper) or wilson, for the two
                   macro-average rows only.
  --skip-latex     Do not test-compile the generated table.
  -h, --help       Show this help.

The table is built from per-pair outputs that must already exist; the script
names the command for any that are missing.  When pdflatex is available the
generated table is test-compiled in a scratch directory, and the run fails if
LaTeX errors or if any cell would typeset as '?' or as a gap.
USAGE
}

while (($#)); do
  case "$1" in
    --out) (($# >= 2)) || { echo "error: --out needs a path" >&2; exit 2; }
           OUT="$2"; shift 2 ;;
    --macro-ci) (($# >= 2)) || { echo "error: --macro-ci needs a value" >&2; exit 2; }
           MACRO_CI="$2"; shift 2 ;;
    --skip-latex) SKIP_LATEX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

echo "=== 1/4  checking Python dependencies ==="
"$PY" - <<'PYEOF'
import importlib.util, sys
missing = [m for m in ('numpy', 'pandas') if importlib.util.find_spec(m) is None]
if missing:
    sys.exit('missing required packages: ' + ', '.join(missing))
print('numpy, pandas present')
PYEOF

echo "=== 2/4  checking per-pair inputs ==="
missing=0
check_input() {   # $1 = path relative to the repo, $2 = command that makes it
  if [[ -s "$HERE/$1" ]]; then
    printf '  %-46s present\n' "$1"
  else
    printf '  %-46s MISSING -- run: %s\n' "$1" "$2" >&2
    missing=1
  fi
}
check_input "loci/baseline_results/Tuebingen.tab" "git submodule update --init --recursive"
check_input "loci_dataset_benchmark/results/per_pair.csv" "./run_laplacian_direct_estimator.sh"
check_input "loci_dataset_benchmark/results/reci.csv" "python loci_dataset_benchmark/run_reci.py"
check_input "loci_dataset_benchmark/results/reci_poly.csv" "python loci_dataset_benchmark/run_reci_poly.py"
check_input "loci_dataset_benchmark/results/qpek.csv" "python loci_dataset_benchmark/run_qpek.py"
check_input "loci_dataset_benchmark/results_loci/loci.csv" "python loci_dataset_benchmark/run_loci.py"
if ((missing)); then
  echo "error: some per-pair outputs are missing; see the commands above" >&2
  exit 1
fi

echo "=== 3/4  building the table ==="
( cd "$BENCH" && "$PY" make_main_table.py --out "$OUT" --macro-ci "$MACRO_CI" )

echo "=== 4/4  validating the generated table ==="
if grep -q '?' "$OUT"; then
  echo "error: the generated table contains a '?'" >&2
  grep -n '?' "$OUT" >&2
  exit 1
fi
echo "  no '?' in the output"

if ((SKIP_LATEX)) || ! command -v pdflatex >/dev/null 2>&1; then
  echo "  test-compile skipped (pdflatex not found or --skip-latex)"
else
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  cp "$OUT" "$TMP/table.tex"
  cat > "$TMP/probe.tex" <<'TEXEOF'
\documentclass[twocolumn]{article}
\usepackage{booktabs,graphicx,xcolor}
\begin{document}
See Table~\ref{tab:main}.
\input{table}
\end{document}
TEXEOF
  ( cd "$TMP" && pdflatex -interaction=nonstopmode probe.tex >/dev/null 2>&1 \
              && pdflatex -interaction=nonstopmode probe.tex >/dev/null 2>&1 ) || true
  if [[ ! -s "$TMP/probe.pdf" ]]; then
    echo "error: the table does not compile" >&2
    grep -m5 -A3 '^!' "$TMP/probe.log" >&2 || true
    exit 1
  fi
  if grep -q '^!' "$TMP/probe.log"; then
    echo "error: LaTeX reported an error" >&2
    grep -m5 -A3 '^!' "$TMP/probe.log" >&2
    exit 1
  fi
  if grep -qi "Missing character" "$TMP/probe.log"; then
    echo "error: a glyph is missing from the font -- it would typeset as a gap" >&2
    grep -m5 -i "Missing character" "$TMP/probe.log" >&2
    exit 1
  fi
  echo "  test-compiled cleanly with pdflatex (needs \\usepackage{xcolor} only"
  echo "  if you add colour markup yourself; the table itself needs booktabs"
  echo "  and graphicx)"
fi

ELAPSED=$((SECONDS - START_SECONDS))
printf '\ndone in %dm %02ds; table is in %s\n' \
  $((ELAPSED / 60)) $((ELAPSED % 60)) "$OUT"
