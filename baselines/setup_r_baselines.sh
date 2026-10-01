#!/usr/bin/env bash
# Fetch and install the R baselines of the paper's table, so they can be timed.
#
# QCCD, RESIT and CAM have no Python implementation: the table's numbers for
# them are the LOCI authors' released per-pair outputs.  Their authors' code is
# R, so this script builds a self-contained R environment for it and leaves the
# rest of the machine alone -- everything lands in one conda environment
# (default: gdeci-r) and in clones under this directory, and
#
#     conda env remove -n gdeci-r && rm -rf baselines/bQCD baselines/GRCI
#
# undoes all of it.
#
# What is installed, and why:
#
#   bQCD (github.com/tagas/bQCD)   Tagasovska et al. 2020.  Provides QCCD itself
#                                  (R/bqcd.R) and, vendored inside it, Peters'
#                                  RESIT code (R/baselines/Slope/resit) and the
#                                  wrappers the authors benchmarked with.
#   CAM 1.0 (CRAN archive)         Buehlmann et al. 2014; archived, so it is
#                                  built from the source tarball.
#   gptk 1.08 (CRAN archive)       the GP regression RESIT fits; also archived.
#
# One compatibility patch is applied, and only one: gptk's `gptk-internal.R`
# tests `class(invCh) == "try-error"`, which errors on R >= 4.2 because `class()`
# of a matrix has length 2 ("the condition has length > 1").  It is replaced by
# the equivalent `inherits(invCh, "try-error")`.  Nothing else in any vendored
# package is modified.
#
# GRCI (github.com/ericstrobl/GRCI) is cloned as well.  Its full dependency set
# (pcalg, Rfast, hetGP, xgboost, treeshap, ...) is not needed: the timed path in
# time_baseline.R sources only causal_direc.R and its helpers, which use RANN and
# DirichletReg, plus Rfast::spdinv, for which base R's chol2inv(chol(.)) stands
# in.  --with-grci installs just those two (pcalg is not on conda-forge anyway).
#
#   ./baselines/setup_r_baselines.sh              QCCD, RESIT and CAM
#   ./baselines/setup_r_baselines.sh --with-grci  also GRCI (RANN, DirichletReg)
#   ENV=my-env ./baselines/setup_r_baselines.sh   into a different conda env
#
# Then `./run_timing_benchmark.py --inventory` should report qccd, resit and cam
# as timeable.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

ENV="${ENV:-gdeci-r}"
WITH_GRCI=0
[ "${1:-}" = "--with-grci" ] && WITH_GRCI=1

command -v conda >/dev/null || { echo "conda not found" >&2; exit 1; }

echo "=== 1/4  conda environment '$ENV' with R and the CRAN dependencies ==="
if conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  echo "    exists, reusing it"
else
  conda create -y -n "$ENV" -c conda-forge \
    r-base r-mgcv r-glmnet r-mboost r-codetools r-quantreg r-kernlab r-mass \
    r-rcpp compilers
fi
conda install -y -n "$ENV" -c conda-forge \
  r-rvinecopulib r-statmod r-quantregforest r-qrnn r-randomforest r-fields \
  r-igraph

PREFIX="$(conda run -n "$ENV" printenv CONDA_PREFIX)"
export PATH="$PREFIX/bin:$PATH"

echo
echo "=== 2/4  the authors' repositories ==="
[ -d bQCD ] || git clone --depth 1 https://github.com/tagas/bQCD.git
[ -d GRCI ] || git clone --depth 1 https://github.com/ericstrobl/GRCI.git

echo
echo "=== 3/4  archived CRAN packages: gptk (patched for R >= 4.2) and CAM ==="
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
( cd "$WORK"
  curl -sO https://cran.r-project.org/src/contrib/Archive/gptk/gptk_1.08.tar.gz
  curl -sO https://cran.r-project.org/src/contrib/Archive/CAM/CAM_1.0.tar.gz
  tar xzf gptk_1.08.tar.gz
  sed -i 's/if ( class(invCh) == "try-error" )/if ( inherits(invCh, "try-error") )/' \
    gptk/R/gptk-internal.R
  grep -q 'inherits(invCh, "try-error")' gptk/R/gptk-internal.R \
    || { echo "gptk patch did not apply" >&2; exit 1; }
  R CMD INSTALL gptk
  R CMD INSTALL CAM_1.0.tar.gz )

if [ "$WITH_GRCI" -eq 1 ]; then
  echo
  echo "=== GRCI dependencies used by the timed path: RANN, DirichletReg ==="
  conda install -y -n "$ENV" -c conda-forge r-rann r-dirichletreg || \
    echo "    GRCI dependencies could not be installed; GRCI stays untimed"
fi

echo
echo "=== 4/4  check ==="
Rscript -e 'for (p in c("rvinecopulib","statmod","quantregForest","qrnn","gptk",
                        "kernlab","mgcv","CAM","glmnet","mboost","RANN",
                        "DirichletReg"))
              cat(sprintf("%-16s %s\n", p, requireNamespace(p, quietly=TRUE)))'
echo
echo "Rscript: $(command -v Rscript)"
echo "Point the benchmark at it with:  export GDECI_RSCRIPT=$(command -v Rscript)"
