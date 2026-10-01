#!/usr/bin/env Rscript
# Time one bivariate causal-direction baseline, inside R, on a pair given as CSV.
#
#   Rscript time_baseline.R METHOD pair.csv REPS
#
# prints a single line `seconds <median elapsed over REPS timed calls>`, plus a
# `decision <0|1|NA>` line, where 1 means the first column was called the cause.
#
# METHOD is QCCD, RESIT or CAM.  Each is called exactly as the authors' own
# benchmark wrappers call it, in `bQCD/R/baselines.R` and `bQCD/R/bqcd.R`:
#
#   QCCD    QCCD(pair)                                    (Tagasovska et al. 2020)
#   RESIT   ICML(pair, model = train_gp,                  (Peters et al. 2014, the
#                 indtest = indtestHsic, output = FALSE)   code vendored in bQCD)
#   CAM     CAM(pair, scoreName = "SEMGAM")               (Buehlmann et al. 2014)
#   GRCI    causal_direc(x, y)                            (Strobl & Lasko 2023)
#
# GRCI is used as a bivariate direction rule, which is what the table reports.
# Its `causal_direc` needs only five files of the package (spline_regressionR,
# normalize01, normalizeData, PartialOut, causal_direc) and three CRAN packages
# (RANN for nn2, Rfast for spdinv, DirichletReg for rdirichlet), so those files
# are sourced directly rather than installing the whole package, whose Depends
# list covers the root-causal-inference pipeline this call never enters.
#
# R's own start-up is not part of the reported time: the method is called once
# untimed to load and warm everything, and only the calls after that are timed.
# The repeats run inside this one process, so no start-up is charged to them.
suppressPackageStartupMessages({

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: time_baseline.R METHOD pair.csv [REPS]")
method <- args[1]
path <- args[2]
reps <- if (length(args) >= 3) as.integer(args[3]) else 3L

self <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))
HERE <- dirname(normalizePath(self))
BQCD <- file.path(HERE, "bQCD")

pair <- as.matrix(read.csv(path))          # two columns, the cause first
colnames(pair) <- c("V1", "V2")

fun <- switch(
  method,
  QCCD = {
    old <- setwd(BQCD); on.exit(setwd(old), add = TRUE)
    source(file.path(BQCD, "R", "bqcd.R"))
    function(p) QCCD(p)
  },
  RESIT = {
    source(file.path(BQCD, "R/baselines/Slope/resit/code/startups/startupICML.R"),
           chdir = TRUE)
    function(p) ICML(p, model = train_gp, indtest = indtestHsic, output = FALSE)
  },
  CAM = {
    library(CAM)
    function(p) CAM(p, scoreName = "SEMGAM")
  },
  GRCI = {
    library(RANN); library(DirichletReg)
    # Rfast supplies exactly one primitive on this path, spdinv (the inverse of
    # a symmetric positive-definite matrix).  Where Rfast is unavailable -- it
    # has no build for this R version and its C++ does not compile against the
    # conda toolchain here -- the identical LAPACK operation from base R stands
    # in.  Nothing else about the method changes, and the substitution is made
    # here rather than in the vendored code.
    if (requireNamespace("Rfast", quietly = TRUE)) {
      spdinv <- Rfast::spdinv
    } else {
      spdinv <- function(m) chol2inv(chol(m))
      cat("note: Rfast absent, using chol2inv(chol(.)) for spdinv\n")
    }
    for (f in c("normalize01.R", "normalizeData.R", "spline_regressionR.R",
                "PartialOut.R", "causal_direc.R"))
      source(file.path(HERE, "GRCI", "R", f))
    function(p) causal_direc(p[, 1], p[, 2])
  },
  stop(paste("unknown method:", method))
)

run <- function() invisible(capture.output(suppressWarnings(res <<- fun(pair))))

res <- NULL
run()                                       # warm-up: not timed
times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- proc.time()[["elapsed"]]
  run()
  times[i] <- proc.time()[["elapsed"]] - t0
}

# Each method reports its direction differently; normalize to 1 = first column
# is the cause, matching the wrappers in bQCD/R/baselines.R.
decision <- tryCatch({
  if (method == "QCCD") as.numeric(res$cd)
  else if (method == "RESIT") if (identical(res$Cd, "->")) 1 else if (identical(res$Cd, "<-")) 0 else NA
  else if (method == "CAM") if (res$Adj[3] == 1) 1 else if (res$Adj[2] == 1) 0 else NA
  else if (method == "GRCI") if (res == 1) 1 else 0
  else NA
}, error = function(e) NA)

})
cat(sprintf("seconds %.9f\n", median(times)))
cat(sprintf("decision %s\n", ifelse(is.na(decision), "NA", decision)))
