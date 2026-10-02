# syntax=docker/dockerfile:1
#
# Reproducible environment for the GDECI estimators.
#
#   docker build -t gdeci .
#   docker run --rm -it gdeci                              # a shell in the repository
#   docker run --rm gdeci ./run_laplacian_direct_estimator.sh
#
# The image carries the two GDECI estimators, the methods they are compared
# against, and the benchmark data: `pairs/` (the 108 Tuebingen pairs) comes from
# this repository, and the LOCI benchmark suite is fetched from the `loci`
# submodule during the build, so the reproduction scripts run without any
# further setup.
FROM python:3.11-slim

# CPU wheels by default (keeps the image at ~1.6 GB).  For a CUDA build, which
# the dense O(n^2) estimator benefits from, pass e.g.
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Set to 0 for a code-only image; the benchmark scripts then need `loci/data`
# bind-mounted from the host.
ARG FETCH_DATA=1

# The submodule commits recorded in this repository's index: `loci` supplies
# every benchmark family under loci/data/, `qpe_cd` only the optional QPE-k
# baseline.  Keep these in step with `git ls-tree HEAD loci qpe_cd`.
ARG LOCI_SHA=da61b8cdf95cbbc00fc42d351f89e64410f535f4
ARG QPE_CD_SHA=f23491c76c9aedeaafdd2ab3432c5130aeec9f06

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/workspace

# git fetches the submodules below; FreeSerif is the face
# accuracy_ranking_figure.py asks for, and matplotlib falls back to a sans one
# without it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git ca-certificates fonts-freefont-ttf \
 && rm -rf /var/lib/apt/lists/*

# numpy/scipy/pandas are enough for lap_nlogn.py; torch is additionally needed by
# laplacian_causality.py, matplotlib and scikit-learn by the figure scripts.
# numpy stays on 1.x, as loci/requirements.txt asks.
RUN pip install --no-cache-dir --extra-index-url "$TORCH_INDEX_URL" \
      numpy==1.26.4 \
      scipy==1.13.1 \
      pandas==2.2.3 \
      matplotlib==3.9.2 \
      scikit-learn==1.5.2 \
      torch==2.5.1

# The Python baselines that make_benchmark_figures.py times beside GDECI: cdt
# supplies RECI, RECI_poly and the two IGCI variants, and the rest is what
# loci/causa (LOCI itself) imports.
RUN pip install --no-cache-dir \
      cdt==0.6.0 \
      gin-config==0.5.0 \
      networkx==3.6.1 \
      seaborn==0.13.2 \
      statsmodels==0.15.0 \
      tqdm==4.70.1 \
      tueplots==0.2.4

# The dependency check in run_laplacian_direct_estimator.sh reaches for
# `importlib.util` without importing it, which only works in environments where
# some other package pulls that submodule in at interpreter start-up (as conda's
# site-packages do).  One site hook restores that assumption, so the script runs
# unchanged on a clean interpreter.
RUN echo 'import importlib.util' \
      > /usr/local/lib/python3.11/site-packages/aa_importlib_util.pth

WORKDIR /workspace

# The submodules, at their pinned commits, without their history.  Done before
# the source is copied in so that editing the source does not refetch 73 MB.
RUN if [ "$FETCH_DATA" = "1" ]; then \
      set -eu; \
      for spec in "loci https://github.com/aleximmer/loci $LOCI_SHA" \
                  "qpe_cd https://github.com/cyisk/qpe_cd $QPE_CD_SHA"; do \
        set -- $spec; \
        git init -q "$1" && git -C "$1" remote add origin "$2"; \
        git -C "$1" fetch -q --depth 1 origin "$3" || git -C "$1" fetch -q origin; \
        git -C "$1" checkout -q "$3"; \
        rm -rf "$1/.git"; \
      done; \
      test -d loci/data && test -d qpe_cd; \
    fi

# Only `baselines/` is needed by the R step below.  Copying it on its own keeps
# an edit anywhere else in the repository from re-provisioning the whole R
# environment on the next build.
COPY baselines /workspace/baselines

# The four R baselines (QCCD, RESIT, CAM and GRCI) are off by default: they need
# an R toolchain, which `baselines/setup_r_baselines.sh` provisions in a conda
# environment, and that costs some 2.5 GB and a long build.  Pass
# `--build-arg WITH_R_BASELINES=1` to run that script verbatim here, so the image
# gets the same environment it would build on a workstation.
#
# The one addition is the `r` channel: the script asks conda-forge for r-qrnn,
# which QCCD needs and which conda-forge does not carry, so a Miniforge install
# -- conda-forge and nothing else -- cannot solve the environment, while an
# Anaconda one resolves it through its own channels.
ARG WITH_R_BASELINES=0
RUN if [ "$WITH_R_BASELINES" = "1" ]; then \
      set -eu; \
      apt-get update \
        && apt-get install -y --no-install-recommends bzip2 curl \
        && rm -rf /var/lib/apt/lists/*; \
      curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh; \
      bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh; \
      /opt/conda/bin/conda config --system --append channels r; \
      PATH=/opt/conda/bin:$PATH bash baselines/setup_r_baselines.sh --with-grci; \
      /opt/conda/bin/conda clean -afy; \
    fi

# Where the script leaves Rscript.  make_benchmark_figures.py prefers this over
# anything on PATH, and ignores it when the path does not exist, so it is
# harmless in the default image.
ENV GDECI_RSCRIPT=/opt/conda/envs/gdeci-r/bin/Rscript

COPY . /workspace

# `docker run -it gdeci` opens a shell inside the repository; the banner recalls
# what can be run from there.  Only interactive shells read /etc/bash.bashrc, so
# `docker run gdeci ./run_nlogn.sh` and friends stay silent.
RUN printf '%s\n' \
      '' \
      'GDECI -- this is the repository, at /workspace.' \
      'The benchmark data is already in place: pairs/ and loci/data/.' \
      '' \
      '  ./run_nlogn.sh --quick                         fast estimator, 1899 pairs, ~1 min' \
      '  ./run_laplacian_direct_estimator.sh            direct estimator, Table (b), ~20 min on CPU' \
      '  ./run_laplacian_direct_estimator.sh --summary  the same table, from the committed scores' \
      '  python laplacian_causality.py                  direct estimator, Tuebingen-99 only' \
      '  ./reproduce_main_table.sh                      Table 1, from the per-pair outputs' \
      '  ./reproduce_figure1.sh                         Figure 1: timing, accuracy, Pareto' \
      '  bash reproduce_orientation_flip_test.sh        orientation-flip validation, ~1 min' \
      '  python lap_nlogn.py --verify 60                self-check against the dense reference' \
      '  python make_benchmark_figures.py --inventory   which methods can be timed here' \
      '' > /etc/gdeci-banner \
 && printf '%s\n' \
      '' \
      '# Opened by docker run -it: show what this image is for.' \
      "PS1='[gdeci \W] '" \
      'cat /etc/gdeci-banner' >> /etc/bash.bashrc

CMD ["bash"]
