# Graph Dirichlet Energy for Fitting-Free Bivariate Causal Inference

<img src="gdeci-small.png" alt="GDECI logo" align="right" width="160">

```
git clone --recurse-submodules https://github.com/povidanius/gdeci
```

Given a sample of two variables the method estimates which causes which.

Two implementations are provided and they agree to `1e-13`:

| | file | cost | needs |
|---|---|---|---|
| exact `O(n log n)` | `lap_nlogn.py` | 0.6 ms at `n=1e3`, 1.1 s at `n=1e6` | numpy, scipy |
| direct `O(n^2)` | `laplacian_causality.py` | 24 ms at `n=1e3`, 1.3 s at `n=1e4` | + torch |

The fast version exploits the fact that after the rank transform the kernel
matrix is Toeplitz, so every matrix-vector product is a convolution and the
`n x n` matrix is never formed.

## Docker build

The image carries both estimators, pinned dependencies (numpy, scipy, pandas,
torch, matplotlib, scikit-learn) and all the data the reproduction scripts
need: the 108 Tuebingen pairs from `pairs/`, plus the full LOCI benchmark
suite, which the build fetches from the `loci` submodule at the commit this
repository pins.

```bash
docker build --build-arg WITH_R_BASELINES=1 -t gdeci:r .  # + R baselines (QCCD, RESIT, CAM, GRCI), ~3.6 GB
```

| build arg | default | meaning |
|---|---|---|
| `TORCH_INDEX_URL` | `.../whl/cpu` | torch wheel index; point it at a `cuNNN` index for GPU support |
| `FETCH_DATA` | `1` | `0` skips the submodule fetch, leaving a code-only image.  The benchmark scripts then need `loci/data` bind-mounted (`-v "$PWD/loci:/workspace/loci:ro"`); `python laplacian_causality.py` still runs, since `pairs/` ships with the repository |
| `WITH_R_BASELINES` | `0` | `1` runs `baselines/setup_r_baselines.sh --with-grci` during the build, adding the R environment for QCCD, RESIT, CAM and GRCI (~2.5 GB) |
| `LOCI_SHA`, `QPE_CD_SHA` | the pinned commits | submodule revisions to fetch; keep them in step with `git ls-tree HEAD loci qpe_cd` |

## Docker run

```bash
docker run --rm -it gdeci
```

opens a shell inside the repository, at `/workspace`, with the benchmark data
already in place; `PYTHONPATH` is `/workspace`, so `lap_nlogn` and
`laplacian_causality` import from any working directory.  A command appended to
`docker run` replaces the shell, which is the form to use in scripts:

```bash
docker run --rm gdeci python lap_nlogn.py --verify 60       # O(n log n) vs. dense reference, 60 real pairs
docker run --rm gdeci ./run_laplacian_direct_estimator.sh --summary
docker run --rm gdeci python make_benchmark_figures.py --inventory   # which methods can be timed here
docker run --rm --gpus all gdeci:cu121 ./run_laplacian_direct_estimator.sh
```

The GPU build needs the NVIDIA Container Toolkit on the host; the dense
estimator picks up CUDA when torch sees it and reports the device it chose on
start-up.  With `gdeci:r` the tag has to be carried through to every
`docker run` that should see R -- plain `docker run gdeci` keeps starting the
R-less `gdeci:latest` and keeps reporting `no Rscript found`.

### Keeping the results

The copy of the repository inside a `--rm` container is discarded with it.  To
let everything written -- `results_nlogn.csv`,
`loci_dataset_benchmark/results/per_pair.csv`, the figures -- land in the
working tree, owned by you, mount the checkout over `/workspace` and run as
yourself:

```bash
docker run --rm -it -u "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" -v /workspace/loci -v /workspace/qpe_cd \
    gdeci
```

The two bare `-v` paths keep the submodules the image fetched visible; they can
be dropped once `git submodule update --init` has been run on the host.  Note
that this mode overwrites the per-pair scores committed here with the ones just
computed.  To keep single outputs without mounting the whole tree, mount one
directory and point the script at it:

```bash
mkdir -p out
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/out:/out" gdeci \
    ./run_nlogn.sh --quick --out /out/results_nlogn.csv
```

### Scoring your own data

```bash
docker run --rm -it -v "$PWD/mydata:/data" gdeci python
>>> import numpy as np
>>> from lap_nlogn import score
>>> x, y = np.loadtxt('/data/mypair.txt', unpack=True)
>>> score(x, y, m=0.07362, rng=np.random.default_rng(0))
```

`x` and `y` are plain 1-D arrays of equal length; `score` returns the forward
score, the reverse score and the inferred direction.

## Scripts

Every script runs from `/workspace` in the container, or from the repository
root with the dependencies installed.

| script | what it does |
|---|---|
| `run_laplacian_direct_estimator.sh` | direct `O(n^2)` estimator over all 1899 pairs, the three columns of Table (b) used in the paper; `--summary` reprints the table from the committed per-pair scores.  ~20 min on eight CPU cores, ~10 min on a GPU |
| `run_nlogn.sh` | fast `O(n log n)` estimator over the same pairs x 10 seeds, `--quick` for a single seed (~1 min), `--scaling` also times `n = 1e3 .. 1e6`.  Experimental, not used in the paper |
| `reproduce_main_table.sh` | rebuilds Table 1 (both panels) as LaTeX from the per-pair outputs; the test-compile is skipped in the container, which carries no `pdflatex` |
| `reproduce_figure1.sh` | Figure 1 -- timing, both accuracy panels and the Pareto panel.  The accuracy panels read the committed outputs, so they show every method; only the timing curves depend on what is installed |
| `reproduce_reci.sh` | the two RECI columns of Table (b) recomputed over the 1899 pairs (~4 min, needs `cdt`); `--summary` prints from saved scores, `--diagnose` shows why the CDT default lands below chance |
| `reproduce_orientation_flip_test.sh` | orientation-flip validation (~1 min), leaving report, per-pair CSVs and figures in `orientation_flip_test/` |

## Paper

Currently paper is work in progress.
