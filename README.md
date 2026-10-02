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

## Install

```bash
pip install numpy scipy pandas          # enough for lap_nlogn.py
pip install torch                       # additionally for laplacian_causality.py
```

## Use it on your own data

```python
import numpy as np
from lap_nlogn import score

# --- a pair with a known ground truth: X causes Y ---
rng = np.random.default_rng(0)
n = 2000
x = rng.uniform(-1, 1, n)                                   # cause
y = np.cos(3 * np.pi * x) + 0.2 * rng.standard_normal(n)    # effect

fwd, rev, x_causes_y = score(x, y, m=0.07362, rng=rng)

print(f'score(X->Y) = {fwd:.6g}')
print(f'score(Y->X) = {rev:.6g}')
print('inferred    :', 'X -> Y' if x_causes_y else 'Y -> X')
print(f'margin      : {abs(fwd - rev) / (abs(fwd) + abs(rev)):.3f}')
```
`x` and `y` are plain 1-D arrays of equal length.  


## Reproducing the papers results for GDECI estimators

```bash
./run_nlogn.sh                       # fast estimator, all 1899 pairs, ~1 min (experimental, not used in the paper)
./run_laplacian_direct_estimator.sh  # direct estimator, Table (b), ~5 min (used in the paper)
```


## Docker

A `Dockerfile` is included, so neither Python nor the benchmark data has to be
set up by hand.  The image carries both estimators, pinned dependencies
(numpy, scipy, pandas, torch, matplotlib, scikit-learn) and all the data the
reproduction scripts need: the 108 Tuebingen pairs from `pairs/`, plus the full
LOCI benchmark suite, which the build fetches from the `loci` submodule at the
commit this repository pins.  A plain `git clone` is therefore enough --
`--recurse-submodules` is not required.

### Build

```bash
docker build -t gdeci .
```

About 1.6 GB and a few minutes, most of it the torch CPU wheel and the 73 MB
of LOCI data.

### Run

```bash
docker run --rm -it gdeci
```

opens a shell inside the repository, at `/workspace`, with the benchmark data
already in place.  Every script of the sections above runs from there:

```
[gdeci workspace] ./run_nlogn.sh --quick                         # fast estimator, 1899 pairs, ~1 min
[gdeci workspace] ./run_laplacian_direct_estimator.sh            # direct estimator, Table (b)
[gdeci workspace] ./run_laplacian_direct_estimator.sh --summary  # the same table, from the committed scores
[gdeci workspace] python laplacian_causality.py                  # direct estimator, Tuebingen-99 only
[gdeci workspace] ./reproduce_main_table.sh                      # Table 1, from the per-pair outputs
[gdeci workspace] python lap_nlogn.py --verify 60                # self-check against the dense reference
[gdeci workspace] ./reproduce_figure1.sh                         # Figure 1: timing, accuracy, Pareto
[gdeci workspace] bash reproduce_orientation_flip_test.sh        # orientation-flip validation, ~1 min
[gdeci workspace] python make_benchmark_figures.py --inventory   # which methods can be timed here
```

`--verify` is the correctness check behind the `1e-13` agreement quoted above:
the `O(n log n)` score against the dense `O(n^2)` reference on 60 real benchmark
pairs.  A command appended to `docker run` replaces the shell, which is the form
to use in scripts:

```bash
docker run --rm gdeci python lap_nlogn.py --verify 60
docker run --rm gdeci ./run_laplacian_direct_estimator.sh --summary
```

`run_laplacian_direct_estimator.sh` forms a dense kernel per pair, so it is the
slow one: 20 min for all 1899 pairs on eight CPU cores (an i7-4790K), and the
recomputed table matches the per-pair scores committed here.  `--summary`
reprints that table from them without recomputing, and the GPU build below is
the other quick way through.

### Baselines and the timing figure

The image also carries the competing methods that have a runnable
implementation, so `make_benchmark_figures.py` has something to time beside
GDECI: `cdt` for RECI, RECI_poly and the two IGCI variants, the `qpe_cd`
submodule for QPE-k, and LOCI's own `causa` package with `gin-config`,
`statsmodels`, `seaborn` and `tueplots`.

```bash
docker run --rm gdeci python make_benchmark_figures.py --inventory
```

reports what the container can time, method by method.  Eight of the twelve are
ready in the default image; the remaining four -- QCCD, RESIT, CAM and GRCI --
are R packages, and `baselines/setup_r_baselines.sh` is run during the build
only when asked:

```bash
docker build --build-arg WITH_R_BASELINES=1 -t gdeci:r .
docker run --rm gdeci:r python make_benchmark_figures.py --inventory
```

`-t gdeci:r` builds a *second* image beside the default one, so the tag has to
be carried through to every `docker run` that should see R: plain `docker run
gdeci` keeps starting the R-less `gdeci:latest` and keeps reporting `no Rscript
found`.  Build it as `-t gdeci` instead to make the R image the only one.

That build installs Miniforge and the script's conda environment, R toolchain
included, and takes the image to 3.6 GB; `GDECI_RSCRIPT` already points at
the `Rscript` it produces, which is where the figure script looks first.

Figure 1 of the paper -- the timing panel, both accuracy panels and the Pareto
panel -- is then `./reproduce_figure1.sh`.  The accuracy panels read the
committed per-pair outputs, so they show every method either way, R baselines
included; only the timing curves depend on what is installed.  To keep the
figure, its caption and the timing CSV, either mount the checkout as in the
previous section or send the script's own output somewhere mounted:

```bash
mkdir -p out
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/out:/out" gdeci \
    python make_benchmark_figures.py --include-accuracy-ranking --include-pareto \
                                     --budget 120 --csv /out/results_timing.csv --out /out/fig.pdf
```

Swap `gdeci` for `gdeci:r` to put the four R methods into panel (a) as well.
`cdt` being present also makes `./reproduce_reci.sh` run in full -- both RECI
columns of Table (b), recomputed over the 1899 pairs in about 4 minutes -- and
`python accuracy_ranking_figure.py` writes its two panels into `figures/`.

`reproduce_orientation_flip_test.sh` runs too, in about a minute, and leaves its
report, per-pair CSVs and figures in `orientation_flip_test/`.
`reproduce_main_table.sh` skips its LaTeX test-compile, since the image carries
no `pdflatex`.

### Scoring your own data

`PYTHONPATH` is `/workspace`, so `lap_nlogn` and `laplacian_causality` import
from any working directory:

```bash
docker run --rm -it -v "$PWD/mydata:/data" gdeci python
>>> import numpy as np
>>> from lap_nlogn import score
>>> x, y = np.loadtxt('/data/mypair.txt', unpack=True)
>>> score(x, y, m=0.07362, rng=np.random.default_rng(0))
```

### Keeping the results

The copy of the repository inside a `--rm` container is discarded with it.  To
edit and score against the checkout on the host instead, mount it over
`/workspace` and run as yourself, so that everything written -- `results_nlogn.csv`,
`loci_dataset_benchmark/results/per_pair.csv`, the figures -- lands in the
working tree, owned by you:

```bash
docker run --rm -it -u "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" -v /workspace/loci -v /workspace/qpe_cd \
    gdeci
```

The two bare `-v` paths keep the submodules the image fetched visible, since the
mount would otherwise expose the empty `loci/` and `qpe_cd/` directories of a
checkout cloned without `--recurse-submodules`; they can be dropped once
`git submodule update --init` has been run on the host.  Note that this mode
overwrites the per-pair scores committed here with the ones just computed.

To keep a single file without mounting the whole tree, mount one directory and
point the script at it:

```bash
mkdir -p out
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/out:/out" gdeci \
    ./run_nlogn.sh --quick --out /out/results_nlogn.csv
```

### GPU

The dense estimator picks up CUDA when torch sees it, and reports the device it
chose on start-up.  Build against a CUDA wheel and pass the GPU in (this needs
the NVIDIA Container Toolkit on the host):

```bash
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t gdeci:cu121 .
docker run --rm --gpus all gdeci:cu121 ./run_laplacian_direct_estimator.sh
```

### Build arguments

| arg | default | meaning |
|---|---|---|
| `TORCH_INDEX_URL` | `.../whl/cpu` | torch wheel index; point it at a `cuNNN` index for GPU support |
| `FETCH_DATA` | `1` | `0` skips the submodule fetch, leaving a code-only image.  The benchmark scripts then need `loci/data` bind-mounted (`-v "$PWD/loci:/workspace/loci:ro"`); `python laplacian_causality.py` still runs, since `pairs/` ships with the repository |
| `WITH_R_BASELINES` | `0` | `1` runs `baselines/setup_r_baselines.sh --with-grci` during the build, adding the R environment for QCCD, RESIT, CAM and GRCI (~2.5 GB) |
| `LOCI_SHA`, `QPE_CD_SHA` | the pinned commits | submodule revisions to fetch; keep them in step with `git ls-tree HEAD loci qpe_cd` |


## Paper
Currently paper is work in progress.

