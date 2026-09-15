# Graph Dirichlet Energy for Fitting-Free Bivariate Causal Inference

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
`x` and `y` are plain 1-D arrays of equal length.  This call is the
`Lap^unif` estimator of the [results table](#results) at its frozen bandwidth
multiplier `m = 0.07362`.



## Results

The main estimator is **`Lap^unif`**: the Dirichlet energy over the *uniform*
(rank-transformed) marginals, normalized by edge mass, at the frozen bandwidth
multiplier `m = 0.07362`.  This is what `score(x, y, m=0.07362)` above computes,
and it is the column to read.  The other three columns are ablations kept for
reference: `Lap^std_raw` / `Lap^std_avg` use standardized instead of uniform
marginals (`m = 0.05535`, raw vs. average ranks), and `EdgeMass` decides on the
edge mass alone, without the energy.

Accuracy per benchmark, in the order of Table (b) of the paper
(1899 pairs, direct estimator, ~5 min):

| benchmark | #pairs | **Lap^unif** | Lap^std_raw | Lap^std_avg | EdgeMass |
|---|---:|---:|---:|---:|---:|
| AN | 100 | **1.000** | 1.000 | 1.000 | 0.940 |
| AN-s | 100 | **0.810** | 1.000 | 1.000 | 0.960 |
| LS | 100 | **1.000** | 1.000 | 1.000 | 0.920 |
| LS-s | 100 | **0.920** | 0.690 | 0.430 | 0.950 |
| MN-U | 100 | **0.990** | 0.910 | 0.710 | 0.990 |
| SIM | 100 | **0.660** | 0.590 | 0.720 | 0.350 |
| SIM-c | 100 | **0.660** | 0.590 | 0.620 | 0.360 |
| SIM-G | 100 | **0.590** | 0.830 | 0.760 | 0.850 |
| SIM-ln | 100 | **0.790** | 0.830 | 0.850 | 0.510 |
| Cha | 300 | **0.517** | 0.470 | 0.500 | 0.583 |
| Multi | 300 | **0.570** | 0.387 | 0.330 | 0.683 |
| Net | 300 | **0.800** | 0.767 | 0.863 | 0.557 |
| Tuebingen | 99 | **0.667** | 0.535 | 0.758 | 0.263 |
| Tuebingen, weighted | 99 | **0.656** | 0.512 | 0.673 | 0.345 |
| **macro mean, 12 non-Tuebingen** | **1800** | **0.776** | 0.755 | 0.732 | 0.721 |

`Lap^unif` has the best macro mean, 0.776 [0.76, 0.79] (Wilson 95%), against
0.755 [0.73, 0.77], 0.732 [0.71, 0.75] and 0.721 [0.70, 0.74] for the ablations.
Tuebingen is LOCI's 99-pair subset; the weighted row uses the pairmeta weights
(effective sample size 56.8).  Per-benchmark intervals are printed by the run
script and recorded in `results.txt`.

## Reproducing the papers results for GDECI estimators

```bash
./run_nlogn.sh                       # fast estimator, all 1899 pairs, ~1 min (experimental, not used in the paper)
./run_laplacian_direct_estimator.sh  # direct estimator, Table (b), ~5 min (used in the paper)
```


The benchmark suites are the 13 distributed with the
[LOCI repository](https://github.com/aleximmer/loci) — AN, AN-s, LS, LS-s, MN-U,
SIM, SIM-c, SIM-G, SIM-ln, Cha, Multi, Net and Tuebingen. Tuebingen uses the
99-pair protocol: of the 108 pairs, six are multivariate (52–55, 71, 105) and
three have a categorical or heavily tied variable (47, 70, 107); excluding the
nine leaves the set the LOCI baselines were run on.

## Layout

```
lap_nlogn.py                      exact O(n log n) estimator + CLI
laplacian_causality.py            direct O(n^2) estimator + CLI
run_nlogn.sh, run_nlogn.txt       fast benchmark run and its documentation
run_laplacian_direct_estimator.sh direct benchmark run
loci_dataset_benchmark/           loaders, LOCI and RECI runners, table builders
loci/                             vendored LOCI repository: estimator, data, baselines
pairs/                            Tuebingen cause-effect pairs
```

## Note
Currently this research is work in progress.

