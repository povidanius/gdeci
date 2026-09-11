# Graph Dirichlet Energy for Fitting-Free Bivariate Causal Inference

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

# --- a pair with a known ground truth: X causes Y through a folded mechanism ---
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

```
score(X->Y) = 0.0129194
score(Y->X) = 0.0823403
inferred    : X -> Y
margin      : 0.729
```

`x` and `y` are plain 1-D arrays of equal length.


### Checking it on repeated draws

```python
ok = 0
for s in range(20):
    r = np.random.default_rng(s)
    xx = r.uniform(-1, 1, n)
    yy = np.cos(3 * np.pi * xx) + 0.2 * r.standard_normal(n)
    if r.random() < 0.5:                       # randomise which is presented first
        ok += not score(yy, xx, 0.07362, r)[2]
    else:
        ok += score(xx, yy, 0.07362, r)[2]
print(f'{ok}/20 correct')                      # -> 20/20 correct
```

## Reproducing the paper

```bash
./run_nlogn.sh                       # fast estimator, all 1899 pairs, ~1 min
./run_laplacian_direct_estimator.sh  # direct estimator, Table (b), ~5 min
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
