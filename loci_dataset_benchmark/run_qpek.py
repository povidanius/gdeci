"""QPE-k (Chen et al., 2025, arXiv:2509.12981) on every LOCI benchmark pair.

QPE-k is the *fast kernel* estimator of the Quantile Partial Effect: it forms a
Nadaraya-Watson estimate of the conditional CDF on a fixed 20x20 grid, takes its
derivative with respect to the conditioning variable, divides by the conditional
density and measures how far the resulting surface is from the span of a small
basis in y.  The direction whose QPE surface is *better* explained by the basis
(smaller least-squares residual, hence larger score) is called causal.

The estimator itself is `qpe_k` from the `qpe_cd` submodule, used verbatim -- the
only thing this script adds is the benchmark loop, the preprocessing and the
bookkeeping.

Protocol
--------
*   Marginals are standardized, as in `qpe_cd/reproduce/bivariate.py`, which
    passes `StandardScaler()` to every loader.  This is not cosmetic: `qpe_k`
    evaluates on the fixed grid `linspace(-2.5, 2.5, 20)` in both variables, so
    unstandardized data would be scored on a grid that misses its support.
*   The default basis of the paper, `[1, y, y^3, tanh(y)]`, is used.
*   Unlike `qpe_cd/reproduce/run_qpek.py` the pair is *not* randomly flipped
    before scoring.  The flip there only serves to symmetrize a run that reports
    a single direction; `qpe_k` returns both scores at once, and this repository
    always puts the true cause first, so scoring in the fixed order is both
    deterministic and equivalent.

A pair is answered correctly when `qpek = score_fwd - score_bwd` is positive.
Exact ties (identical scores in both directions) count as failures, matching the
`score_12 > score_21` rule of the upstream reproduction script.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'qpe_cd'))
sys.path.insert(0, HERE)

from qpek import qpe_k

import datasets as D

BASIS_FUNCS = [
    lambda y: np.ones_like(y),
    lambda y: y,
    lambda y: y ** 3,
    lambda y: np.tanh(y),
]


def standardize(v):
    """Zero mean, unit variance; constant variables are only centred."""
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v - v.mean()


def main():
    rows = []
    for name, _, _, _ in D.BENCHMARKS:
        t0 = time.time()
        for pid, cause, effect, weight in D.load(name):
            x = standardize(cause).reshape(-1, 1)
            y = standardize(effect).reshape(-1, 1)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                _, s_fwd, _, s_bwd = qpe_k(x, y, basis_funcs=BASIS_FUNCS)
            rows.append(dict(benchmark=name, pair_id=pid, n=len(cause),
                             weight=weight, qpek_fwd=float(s_fwd),
                             qpek_bwd=float(s_bwd),
                             qpek=float(s_fwd) - float(s_bwd)))
        k = sum(r['benchmark'] == name for r in rows)
        print(f'{name:10s} done ({k} pairs, {time.time() - t0:.1f}s)', flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, 'results', 'qpek.csv')
    df.to_csv(out, index=False)
    print('wrote', out, df.shape)

    bad = int(df.qpek.isna().sum())
    if bad:
        print(f'warning: {bad} pairs scored NaN (counted as failures)')
    ties = int((df.qpek == 0).sum())
    if ties:
        print(f'{ties} exact ties (counted as failures)')

    print('\naccuracy by benchmark (qpek > 0 means the true cause was chosen):')
    for name in df.benchmark.unique():
        d = df[df.benchmark == name]
        print(f'  {name:10s} {(d.qpek > 0).mean():.3f}')
    d = df[df.benchmark == 'Tuebingen']
    w = d.weight.values
    print(f'  {"Tue-w":10s} {((d.qpek > 0).values * w).sum() / w.sum():.3f}')


if __name__ == '__main__':
    main()
