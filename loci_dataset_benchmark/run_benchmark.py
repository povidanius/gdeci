"""Score every LOCI benchmark pair with the Laplacian criteria of the paper.

For a candidate direction the graph is built on the putative cause and the
putative effect is the graph signal:

    sigma = m * median_{i<j}|x_i - x_j|      (median recomputed per pair,
    W_ij  = exp(-(x_i-x_j)^2 / 2 sigma^2)     direction and preprocessing)
    E     = y^T (I - D^{-1/2} W D^{-1/2}) y,   M = sum_ij W_ij

A single Gaussian bandwidth is used everywhere; no multi-scale kernel and no
aggregation.  The multiplier is the one frozen in the paper, selected on 300
Causal Discovery Toolbox pairs disjoint from every benchmark here:

    m = 0.05535   under standardization
    m = 0.07362   under the rank transform

All observations of every pair are used; nothing is subsampled.  Raw scores are
written per pair so that decision rules and sign conventions can be varied in
`make_table.py` without recomputing anything.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laplacian_causality import _energy_and_mass, median_bandwidth, mean_distance

import datasets as D

M_STD = 0.05535        # frozen multiplier, standardized marginals
M_RANK = 0.07362       # frozen multiplier, rank transform
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def minmax(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else v - lo


def standardize(v):
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v - v.mean()


def rank_transform(v):
    """Ranks rescaled to (0,1), ties broken deterministically by index.

    Tie-breaking matters.  With `method='ordinal'` both variables carry exactly
    the grid {1..n}/(n+1), so the two directions share a bandwidth and an edge
    mass, and the raw and edge-mass-normalized scores make identical decisions --
    the property the paper relies on to report a single Lap^unif column.  With
    average ranks the ties survive, the edge masses differ (92 of the 99
    Tuebingen pairs, 102 of the 300 Cha pairs) and the two scores come apart by
    up to 12 accuracy points.
    """
    return rankdata(v, method='ordinal') / (len(v) + 1.0)


def bandwidth(t):
    """Median heuristic, with a mean-distance fallback when >50% of pairs tie."""
    s = median_bandwidth(t)
    return s if s > 0 else mean_distance(t)


def both_directions(x, y, m):
    """(E, M) for x->y and for y->x, each at its own median bandwidth."""
    X = torch.as_tensor(x, dtype=torch.float64, device=DEV).reshape(-1, 1)
    Y = torch.as_tensor(y, dtype=torch.float64, device=DEV).reshape(-1, 1)
    ef, mf = _energy_and_mass(X, Y, [m * bandwidth(X)])
    eb, mb = _energy_and_mass(Y, X, [m * bandwidth(Y)])
    return ef, mf, eb, mb


def main():
    rows = []
    for name, _, _, _ in D.BENCHMARKS:
        for pid, cause, effect, weight in D.load(name):
            xs, ys = standardize(cause), standardize(effect)
            xr, yr = rank_transform(cause), rank_transform(effect)
            ef, mf, eb, mb = both_directions(xs, ys, M_STD)
            rf, nf, rb, nb = both_directions(xr, yr, M_RANK)
            rows.append(dict(
                benchmark=name, pair_id=pid, n=len(cause), weight=weight,
                E_std_fwd=ef, M_std_fwd=mf, E_std_bwd=eb, M_std_bwd=mb,
                E_unif_fwd=rf, M_unif_fwd=nf, E_unif_bwd=rb, M_unif_bwd=nb,
                var_cause=float(np.var(cause)), var_effect=float(np.var(effect)),
                var_mm_cause=float(np.var(minmax(cause))),
                var_mm_effect=float(np.var(minmax(effect)))))
        print(f'{name:10s} done ({sum(r["benchmark"] == name for r in rows)} pairs)',
              flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                       'per_pair.csv')
    df.to_csv(out, index=False)
    print('wrote', out, df.shape)


if __name__ == '__main__':
    main()
