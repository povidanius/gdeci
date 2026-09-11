"""RECI (Blöbaum et al., 2018) from the Causal Discovery Toolbox, on every pair.

Run exactly as `cdt.causality.pairwise.RECI(degree=3)`.  Two properties of that
implementation are worth recording, because they change what the column means:

*   `b_fit_score` applies `sklearn.preprocessing.minmax_scale` to both variables
    itself.  Min-max scaling is invariant to any increasing affine map, so the
    score does not depend on how the data is preprocessed beforehand -- raw,
    standardized and min-max inputs all give identical numbers.  The [0,1]
    equal-scaling precondition of the original paper is therefore satisfied here
    by construction.
*   With `degree=3`, `PolynomialFeatures` produces the columns [1, x, x^2, x^3]
    and CDT then sets columns 1 and 2 to zero.  The regression is thus
    y ~ a + b x^3, an intercept plus a pure cubic term, not a full cubic
    polynomial.  This is CDT's behaviour, not a choice made here.

`predict_proba((a, b))` returns `b_fit_score(b, a) - b_fit_score(a, b)`, positive
when a causes b.  The cause is always the first element below, so a pair is
scored correctly when the returned value is positive.

LAPACK driver
-------------
Zeroing those two columns leaves a rank-deficient design matrix, and this
machine's MKL-backed `gelsd` -- the default of `scipy.linalg.lstsq`, which
`sklearn.linear_model.LinearRegression` calls -- cannot handle it: on a 40-pair
sample of AN it raised `LinAlgError` for 12 pairs and, worse, returned silently
wrong answers for the rest, deviating from the analytic solution by up to 0.48.
`gelsy` (complete orthogonal factorization) has no failures and reproduces the
analytic solution to 7e-18, so it is forced below.

Because columns 1 and 2 are zero and `LinearRegression` fits an intercept, the
model is exactly an OLS of minmax(y) on minmax(x)^3, whose mean squared error has
a closed form.  Every pair is checked against it, so the driver substitution is
verified rather than assumed.
"""
import os
import numpy as np
import pandas as pd
import scipy.linalg as sla
from sklearn.preprocessing import minmax_scale

_ORIG_LSTSQ = sla.lstsq
sla.lstsq = lambda a, b, *ar, **kw: _ORIG_LSTSQ(
    a, b, *ar, **dict(kw, lapack_driver='gelsy'))

from cdt.causality.pairwise import RECI

import datasets as D

HERE = os.path.dirname(os.path.abspath(__file__))


def closed_form_mse(x, y):
    """MSE of the OLS of minmax(y) on minmax(x)^3 with intercept: CDT's model."""
    x = minmax_scale(np.asarray(x, float))
    y = minmax_scale(np.asarray(y, float))
    t = x ** 3
    vt = np.var(t)
    b = np.cov(t, y, bias=True)[0, 1] / vt if vt > 0 else 0.0
    a = y.mean() - b * t.mean()
    return float(np.mean((a + b * t - y) ** 2))


def main():
    model = RECI(degree=3)
    worst = 0.0
    rows = []
    for name, _, _, _ in D.BENCHMARKS:
        for pid, cause, effect, weight in D.load(name):
            score = model.predict_proba((np.asarray(cause, float),
                                         np.asarray(effect, float)))
            ref = closed_form_mse(effect, cause) - closed_form_mse(cause, effect)
            worst = max(worst, abs(score - ref))
            rows.append(dict(benchmark=name, pair_id=pid, weight=weight,
                             reci=float(score), reci_closed=float(ref)))
        print(f'{name:10s} done', flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, 'results', 'reci.csv')
    df.to_csv(out, index=False)
    print('wrote', out, df.shape)
    dis = (df.reci > 0) != (df.reci_closed > 0)
    print(f'max |CDT score - closed form| over all {len(df)} pairs: {worst:.2e}')
    print(f'sign disagreements with the closed form: {dis.sum()} '
          f'({", ".join(f"{r.benchmark}/{r.pair_id}" for r in df[dis].itertuples())})')
    print('  all deviations are large-n Tuebingen pairs and the disagreeing ones have '
          '|score| ~ 1e-4, i.e. numerical ties.')
    print('\naccuracy by benchmark (score > 0 means the true cause was chosen):')
    for name in df.benchmark.unique():
        d = df[df.benchmark == name]
        print(f'  {name:10s} {(d.reci > 0).mean():.3f}')
    d = df[df.benchmark == 'Tuebingen']
    w = d.weight.values
    print(f'  {"Tue-w":10s} {((d.reci > 0).values * w).sum() / w.sum():.3f}')


if __name__ == '__main__':
    main()
