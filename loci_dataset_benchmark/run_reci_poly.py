"""RECI as its paper specifies it: the POLY class instead of a zeroed design.

`run_reci.py` runs `cdt.causality.pairwise.RECI(degree=3)` as shipped and lands
below chance on suites every other method solves (macro 0.428).  The reason is
visible in the toolbox source: `b_fit_score` builds
`PolynomialFeatures(degree=3)` -> [1, x, x^2, x^3] and then sets columns 1 and 2
to zero, so the fitted model is

    y ~ a + b x^3,

an intercept plus a single cubic term.  In the vocabulary of the paper
(Bloebaum, Janzing, Washio, Shimizu, Schoelkopf, "Cause-Effect Inference by
Comparing Regression Errors", AISTATS 2018) that is a *shifted monomial*, the
MON class of its Section 5, not the POLY class.  The paper evaluates both:

    MON   a x^n + b,             n in [2, 9]
    POLY  sum_{i=0}^{k} a_i x^i, k in [1, 9]

and its own per-class tables (Tables 5 and 6, standardized CEP and SIM-G) show
why the distinction matters.  The monomials are erratic -- on SIM-G they run
24%, 60%, 29%, 58%, 31%, 58%, 36%, 64% across n = 2..9 -- while the polynomials
are uniformly good and stable, 78-85% for every k >= 2.  The shipped
implementation therefore picks, by accident, one member of the family the
authors found unreliable.

RECI_poly changes the function class and nothing else:

    keep  the toolbox class, its decision rule and its sign convention
          (`predict_proba` is inherited, unchanged);
    keep  the min-max rescaling to [0, 1].  This is not an interchangeable
          preprocessing choice: the paper's analysis is stated for variables
          rescaled to the unit interval, so the method keeps it;
    use   the paper's POLY class, sum_{i=0}^{k} a_i x^i, with k taken from the
          family the paper declares, k in [1, 9];
    pick  k per pair by 5-fold cross-validated fit error, the same k for both
          directions, chosen on the covariates alone.

The degree is the one thing the shipped implementation leaves unusable, so it
is the one thing selected -- by prediction error, never by a causal label, an
accuracy on this benchmark or a held-out suite.  Folds are strided (every fifth
point), so the rule is deterministic and needs no seed.  Selection is over the
paper's own family, and the fits it compares are the same least-squares fits
RECI already performs.

`--sweep` reports every fixed degree k = 1..9 beside the cross-validated rule,
so a reader can see what the choice is worth and what any fixed alternative
would give.  For the most conservative possible statement -- one single change
from the shipped default, its degree parameter untouched -- read the k = 3 row.

Numerics
--------
The fit is done in a Legendre basis on the rescaled variable.  That spans the
same space as [1, x, x^2, x^3], so the fitted values and the MSE are
mathematically identical to a raw-monomial fit; only the solve is better
conditioned.  This is not cosmetic here: on the large, heavily tied Tuebingen
pairs `LinearRegression` on raw `PolynomialFeatures` returns solutions that are
not minimizers, and two runs of it disagree -- the MKL/LAPACK failure documented
in `run_reci.py`.  Because MSE is the quantity being minimized the check is
objective and is run for every pair: the basis used here never has a *larger*
in-sample MSE than the sklearn path, and both totals are printed.

    python run_reci_poly.py              score all 1899 pairs -> results/reci_poly.csv
    python run_reci_poly.py --sweep      accuracy of the POLY family, k = 1..9
    python run_reci_poly.py --diagnose   why the shipped design lands below chance
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.linalg as sla
from numpy.polynomial.legendre import legvander
from sklearn.preprocessing import minmax_scale

# Same substitution as run_reci.py: this machine's MKL-backed `gelsd`, the
# default of `scipy.linalg.lstsq` and hence of `LinearRegression`, is unreliable
# on these designs -- it raises `LinAlgError` and silently returns
# non-minimizers.  `gelsy` is forced so that the sklearn path used for the
# numerical cross-check below is given its best chance.
_ORIG_LSTSQ = sla.lstsq
sla.lstsq = lambda a, b, *ar, **kw: _ORIG_LSTSQ(
    a, b, *ar, **dict(kw, lapack_driver='gelsy'))

from cdt.causality.pairwise import RECI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets as D

DEGREES = [1, 2, 3, 4, 5, 6, 7, 8, 9]   # the paper's POLY family
FOLDS = 5                               # strided, hence deterministic
DEGREE = 3                              # the toolbox's own default, for --sweep


def rescale(v):
    """The paper's rescaling, and the toolbox's: min-max onto [0, 1]."""
    return minmax_scale(np.asarray(v, float))


def cv_mse(x, y, degree, folds=FOLDS):
    """Cross-validated squared error of the degree-k fit of y on x.

    Folds take every `folds`-th point rather than a contiguous block, so a file
    stored in sorted order still yields folds that span the whole range, and no
    random number generator is involved.
    """
    a = _design(x, degree)
    index = np.arange(len(y))
    total = 0.0
    for fold in range(folds):
        test = index % folds == fold
        coef, *_ = np.linalg.lstsq(a[~test], y[~test], rcond=None)
        total += float(np.sum((y[test] - a[test] @ coef) ** 2))
    return total / len(y)


def select_degree(x, y, degrees=DEGREES):
    """Degree with the smallest total cross-validated error over both directions.

    One degree serves both directions, so neither is fitted with more capacity
    than the other, and the criterion never sees a causal label.
    """
    scores = [cv_mse(x, y, k) + cv_mse(y, x, k) for k in degrees]
    return degrees[int(np.argmin(scores))]


def _design(x, degree):
    """Legendre design of the given degree; spans [1, x, ..., x^degree]."""
    lo, hi = x.min(), x.max()
    u = 2 * (x - lo) / (hi - lo) - 1 if hi > lo else np.zeros_like(x)
    return legvander(u, degree)


def poly_mse(x, y, degree):
    """In-sample MSE of the least-squares polynomial fit of y on x."""
    a = _design(x, degree)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return float(np.mean((y - a @ coef) ** 2))


def sklearn_mse(x, y, degree):
    """The same fit through the classes CDT itself uses, for the numerical check.

    Returns inf when the LAPACK solve fails outright; such a solve is counted as
    a failure of that path.
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    px = PolynomialFeatures(degree=degree).fit_transform(x.reshape(-1, 1))
    try:
        reg = LinearRegression().fit(px, y.reshape(-1, 1))
    except np.linalg.LinAlgError:
        return float('inf')
    return float(mean_squared_error(reg.predict(px), y.reshape(-1, 1)))


class RECIPoly(RECI):
    """CDT's RECI with the paper's POLY class in place of the zeroed design.

    `predict_proba` is inherited unchanged, so the decision rule and its sign
    convention are the toolbox's: b_fit_score(b, a) - b_fit_score(a, b),
    positive when a causes b.  `degree=None` selects the degree per pair by
    cross-validated fit error; an integer fixes it.

    The selected degree is cached per pair so that the two calls `predict_proba`
    makes, one per direction, share it; without that the two directions could be
    fitted with different capacity, which the comparison of their errors assumes
    they are not.
    """

    def __init__(self, degree=None, degrees=DEGREES):
        super(RECIPoly, self).__init__(degree=degree)
        self.degrees = degrees
        self._cached = None

    def degree_for(self, x, y):
        """The degree used for this pair, in whichever order it is presented."""
        if self.degree is not None:
            return self.degree
        signature = lambda v: (len(v), float(np.sum(v)), float(np.sum(v * v)))
        key = tuple(sorted((signature(np.asarray(x, float)),
                            signature(np.asarray(y, float)))))
        if self._cached is None or self._cached[0] != key:
            self._cached = (key, select_degree(rescale(x), rescale(y),
                                               self.degrees))
        return self._cached[1]

    def b_fit_score(self, x, y):
        return poly_mse(rescale(x), rescale(y), self.degree_for(x, y))


def score_all():
    model = RECIPoly()
    rows, worse, worst, failed = [], 0, 0.0, 0
    t0 = time.time()
    for name, _, _, _ in D.BENCHMARKS:
        for pid, cause, effect, weight in D.load(name):
            cause = np.asarray(cause, float)
            effect = np.asarray(effect, float)
            score = model.predict_proba((cause, effect))
            # Objective check against the raw-monomial sklearn path: MSE is what
            # is being minimized, so a larger value is a failed solve.
            u, v = rescale(cause), rescale(effect)
            degree = model.degree_for(cause, effect)
            for a, b in ((u, v), (v, u)):
                sk = sklearn_mse(a, b, degree)
                if not np.isfinite(sk):
                    failed += 1
                    continue
                gap = sk - poly_mse(a, b, degree)
                if gap > 1e-9:
                    worse += 1
                    worst = max(worst, gap)
                elif gap < -1e-9:                 # would mean our solve is worse
                    sys.exit(f'orthogonal-basis fit is not the minimizer on '
                             f'{name}/{pid} (gap {gap:.3e})')
            rows.append(dict(benchmark=name, pair_id=pid, n=len(cause),
                             weight=weight, reci_poly=float(score),
                             degree=model.degree_for(cause, effect)))
        print(f'{name:10s} done ({time.time() - t0:5.1f}s)', flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, 'results', 'reci_poly.csv')
    df.to_csv(out, index=False)
    print('wrote', out, df.shape)
    print(f'numerical check over all {2 * len(df)} regressions: the raw-monomial '
          f'sklearn fit raised LinAlgError on {failed} of them and returned a '
          f'non-minimizer on {worse} more (worst excess MSE {worst:.2e}); the '
          f'Legendre-basis fit used here solved every one and was never the worse '
          f'of the two.')

    print('\naccuracy by benchmark (reci_poly > 0 means the true cause was chosen):')
    for name in df.benchmark.unique():
        d = df[df.benchmark == name]
        print(f'  {name:10s} {(d.reci_poly > 0).mean():.3f}')
    d = df[df.benchmark == 'Tuebingen']
    w = d.weight.values
    print(f'  {"Tue-w":10s} {((d.reci_poly > 0).values * w).sum() / w.sum():.3f}')
    counts = df.degree.value_counts().sort_index()
    print('\nselected degree, over all pairs: '
          + ', '.join(f'k={k}: {v}' for k, v in counts.items()))


def sweep():
    """Every fixed degree beside the cross-validated rule, all on min-max data."""
    synth = list(D.SYNTHETIC)
    rows = []
    t0 = time.time()
    for name, _, _, _ in D.BENCHMARKS:
        for pid, cause, effect, weight in D.load(name):
            a, b = rescale(cause), rescale(effect)
            rec = dict(benchmark=name, pair_id=pid)
            for k in DEGREES:
                # CDT's rule: correct when mse(b|a) < mse(a|b)
                rec[f'k{k}'] = poly_mse(a, b, k) < poly_mse(b, a, k)
            picked = select_degree(a, b)
            rec['cv'] = poly_mse(a, b, picked) < poly_mse(b, a, picked)
            rec['cv_margin'] = cv_mse(a, b, picked) < cv_mse(b, a, picked)
            rows.append(rec)
        print(f'{name:10s} done ({time.time() - t0:5.1f}s)', flush=True)
    df = pd.DataFrame(rows)
    bench = df.benchmark.values

    def report(label, column, note=''):
        macro = np.mean([column[bench == s].mean() for s in synth])
        print(f'{label:>26s} {macro:12.3f} {column[bench == "Tuebingen"].mean():11.3f}'
              f'{note}')

    print('\nPOLY family on min-max scaled data, sum_{i=0}^{k} a_i x^i')
    print(f'{"rule":>26s} {"macro mean 12":>12s} {"Tuebingen":>11s}')
    ref = os.path.join(HERE, 'results', 'reci.csv')
    if os.path.exists(ref):
        r = pd.read_csv(ref)
        shipped = (r.set_index(['benchmark', 'pair_id']).reci
                   .reindex(pd.MultiIndex.from_frame(
                       df[['benchmark', 'pair_id']])).values > 0)
        report('CDT as shipped', shipped,
               '   <- a + b x^3, the paper\'s MON class')
    for k in DEGREES:
        report(f'fixed k = {k}', df[f'k{k}'].values,
               '   <- the toolbox\'s degree parameter' if k == DEGREE else '')
    report('cross-validated k', df['cv'].values, '   <- reported column')
    report('cross-validated k, CV margin', df['cv_margin'].values,
           '   (sensitivity: compare CV errors instead of fit errors)')


def diagnose():
    """Show what the shipped design fits, and what it reduces the score to."""
    path = os.path.join(HERE, 'results', 'reci.csv')
    if not os.path.exists(path):
        sys.exit(f'missing {path}; run run_reci.py first')
    reci = pd.read_csv(path)

    print('1. the CDT design matrix, degree=3')
    from sklearn.preprocessing import PolynomialFeatures
    px = PolynomialFeatures(degree=3).fit_transform(np.linspace(0, 1, 5).reshape(-1, 1))
    px[:, 1] = 0
    px[:, 2] = 0
    print('   columns [1, x, x^2, x^3] after CDT zeroes columns 1 and 2:')
    print('  ', np.array2string(px, precision=3).replace('\n', '\n   '))
    print("   -> the model is y ~ a + b x^3: the paper's MON class with n = 3,")
    print('      not its POLY class.  mse(y|x) ~ var(y), so the score collapses')
    print('      onto a difference of marginal variances.\n')

    print('2. the score against var(minmax(x)) - var(minmax(y)), per suite')
    rows = []
    for name in ['AN', 'LS', 'MN-U', 'Multi']:
        for pid, cause, effect, _ in D.load(name):
            rows.append((name, pid, rescale(cause).var() - rescale(effect).var()))
    dv = pd.DataFrame(rows, columns=['benchmark', 'pair_id', 'dvar'])
    m = reci.merge(dv, on=['benchmark', 'pair_id'])
    for name in ['AN', 'LS', 'MN-U', 'Multi']:
        d = m[m.benchmark == name]
        print(f'   {name:6s} corr = {np.corrcoef(d.reci, d.dvar)[0, 1]:+.3f}   '
              f'same sign on {((d.reci > 0) == (d.dvar > 0)).mean():.0%} of pairs   '
              f'median score/dvar = {np.median(d.reci / d.dvar):.2f}')

    print("\n3. against the repository's own marginal control (VarRule:")
    print('   "the smaller min-max variance is the cause")')
    pp = os.path.join(HERE, 'results', 'per_pair.csv')
    if os.path.exists(pp):
        p = pd.read_csv(pp)[['benchmark', 'pair_id', 'var_mm_cause', 'var_mm_effect']]
        m = reci.merge(p, on=['benchmark', 'pair_id'])
        agree = ((m.reci > 0) == (m.var_mm_effect > m.var_mm_cause))
        print(f'   CDT RECI and VarRule choose the same direction on '
              f'{agree.mean():.1%} of all {len(m)} pairs')
        for name in ['AN', 'LS', 'MN-U']:
            d = m[m.benchmark == name]
            print(f'   {name:6s} RECI {(d.reci > 0).mean():.3f}   '
                  f'VarRule {(d.var_mm_effect > d.var_mm_cause).mean():.3f}   '
                  f'agreement {((d.reci > 0) == (d.var_mm_effect > d.var_mm_cause)).mean():.3f}')
    else:
        print(f'   skipped: {pp} not present (run run_benchmark.py)')


def main():
    args = sys.argv[1:]
    if '--sweep' in args:
        sweep()
    elif '--diagnose' in args:
        diagnose()
    elif args:
        sys.exit(f'unknown option: {args[0]}')
    else:
        score_all()


if __name__ == '__main__':
    main()
