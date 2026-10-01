"""Print the two RECI columns side by side, in the layout of Table (b).

Reads `results/reci.csv` (the Causal Discovery Toolbox default, written by
`run_reci.py`) and `results/reci_poly.csv` (the same toolbox run as RECI's paper
specifies it, written by `run_reci_poly.py`), and reports both on exactly the
same pair sets, with Wilson 95% intervals, in the paper's benchmark order.

A pair is answered correctly when the score is positive: both columns use CDT's
own decision rule `b_fit_score(b, a) - b_fit_score(a, b)`, and the loaders always
put the true cause first.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets as D
from make_table import wilson

ORDER = [b[0] for b in D.BENCHMARKS]
SYNTH = [b for b in ORDER if b != 'Tuebingen']
COLS = [('reci', 'RECI (CDT default)'), ('reci_poly', 'RECI_poly')]


def load():
    frames = []
    for col, _ in COLS:
        path = os.path.join(HERE, 'results',
                            'reci.csv' if col == 'reci' else 'reci_poly.csv')
        if not os.path.exists(path):
            sys.exit(f'missing {path}; run '
                     f'{"run_reci.py" if col == "reci" else "run_reci_poly.py"} first')
        df = pd.read_csv(path)[['benchmark', 'pair_id', 'weight', col]]
        frames.append(df)
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df.drop(columns='weight'), on=['benchmark', 'pair_id'])
    if len(out) != 1899:
        sys.exit(f'expected 1899 pairs common to both runs, got {len(out)}')
    for col, _ in COLS:
        out[col] = out[col] > 0
    return out


def main():
    dec = load()
    head = ' '.join(f'{h:>22s}' for _, h in COLS)
    print(f'{"benchmark":30s} {"#pairs":>7s} {head}')
    acc = {}
    for b in ORDER:
        d = dec[dec.benchmark == b]
        cells = []
        for col, _ in COLS:
            a = float(d[col].mean())
            acc[(b, col)] = a
            lo, hi = wilson(a, len(d))
            cells.append(f'{a:.3f} [{lo:.2f},{hi:.2f}]'.rjust(22))
        print(f'{b:30s} {len(d):7d} ' + ' '.join(cells))

    t = dec[dec.benchmark == 'Tuebingen']
    w = t.weight.values
    ne = w.sum() ** 2 / (w ** 2).sum()
    cells = []
    for col, _ in COLS:
        a = float((t[col].values * w).sum() / w.sum())
        lo, hi = wilson(a, ne)
        cells.append(f'{a:.3f} [{lo:.2f},{hi:.2f}]'.rjust(22))
    print(f'{"Tuebingen, weighted":30s} {len(t):7d} ' + ' '.join(cells))

    cells = []
    for col, _ in COLS:
        a = float(np.mean([acc[(b, col)] for b in SYNTH]))
        lo, hi = wilson(a, 1800)
        cells.append(f'{a:.3f} [{lo:.2f},{hi:.2f}]'.rjust(22))
    print(f'{"macro mean, 12 non-Tuebingen":30s} {1800:7d} ' + ' '.join(cells))

    n_flip = int((dec.reci != dec.reci_poly).sum())
    print(f'\n(RECI (CDT default) = cdt.causality.pairwise.RECI(degree=3) as shipped:'
          f'\n an intercept-plus-pure-cubic fit, i.e. the MON class of the RECI'
          f'\n paper, on min-max scaled data.'
          f'\n RECI_poly = the same class, decision rule and min-max rescaling'
          f'\n with the paper\'s POLY design, sum_i a_i x^i, k from the paper\'s family'
          f'\n [1,9] chosen per pair by 5-fold cross-validated fit error, one k for'
          f'\n both directions; no causal label enters the choice.  run_reci_poly.py'
          f'\n --sweep lists every fixed degree beside it.'
          f'\n The two disagree on {n_flip} of {len(dec)} pairs.'
          f'\n Tuebingen is LOCI\'s 99-pair subset; the weighted row uses the pairmeta'
          f'\n weights, effective sample size {ne:.1f}.)')


if __name__ == '__main__':
    main()
