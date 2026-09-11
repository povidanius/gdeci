"""Print the direct (dense) Laplacian estimator's results in the paper's layout.

Reads `results/per_pair.csv` written by `run_benchmark.py` and reports the four
columns of Table (b) of paper_sp_letters/.../short_version.tex that this
repository computes -- Lap^std_raw, Lap^std_avg, Lap^unif and EdgeMass -- with
Wilson 95% intervals, in the paper's benchmark order.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets as D
from make_table import decisions, wilson

ORDER = [b[0] for b in D.BENCHMARKS]
SYNTH = [b for b in ORDER if b != 'Tuebingen']
COLS = ['Lap_std_raw', 'Lap_std_avg', 'Lap_unif', 'EdgeMass']
HEAD = {'Lap_std_raw': 'Lap^std_raw', 'Lap_std_avg': 'Lap^std_avg',
        'Lap_unif': 'Lap^unif', 'EdgeMass': 'EdgeMass'}


def main():
    path = os.path.join(HERE, 'results', 'per_pair.csv')
    if not os.path.exists(path):
        sys.exit(f'missing {path}; run run_benchmark.py first')
    per = pd.read_csv(path)
    dec = pd.concat([per[['benchmark', 'pair_id', 'weight']], decisions(per)], axis=1)

    print(f'{"benchmark":30s} {"#pairs":>7s} ' +
          ' '.join(f'{HEAD[c]:>22s}' for c in COLS))
    acc = {}
    for b in ORDER:
        d = dec[dec.benchmark == b]
        n = len(d)
        cells = []
        for c in COLS:
            a = float(d[c].mean())
            acc[(b, c)] = a
            lo, hi = wilson(a, n)
            cells.append(f'{a:.3f} [{lo:.2f},{hi:.2f}]'.rjust(22))
        print(f'{b:30s} {n:7d} ' + ' '.join(cells))

    t = dec[dec.benchmark == 'Tuebingen']
    w = t.weight.values
    ne = w.sum() ** 2 / (w ** 2).sum()
    cells = []
    for c in COLS:
        a = float((t[c].values * w).sum() / w.sum())
        lo, hi = wilson(a, ne)
        cells.append(f'{a:.3f} [{lo:.2f},{hi:.2f}]'.rjust(22))
    print(f'{"Tuebingen, weighted":30s} {len(t):7d} ' + ' '.join(cells))

    cells = []
    for c in COLS:
        a = float(np.mean([acc[(b, c)] for b in SYNTH]))
        lo, hi = wilson(a, 1800)
        cells.append(f'{a:.3f} [{lo:.2f},{hi:.2f}]'.rjust(22))
    print(f'{"macro mean, 12 non-Tuebingen":30s} {1800:7d} ' + ' '.join(cells))
    print(f'\n(Tuebingen is LOCI\'s 99-pair subset; weighted accuracy uses the '
          f'pairmeta weights,\n effective sample size {ne:.1f}.)')


if __name__ == '__main__':
    main()
