"""Cell-by-cell comparison of the replication against Table 1 of papers/main.tex.

Every column that can be recomputed from the LOCI repository is compared.  The
Tuebingen row is expected to differ for the `Lap` and control columns: the paper
scores those on 102 pairs, whereas everything here uses LOCI's own 99-pair subset
(102 minus the discrete pairs 47, 70 and 107 that LOCI blacklists).  The
baseline columns on that row already use the 99-pair subset in the paper, so they
should agree exactly.
"""
import os
import numpy as np
import pandas as pd

import datasets as D
from make_table import decisions, baseline_decisions, PLAIN, HERE

# Table 1 of papers/main.tex, columns that are recomputable here.  RECI is the
# starred column, transcribed there from Chen et al.  LOCI is the paper's own
# earlier run of the released code, marked '---' on the three Dataverse
# benchmarks it was not run on; those are scored here for the first time.
# RECI is it is absent from LOCI's
# baseline files, so it is the one baseline column actually rerun here.  Its
# Tuebingen entry in the paper uses a third pair set (102 minus nine discrete
# pairs = 93), so that cell is not comparable.
PAPER = {
#                LOCI   GRCI   QCCD    CAM  IGCI_G  RESIT   IGCI   RECI    Lraw   Lavg   Lunif  Edge   Var
'AN':        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.200, 0.180, 1.000, 1.000, 1.000, 0.940, 0.160],
'AN-s':      [1.000, 0.940, 0.820, 1.000, 1.000, 1.000, 0.350, 0.350, 1.000, 1.000, 0.810, 0.960, 0.190],
'LS':        [0.940, 0.980, 1.000, 1.000, 0.980, 0.600, 0.460, 0.220, 1.000, 1.000, 1.000, 0.920, 0.170],
'LS-s':      [0.890, 0.870, 0.960, 0.530, 0.990, 0.030, 0.340, 0.440, 0.690, 0.430, 0.920, 0.950, 0.080],
'MN-U':      [1.000, 0.880, 0.990, 0.860, 1.000, 0.050, 0.110, 0.130, 0.910, 0.710, 0.990, 0.990, 0.030],
'SIM':       [0.780, 0.770, 0.620, 0.570, 0.380, 0.780, 0.370, 0.440, 0.590, 0.720, 0.660, 0.350, 0.460],
'SIM-c':     [0.820, 0.770, 0.720, 0.600, 0.390, 0.820, 0.450, 0.530, 0.590, 0.620, 0.660, 0.360, 0.500],
'SIM-G':     [0.780, 0.700, 0.640, 0.810, 0.830, 0.770, 0.530, 0.390, 0.830, 0.760, 0.590, 0.850, 0.400],
'SIM-ln':    [0.730, 0.770, 0.800, 0.870, 0.590, 0.870, 0.510, 0.440, 0.830, 0.850, 0.790, 0.510, 0.450],
'Cha':       [np.nan, 0.700, 0.537, 0.467, 0.580, 0.343, 0.550, 0.560, 0.470, 0.500, 0.517, 0.583, 0.550],
'Multi':     [np.nan, 0.773, 0.507, 0.347, 0.680, 0.373, 0.923, 0.850, 0.387, 0.330, 0.570, 0.683, 0.880],
'Net':       [np.nan, 0.847, 0.803, 0.783, 0.597, 0.783, 0.553, 0.600, 0.767, 0.863, 0.800, 0.557, 0.557],
'Tuebingen': [0.598, 0.798, 0.697, 0.576, 0.606, 0.525, 0.657, 0.640, 0.539, 0.745, 0.657, 0.284, 0.618],
}
PAPER_MEAN12 = [np.nan, 0.833, 0.783, 0.736, 0.751, 0.618, 0.446, 0.508, 0.755, 0.732, 0.776, 0.721, 0.369]
PAPER_MEAN9 = [0.882, 0.853, 0.839, 0.804, 0.796, 0.658, 0.369, 0.347, 0.827, 0.788, 0.824, 0.759, 0.271]
PAPER_TUEW = [0.605, 0.816, 0.771, 0.578, 0.563, 0.620, 0.680, np.nan, 0.524, 0.646, 0.618, 0.396, 0.698]

COLS = ['LOCI', 'GRCI', 'QCCD', 'CAM', 'IGCI_G', 'RESIT', 'IGCI', 'RECI',
        'Lap_std_raw', 'Lap_std_avg', 'Lap_unif', 'EdgeMass', 'VarRule']


def main():
    per = pd.read_csv(os.path.join(HERE, 'results', 'per_pair.csv'))
    reci = pd.read_csv(os.path.join(HERE, 'results', 'reci.csv'))
    loci = pd.read_csv(os.path.join(HERE, 'results_loci', 'loci.csv'))
    mine, correct = {}, {}
    for name in [b[0] for b in D.BENCHMARKS]:
        d = per[per.benchmark == name].set_index('pair_id')
        ours = decisions(d)
        base = baseline_decisions(name).reindex(ours.index)
        r = reci[reci.benchmark == name].set_index('pair_id')
        ours['RECI'] = (r.reci > 0).reindex(ours.index)
        l = loci[loci.benchmark == name].set_index('pair_id')
        ours['LOCI'] = (l.loci > 0).reindex(ours.index)
        allm = pd.concat([base, ours], axis=1)
        correct[name] = allm
        mine[name] = allm.mean(0)

    d = per[per.benchmark == 'Tuebingen'].set_index('pair_id')
    w = d.weight.values
    mine['Tue-w'] = pd.Series({c: float((correct['Tuebingen'][c].values * w).sum() / w.sum())
                               for c in COLS})
    for label, group in (('mean12', D.SYNTHETIC), ('mean9', D.MOOIJ)):
        mine[label] = pd.Series({c: np.mean([mine[g][c] for g in group]) for c in COLS})

    rows = list(PAPER.items()) + [('mean12', PAPER_MEAN12), ('mean9', PAPER_MEAN9),
                                  ('Tue-w', PAPER_TUEW)]
    print(f'{"benchmark":11s} ' + ' '.join(f'{c[:11]:>13s}' for c in COLS))
    n_exact = n_cell = 0
    worst = []
    for name, vals in rows:
        cells = []
        for c, pv in zip(COLS, vals):
            if not np.isfinite(pv):
                cells.append('   n/a'.rjust(13))
                continue
            mv = mine[name][c]
            diff = mv - pv
            n_cell += 1
            n_exact += abs(diff) < 5e-4
            if abs(diff) >= 5e-4:
                worst.append((abs(diff), name, c, pv, mv))
            cells.append(f'{mv:.3f}{"  =" if abs(diff) < 5e-4 else f"{diff:+.3f}"}'.rjust(13))
        print(f'{name:11s} ' + ' '.join(cells))
    print(f'\n{n_exact}/{n_cell} cells reproduce the paper exactly '
          f'({n_exact / n_cell:.1%}).')
    print('\ncells that differ, largest first:')
    for a, name, c, pv, mv in sorted(worst, reverse=True):
        print(f'  {name:11s} {c:12s} paper {pv:.3f}  ours {mv:.3f}  ({mv - pv:+.3f})')


if __name__ == '__main__':
    main()
