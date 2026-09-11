"""Accuracy table for the LOCI benchmarks, in the format of Table 1 of the paper,
with 95% confidence intervals.

Point estimates come from `results/per_pair.csv` (our Laplacian criteria) and from
`loci/baseline_results/*.tab` (the LOCI authors' own per-pair runs of QCCD, GRCI,
CAM, IGCI, IGCI_G and RESIT), evaluated on exactly the same pair sets.

Intervals are Wilson score intervals at 95%.  For the weighted Tuebingen row the
Wilson formula is applied at the effective sample size
(sum w)^2 / sum w^2 = 58.9 rather than at 99, since the dataset weights make the
row far less precise than the pair count suggests.  The two macro-average rows
average dataset accuracies with equal weight, as in the paper, so their intervals
come from a stratified bootstrap that resamples pairs within each dataset.
"""
import os
import numpy as np
import pandas as pd

import datasets as D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), 'loci', 'baseline_results')
Z = 1.959963984540054
RNG = np.random.default_rng(0)
NBOOT = 10000

TAB = {'AN': 'AN', 'AN-s': 'ANs', 'LS': 'LS', 'LS-s': 'LSs', 'MN-U': 'MNU',
       'SIM': 'SIM', 'SIM-c': 'SIMc', 'SIM-G': 'SIMG', 'SIM-ln': 'SIMln',
       'Cha': 'Cha', 'Multi': 'Multi', 'Net': 'Net', 'Tuebingen': 'Tuebingen'}
# The paper's baseline columns come from these files.  Note RESIT: the paper's
# published figures match the `RESIT` column, whereas LOCI's own
# generate_figures_and_tables.py scores `RESIT_std`.  The two agree on most
# benchmarks but not on Cha (0.343 vs 0.723), so both are reported.
BASELINES = [('QCCD', 'QCCD'), ('GRCI', 'GRCI'), ('CAM', 'CAM'),
             ('IGCI', 'IGCI'), ('IGCI$_G$', 'IGCI_G'), ('RESIT', 'RESIT'),
             ('RESIT$_{\mathrm{std}}$', 'RESIT_std')]
OURS = [r'\textsc{Lap}$^{\mathrm{std}}_{\mathrm{raw}}$', 
        r'\textsc{Lap}$^{\mathrm{std}}_{\mathrm{avg}}$',
        r'\textsc{Lap}$^{\mathrm{unif}}$']
CONTROLS = [r'\emph{EdgeMass}', r'\emph{VarRule}']
RERUN = [r'\textsc{Loci}$^{\dagger}$', r'RECI$^{\dagger}$']
PLAIN = ['QCCD', 'GRCI', 'CAM', 'IGCI', 'IGCI_G', 'RESIT', 'RESIT_std',
         'LOCI', 'RECI',
         'Lap_std_raw', 'Lap_std_avg', 'Lap_unif', 'EdgeMass', 'VarRule']


def wilson(k_or_p, n):
    p = k_or_p
    d = 1.0 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z / d * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def positive(series):
    """1.0 / 0.0 for a scored pair, NaN for one that failed or was not run.

    A NaN score must not collapse to False: that would count an unscored pair as
    a wrong answer and leave the denominator unchanged.  Everything downstream
    counts non-missing entries per method instead.
    """
    return pd.Series(np.where(series.isna(), np.nan, (series > 0).astype(float)),
                     index=series.index)


def decisions(df):
    """Per-pair correctness of each method; the cause is column 1 by construction."""
    out = pd.DataFrame(index=df.index)
    out['Lap_std_raw'] = df.E_std_fwd < df.E_std_bwd
    out['Lap_std_avg'] = df.E_std_fwd / df.M_std_fwd < df.E_std_bwd / df.M_std_bwd
    out['Lap_unif'] = df.E_unif_fwd / df.M_unif_fwd < df.E_unif_bwd / df.M_unif_bwd
    # Both controls follow the sign convention of the paper's table: the variable
    # with the SMALLER edge mass, respectively the smaller min-max variance, is
    # called the cause.  Reversing either rule gives one minus these accuracies.
    out['EdgeMass'] = df.M_std_fwd < df.M_std_bwd
    out['VarRule'] = df.var_mm_effect < df.var_mm_cause
    return out


def baseline_decisions(name):
    t = pd.read_csv(os.path.join(BASE, TAB[name] + '.tab'), sep='\t', index_col=0)
    out = pd.DataFrame(index=t.index)
    for label, col in BASELINES:
        out[col if col.startswith('RESIT') else label.replace('$_G$', '_G')] = \
            (t[col] * t['GroundTruth']) > 0
    return out


def main():
    per = pd.read_csv(os.path.join(HERE, 'results', 'per_pair.csv'))
    reci = pd.read_csv(os.path.join(HERE, 'results', 'reci.csv'))
    loci = pd.read_csv(os.path.join(HERE, 'results_loci', 'loci.csv'))
    acc, lo, hi, correct, cnt = {}, {}, {}, {}, {}
    names = [b[0] for b in D.BENCHMARKS]

    for name in names:
        d = per[per.benchmark == name].set_index('pair_id')
        ours = decisions(d)
        base = baseline_decisions(name).reindex(ours.index)
        r = reci[reci.benchmark == name].set_index('pair_id')
        ours['RECI'] = positive(r.reci).reindex(ours.index)
        l = loci[loci.benchmark == name].set_index('pair_id')
        ours['LOCI'] = positive(l.loci).reindex(ours.index)
        allm = pd.concat([base, ours], axis=1)[PLAIN]
        correct[name] = allm
        acc[name] = allm.mean(0)                      # NaN-skipping
        cnt[name] = allm.notna().sum(0)               # pairs actually scored
        lo[name], hi[name] = {}, {}
        for m in PLAIN:
            k = int(cnt[name][m])
            lo[name][m], hi[name][m] = (wilson(acc[name][m], k) if k else (np.nan, np.nan))

    # weighted Tuebingen, LOCI's own scoring rule, at the effective sample size
    d = per[per.benchmark == 'Tuebingen'].set_index('pair_id')
    w = d.weight.values
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    tw = correct['Tuebingen']
    acc['Tue-w'], lo['Tue-w'], hi['Tue-w'], cnt['Tue-w'] = {}, {}, {}, {}
    for m in PLAIN:
        ok = tw[m].values.astype(float)
        keep = ~np.isnan(ok)
        ww = w[keep]
        acc['Tue-w'][m] = float((ok[keep] * ww).sum() / ww.sum()) if keep.any() else np.nan
        ne = ww.sum() ** 2 / (ww ** 2).sum() if keep.any() else 0.0
        cnt['Tue-w'][m] = int(keep.sum())
        lo['Tue-w'][m], hi['Tue-w'][m] = (wilson(acc['Tue-w'][m], ne) if keep.any()
                                          else (np.nan, np.nan))
    acc['Tue-w'] = pd.Series(acc['Tue-w'])

    # macro-averages with a stratified bootstrap
    for label, group in (('mean, 12 synthetic', D.SYNTHETIC), ('mean, 9 Mooij', D.MOOIJ)):
        acc[label] = pd.Series({m: np.nanmean([acc[g][m] for g in group]) for m in PLAIN})
        cnt[label] = pd.Series({m: int(sum(cnt[g][m] for g in group)) for m in PLAIN})
        arrs = {g: correct[g][PLAIN].values.astype(float) for g in group}
        boot = np.empty((NBOOT, len(PLAIN)))
        for b in range(NBOOT):
            boot[b] = np.nanmean(
                [np.nanmean(arrs[g][RNG.integers(0, len(arrs[g]), len(arrs[g]))], axis=0)
                 for g in group], axis=0)
        q = np.percentile(boot, [2.5, 97.5], axis=0)
        lo[label] = dict(zip(PLAIN, q[0]))
        hi[label] = dict(zip(PLAIN, q[1]))

    order = names + ['Tue-w', 'mean, 12 synthetic', 'mean, 9 Mooij']
    npairs = {n: len(correct[n]) for n in names}
    npairs['Tue-w'] = f'99 (n_eff {n_eff:.1f})'
    npairs['mean, 12 synthetic'] = 1800
    npairs['mean, 9 Mooij'] = 900

    print(f'{"benchmark":20s} {"pairs":>14s} ' +
          ' '.join(f'{m:>21s}' for m in PLAIN))
    for r in order:
        cells = ' '.join(f'{acc[r][m]:.3f} [{lo[r][m]:.2f},{hi[r][m]:.2f}]'.rjust(21)
                         for m in PLAIN)
        print(f'{r:20s} {str(npairs[r]):>14s} {cells}')

    miss = {m: {r: int(len(correct[r]) - cnt[r][m]) for r in names
                if cnt[r][m] != len(correct[r])} for m in PLAIN}
    miss = {m: v for m, v in miss.items() if v}
    print('\ncoverage: ' + ('every method scored every pair'
          if not miss else 'MISSING SCORES -> ' + str(miss)))

    # sanity check on the claim that raw and averaged coincide under the rank transform
    rel = ((per.M_unif_fwd - per.M_unif_bwd).abs() /
           per[['M_unif_fwd', 'M_unif_bwd']].max(1))
    same = ((per.E_unif_fwd < per.E_unif_bwd) ==
            (per.E_unif_fwd / per.M_unif_fwd < per.E_unif_bwd / per.M_unif_bwd))
    print(f'\nrank transform, |M_fwd-M_bwd|/max: median {rel.median():.2e}, '
          f'max {rel.max():.2e}')
    print(f'raw and averaged agree on {same.mean():.4f} of the {len(per)} pairs '
          f'({(~same).sum()} disagreements)')
    for name in names:
        s = same[per.benchmark == name]
        if not s.all():
            print(f'    {name}: {(~s).sum()} pairs differ')

    # LaTeX
    head = [l for l, _ in BASELINES] + RERUN + OURS + CONTROLS
    tex = [r'\begin{table}[t]\centering\scriptsize',
           r'\caption{Accuracies with $95\%$ confidence intervals on the bivariate '
           r'benchmarks distributed with the LOCI repository. \textbf{Only the last '
           r'five columns were computed by us.} The six baseline columns were '
           r'\emph{not} rerun: they are the LOCI authors'"'"' released per-pair scores '
           r'(\texttt{baseline\_results/*.tab}) rescored with that repository'"'"'s own '
           r'rule, $(\mathrm{score}\times\mathrm{GroundTruth})>0$, on the same pair '
           r'sets. Intervals are Wilson intervals computed from the per-pair '
           r'correct/incorrect vector, so they capture sampling variability over '
           r'pairs only; for the baselines they are conditional on the single '
           r'released run and do not integrate over those methods'"'"' own '
           r'stochasticity. The weighted Tuebingen row uses the effective sample '
           f'size $n_{{\\mathrm{{eff}}}}={n_eff:.1f}$; the macro-average rows use a '
           r'stratified bootstrap. Pairs are treated as independent, which '
           r'overstates precision on Tuebingen, where pairs cluster by source '
           r'dataset. $^{\dagger}$RECI is '
           r'\texttt{cdt.causality.pairwise.RECI(degree=3)} and \textsc{Loci} is '
           r'\texttt{causa.loci.loci} at its packaged defaults (neural '
           r'heteroscedastic estimator, 5000 epochs, HSIC residual test, '
           r'standardized inputs); both were run here. Neither appears in the '
           r'LOCI baseline files.}',
           r'\label{tab:loci-replication}',
           r'\resizebox{\textwidth}{!}{%',
           r'\begin{tabular}{l r ' + 'r' * 7 + ' rr ' + 'r' * 3 + ' ' + 'r' * 2 + '}',
           r'\toprule',
           r'& & \multicolumn{7}{c}{published per-pair outputs, not rerun} & '
           r'\multicolumn{2}{c}{rerun here} & '
           r'\multicolumn{3}{c}{ours} & \multicolumn{2}{c}{marginal-only}\\',
           r'\cmidrule(lr){3-9}\cmidrule(lr){10-11}\cmidrule(lr){12-14}'
           r'\cmidrule(lr){15-16}',
           'benchmark & $n$ & ' + ' & '.join(head) + r'\\',
           r'\midrule']
    npairs_tex = dict(npairs, **{'Tue-w': 99})   # n_eff goes in the caption, not a cell
    for r in order:
        if r in ('Tue-w', 'mean, 12 synthetic'):
            tex.append(r'\midrule')
        label = {'Tue-w': r'\emph{Tuebingen, weighted}',
                 'mean, 12 synthetic': r'\emph{mean, 12 synthetic}',
                 'mean, 9 Mooij': r'\emph{mean, 9 Mooij}'}.get(r, r)
        cells = [f'${acc[r][m]:.3f}$ {{\\tiny$[{lo[r][m]:.2f},{hi[r][m]:.2f}]$}}'
                 for m in PLAIN]
        tex.append(f'{label} & {npairs_tex[r]} & ' + ' & '.join(cells) + r'\\')
    tex += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    path = os.path.join(HERE, 'results', 'table.tex')
    open(path, 'w').write('\n'.join(tex) + '\n')
    print('\nwrote', path)


if __name__ == '__main__':
    main()
