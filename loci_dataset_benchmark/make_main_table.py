"""Table 1 of the paper -- both panels -- from the repository's per-pair outputs.

Nothing is transcribed: every number is recomputed from the per-pair files, so
the table cannot drift from the benchmark it reports.

    panel (a)  QCCD, GRCI, CAM, IGCI, IGCI_G, RESIT
               loci/baseline_results/*.tab -- the LOCI authors' released
               per-pair scores, rescored with that repository's own rule
    panel (b)  LOCI       results_loci/loci.csv        (run_loci.py)
               RECI       results/reci.csv             (run_reci.py)
               RECI_poly  results/reci_poly.csv        (run_reci_poly.py)
               QPE-k      results/qpek.csv             (run_qpek.py)
               GDECI x2   results/per_pair.csv         (run_benchmark.py)
               EdgeMass   results/per_pair.csv

The two GDECI columns are the unnormalized Laplacian L = D - W of the paper:
GDECI^std_avg is E / M under standardization, and GDECI is E / M under the rank
transform.  (E alone under standardization, formerly GDECI^std_raw, is below
chance on several benchmarks and is no longer reported.)  EdgeMass decides
on M alone and involves no Laplacian at all.

Intervals.  Per-benchmark and Tuebingen rows are Wilson intervals at the number
of pairs actually scored; the weighted Tuebingen row is Wilson at the effective
sample size (sum w)^2 / sum w^2, since the dataset weights make that row far
less precise than 99 pairs would suggest.  The macro-average rows average
benchmark accuracies with equal weight, so their intervals come from a
stratified bootstrap that resamples pairs within each benchmark -- `--macro-ci
wilson` switches them to a Wilson interval at the pooled pair count instead.

    python make_main_table.py                  -> results/main_table.tex
    python make_main_table.py --out FILE       write somewhere else
    python make_main_table.py --stdout         print instead of writing
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import datasets as D
from make_table import baseline_decisions, decisions, positive, wilson

NBOOT = 10000
SEED = 0

# (column header as LaTeX, key) for each panel, in printing order.
PANEL_A = [('QCCD', 'QCCD'), ('GRCI', 'GRCI'), ('CAM', 'CAM'),
           ('IGCI', 'IGCI'), (r'IGCI$_G$', 'IGCI_G'), ('RESIT', 'RESIT')]
PANEL_B = [(r'\textsc{LOCI}$^{\dagger}$', 'LOCI'),
           (r'RECI$^{\dagger}$', 'RECI'),
           (r'RECI$^{\dagger}_{\mathrm{poly}}$', 'RECI_poly'),
           (r'QPE-k$^{\dagger}$', 'QPE_k'),
           (r'\textsc{GDECI}$^{\mathrm{std}}_{\mathrm{avg}}$', 'Lap_std_avg'),
           (r'\textsc{GDECI}', 'Lap_unif'),
           (r'\emph{EdgeMass}', 'EdgeMass')]
# key -> (path relative to this directory, score column); correct when positive.
RERUN = {'LOCI': (os.path.join('results_loci', 'loci.csv'), 'loci'),
         'RECI': (os.path.join('results', 'reci.csv'), 'reci'),
         'RECI_poly': (os.path.join('results', 'reci_poly.csv'), 'reci_poly'),
         'QPE_k': (os.path.join('results', 'qpek.csv'), 'qpek')}
BASELINE_KEYS = [k for _, k in PANEL_A]
OURS_KEYS = ['Lap_std_avg', 'Lap_unif', 'EdgeMass']
KEYS = [k for _, k in PANEL_A + PANEL_B]


def load_correctness():
    """{benchmark: DataFrame of per-pair correctness}, one column per method."""
    per_path = os.path.join(HERE, 'results', 'per_pair.csv')
    if not os.path.exists(per_path):
        sys.exit(f'missing {per_path}\n'
                 'run ./run_laplacian_direct_estimator.sh first')
    per = pd.read_csv(per_path)

    rerun, missing = {}, []
    for key, (rel, col) in RERUN.items():
        path = os.path.join(HERE, rel)
        if os.path.exists(path):
            rerun[key] = (pd.read_csv(path), col)
        else:
            missing.append((key, rel))
    if missing:
        for key, rel in missing:
            print(f'warning: {rel} absent; {key} column will be all "---"',
                  file=sys.stderr)

    out = {}
    for name, _, _, _ in D.BENCHMARKS:
        d = per[per.benchmark == name].set_index('pair_id')
        ours = decisions(d)
        frame = pd.DataFrame(index=ours.index)
        for key in OURS_KEYS:
            frame[key] = ours[key].astype(float)
        base = baseline_decisions(name).reindex(ours.index)
        for key in BASELINE_KEYS:
            frame[key] = base[key].astype(float)
        for key, (table, col) in rerun.items():
            sub = table[table.benchmark == name].set_index('pair_id')
            frame[key] = positive(sub[col]).reindex(ours.index)
        for key in KEYS:                       # absent sources -> all NaN
            if key not in frame:
                frame[key] = np.nan
        out[name] = frame[KEYS]
    out['_weights'] = per[per.benchmark == 'Tuebingen'].set_index('pair_id').weight
    return out


def cell(accuracy, lo, hi):
    if not np.isfinite(accuracy):
        return '---'
    return f'${accuracy:.3f}$ {{\\tiny$[{lo:.2f},{hi:.2f}]$}}'


def benchmark_row(frame, key):
    ok = frame[key].to_numpy(float)
    keep = ~np.isnan(ok)
    if not keep.any():
        return cell(np.nan, 0, 0)
    a, n = float(ok[keep].mean()), int(keep.sum())
    return cell(a, *wilson(a, n))


def weighted_tuebingen_row(frame, weights, key):
    ok = frame[key].to_numpy(float)
    keep = ~np.isnan(ok)
    if not keep.any():
        return cell(np.nan, 0, 0)
    w = weights.reindex(frame.index).to_numpy(float)[keep]
    a = float((ok[keep] * w).sum() / w.sum())
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    return cell(a, *wilson(a, n_eff))


def macro_row(correct, group, key, how):
    """Macro mean over `group`, with a bootstrap or Wilson interval."""
    per_suite = [float(np.nanmean(correct[b][key].to_numpy(float)))
                 for b in group]
    if not np.all(np.isfinite(per_suite)):
        return cell(np.nan, 0, 0)
    a = float(np.mean(per_suite))
    total = int(sum(correct[b][key].notna().sum() for b in group))
    if how == 'wilson':
        return cell(a, *wilson(a, total))
    rng = np.random.default_rng(SEED)
    arrays = [correct[b][key].to_numpy(float) for b in group]
    boot = np.empty(NBOOT)
    for i in range(NBOOT):
        boot[i] = np.mean([np.nanmean(x[rng.integers(0, len(x), len(x))])
                           for x in arrays])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return cell(a, lo, hi)


def panel(correct, columns, caption, colspec, macro_ci, indent=''):
    names = [b[0] for b in D.BENCHMARKS]
    synthetic = list(D.SYNTHETIC)
    mooij = list(D.MOOIJ)
    weights = correct['_weights']
    lines = [indent + r'{\footnotesize\textbf{' + caption + r'}}\\[2pt]',
             indent + r'\resizebox{\textwidth}{!}{%',
             indent + r'\begin{tabular}{' + colspec + '}',
             indent + r'\toprule',
             indent + r'benchmark & $N_{\mathrm{pairs}}$ & '
             + ' & '.join(h for h, _ in columns) + r'\\',
             indent + r'\midrule']
    for name in names:
        frame = correct[name]
        cells = [benchmark_row(frame, key) for _, key in columns]
        lines.append(f'{indent}{name} & {len(frame)} & ' + ' & '.join(cells) + r'\\')
    lines.append(indent + r'\midrule')
    tue = correct['Tuebingen']
    cells = [weighted_tuebingen_row(tue, weights, key) for _, key in columns]
    lines.append(indent + r'\emph{Tuebingen, weighted} & 99 & '
                 + ' & '.join(cells) + r'\\')
    lines.append(indent + r'\midrule')
    cells = [macro_row(correct, synthetic, key, macro_ci) for _, key in columns]
    lines.append(indent + r'\emph{macro mean, 12 non-Tuebingen} & 1800 & '
                 + ' & '.join(cells) + r'\\')
    cells = [macro_row(correct, mooij, key, macro_ci) for _, key in columns]
    lines.append(indent + r'%\emph{macro mean, 9 100-pair synthetic} & 900 & '
                 + ' & '.join(cells) + r'\\')
    lines += [indent + r'\bottomrule', indent + r'\end{tabular}}']
    return lines


def build(macro_ci):
    correct = load_correctness()
    out = [r'\begin{table*}[!t]\centering\scriptsize'
           r'\renewcommand{\arraystretch}{0.92}\setlength{\tabcolsep}{4pt}',
           r'\caption{Accuracies with $95\%$ Wilson confidence intervals. Panel '
           r'\textbf{(a)} reflects the published data (LOCI repository), and '
           r'panel \textbf{(b)} the methods run here together.}',
           r'\label{tab:main}', '']
    out += panel(correct, PANEL_A,
                 r'(a) published per-pair outputs, not rerun here',
                 'l r ' + 'r' * len(PANEL_A), macro_ci)
    out += ['', r'\vspace{6pt}', '', '']
    out += panel(correct, PANEL_B,
                 r'(b) rerun here $(\dagger)$, ours, and marginal-only control',
                 'l r rrrr rr r', macro_ci, indent='  ')
    out += [r'\vspace{3pt}', r'\end{table*}']
    return '\n'.join(out) + '\n'


def check(text, columns_a, columns_b):
    """Fail loudly on anything that would typeset as a gap or a '?'."""
    problems = []
    if '?' in text:
        problems.append("the generated table contains a literal '?'")
    for line in text.splitlines():
        if not line.rstrip().endswith(r'\\') or r'\multicolumn' in line:
            continue
        body = line.rstrip()[:-2]
        n = body.count('&') + 1
        expected = {2 + len(columns_a), 2 + len(columns_b)}
        if n not in expected:
            problems.append(f'row has {n} cells, expected one of '
                            f'{sorted(expected)}: {body[:60]}...')
        if '&&' in body.replace(' ', '') or body.strip().endswith('&'):
            problems.append(f'empty cell in: {body[:60]}...')
    if '---' in text:
        problems.append('some cells are "---": a per-pair source file is missing')
    if problems:
        raise SystemExit('table check failed:\n  - ' + '\n  - '.join(problems))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'main_table.tex'))
    ap.add_argument('--stdout', action='store_true', help='print instead of writing')
    ap.add_argument('--macro-ci', default='bootstrap', choices=['bootstrap', 'wilson'],
                    help='interval for the two macro-average rows (default: '
                         'bootstrap, as in the paper)')
    args = ap.parse_args()

    text = build(args.macro_ci)
    check(text, PANEL_A, PANEL_B)
    if args.stdout:
        print(text, end='')
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write(text)
    print(f'wrote {os.path.abspath(args.out)} '
          f'({len(PANEL_A)} + {len(PANEL_B)} method columns, '
          f'macro interval: {args.macro_ci})')


if __name__ == '__main__':
    main()
