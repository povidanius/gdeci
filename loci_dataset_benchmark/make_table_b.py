"""Split `results/table.tex` into the two panels the paper's Table (b) uses.

The paper prints the wide table of `make_table.py` as two stacked panels: (a) the
published per-pair outputs that were not rerun, and (b) the methods rerun here
($\\dagger$), ours, and the marginal-only control.  Rather than re-deriving the
numbers, this reads the generated `results/table.tex` and re-emits its cells in
that layout, so the panels cannot drift from the table they are cut from.

Panel (b) carries the QPE-k column in the rerun group, next to LOCI and RECI:
like those two it is run in this repository, not read from LOCI's released files.

    python make_table.py && python make_table_b.py     -> results/table_ab.tex
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

PANEL_A = ['QCCD', 'GRCI', 'CAM', 'IGCI', 'IGCI_G', 'RESIT', 'RESIT_std']
PANEL_B = ['LOCI', 'RECI', 'QPE_k', 'Lap_std_avg', 'Lap_unif', 'EdgeMass']
HEAD = {'QCCD': 'QCCD', 'GRCI': 'GRCI', 'CAM': 'CAM', 'IGCI': 'IGCI',
        'IGCI_G': r'IGCI$_G$', 'RESIT': 'RESIT',
        'RESIT_std': r'RESIT$_{\mathrm{std}}$',
        'LOCI': r'\textsc{Loci}$^{\dagger}$', 'RECI': r'RECI$^{\dagger}$',
        'QPE_k': r'QPE-k$^{\dagger}$',
        'Lap_std_avg': r'\textsc{Lap}$^{\mathrm{std}}_{\mathrm{avg}}$',
        'Lap_unif': r'\textsc{Lap}$^{\mathrm{unif}}$',
        'EdgeMass': r'\emph{EdgeMass}'}
# label in table.tex -> (label in the paper, pair count, commented out?)
RELABEL = {r'\emph{mean, 12 synthetic}':
           (r'\emph{macro mean, 12 non-Tuebingen}', '1800', False),
           r'\emph{mean, 9 Mooij}':
           (r'\emph{macro mean, 9 100-pair synthetic}', '900', True)}


def column_index():
    """{key: column} read from the wide table's own header row.

    `make_table.py` has emitted different column sets over time -- QPE-k is
    present in some runs and not in others -- so the layout is taken from the
    file being split rather than hard-coded here, and a panel column that the
    table does not carry is dropped instead of shifting every later cell.
    """
    by_head = {head: key for key, head in HEAD.items()}
    for line in open(os.path.join(HERE, 'results', 'table.tex')):
        line = line.rstrip('\n')
        if not line.startswith('benchmark &') or not line.endswith(r'\\'):
            continue
        heads = [p.strip() for p in line[:-2].split('&')][2:]
        missing = [h for h in heads if h not in by_head]
        return ({by_head[h]: i for i, h in enumerate(heads) if h in by_head},
                len(heads), missing)
    raise SystemExit('no header row found in results/table.tex')


IDX, NCOLS, UNKNOWN_HEADS = column_index()


def rows():
    """(label, n_pairs, cells) for every body row of the wide table."""
    out = []
    for line in open(os.path.join(HERE, 'results', 'table.tex')):
        line = line.rstrip('\n')
        if not line.endswith(r'\\') or r'\multicolumn' in line:
            continue
        parts = [p.strip() for p in line[:-2].split('&')]
        if len(parts) != 2 + NCOLS or not re.match(r'^[\d ]', parts[1]):
            continue
        out.append((parts[0], parts[1], parts[2:]))
    return out


def panel(caption, cols, colspec, body):
    tex = [r'{\footnotesize\textbf{' + caption + r'}}\\[2pt]',
           r'\resizebox{\textwidth}{!}{%',
           r'\begin{tabular}{l r ' + colspec + '}',
           r'\toprule',
           'benchmark & $N_pairs$ & ' +
           ' & '.join(HEAD[c] for c in cols) + r'\\',
           r'\midrule']
    for label, n, cells in body:
        if label.startswith(r'\emph{'):
            tex.append(r'\midrule')
        pre = ''
        if label in RELABEL:
            label, n, commented = RELABEL[label]
            pre = '%' if commented else ''
            if commented:
                tex.pop()                      # no \midrule before a commented row
        tex.append(pre + f'{label} & {n} & ' +
                   ' & '.join(cells[IDX[c]] for c in cols) + r'\\')
    tex += [r'\bottomrule', r'\end{tabular}}']
    return tex


def main():
    body = rows()
    if not body:
        raise SystemExit('no body rows parsed from results/table.tex; '
                         'run make_table.py first')
    panel_a = [c for c in PANEL_A if c in IDX]
    panel_b = [c for c in PANEL_B if c in IDX]
    for name, present, wanted in (('(a)', panel_a, PANEL_A),
                                  ('(b)', panel_b, PANEL_B)):
        dropped = [c for c in wanted if c not in present]
        if dropped:
            print(f'note: panel {name} omits {", ".join(dropped)}; '
                  f'results/table.tex does not carry those columns')
    tex = panel(r'(a) published per-pair outputs, not rerun here', panel_a,
                'r' * len(panel_a), body)
    tex += ['', r'\vspace{6pt}']
    # grouped as in the paper: the ones rerun here, then ours, then the control
    tex += panel(r'(b) rerun here $(\dagger)$, ours, and marginal-only control',
                 panel_b, 'r' * len(panel_b), body)
    out = os.path.join(HERE, 'results', 'table_ab.tex')
    open(out, 'w').write('\n'.join(tex) + '\n')
    print('wrote', out, f'({len(body)} rows)')


if __name__ == '__main__':
    main()
