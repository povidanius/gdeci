"""Loaders for every bivariate benchmark shipped with the LOCI repository.

These mirror `loci/causa/datasets.py` exactly -- same files, same column
conventions, same orientation of the returned (cause, effect) tuple -- so that
the pair sets scored here are identical to the ones the LOCI baselines were run
on.  Differences from that module: no torch, no gin, and the multivariate
Tuebingen pairs are dropped rather than returned as matrices.

Tuebingen follows LOCI's own protocol in `generate_figures_and_tables.py`:
the blacklist is three discrete pairs (47, 70, 107) plus six multivariate ones
(52--55, 71, 105), leaving **99** pairs, and `pairmeta.txt` supplies both the
cause column and the dataset weight.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'loci', 'data')

DISCRETE_PAIRS = [47, 70, 107]
MULTIVARIATE_PAIRS = [52, 53, 54, 55, 71, 105]
TUEBINGEN_BLACKLIST = DISCRETE_PAIRS + MULTIVARIATE_PAIRS

# name -> (family, subfolder or file stem, number of pairs)
BENCHMARKS = [
    ('AN',        'anlsmn',    'ANLSMN_pairs/AN',          100),
    ('AN-s',      'anlsmn',    'ANLSMN_pairs/AN-s',        100),
    ('LS',        'anlsmn',    'ANLSMN_pairs/LS',          100),
    ('LS-s',      'anlsmn',    'ANLSMN_pairs/LS-s',        100),
    ('MN-U',      'anlsmn',    'ANLSMN_pairs/MN-U',        100),
    ('SIM',       'simulated', 'Benchmark_simulated/SIM',    100),
    ('SIM-c',     'simulated', 'Benchmark_simulated/SIM-c',  100),
    ('SIM-G',     'simulated', 'Benchmark_simulated/SIM-G',  100),
    ('SIM-ln',    'simulated', 'Benchmark_simulated/SIM-ln', 100),
    ('Cha',       'dataverse', 'CE-Cha',                   300),
    ('Multi',     'dataverse', 'CE-Multi',                 300),
    ('Net',       'dataverse', 'CE-Net',                   300),
    ('Tuebingen', 'tuebingen', 'Tuebingen',                108),
]

SYNTHETIC = [b[0] for b in BENCHMARKS if b[0] != 'Tuebingen']
MOOIJ = ['AN', 'AN-s', 'LS', 'LS-s', 'MN-U', 'SIM', 'SIM-c', 'SIM-G', 'SIM-ln']


def _anlsmn(folder, pair_id):
    d = os.path.join(DATA_DIR, folder)
    df = pd.read_csv(f'{d}/pair_{pair_id}.txt', delimiter=',')
    gt = pd.read_csv(f'{d}/pairs_gt.txt', header=None).iloc[pair_id - 1].values[0]
    a, b = df.iloc[:, 1].values, df.iloc[:, 2].values
    if gt == 1:
        return a, b
    if gt == 0:
        return b, a
    raise ValueError(gt)


def _meta(folder):
    return pd.read_csv(os.path.join(DATA_DIR, folder, 'pairmeta.txt'),
                       delim_whitespace=True, header=None,
                       names=['id', 'cs', 'ce', 'es', 'ee', 'weight'],
                       index_col=0).astype(float)


def _pairmeta_style(folder, pair_id, meta):
    df = pd.read_csv(os.path.join(DATA_DIR, folder, f'pair{pair_id:04d}.txt'),
                     delim_whitespace=True, header=None)
    m = meta.loc[pair_id]
    cause = df.iloc[:, int(m['cs']) - 1:int(m['ce'])].values
    effect = df.iloc[:, int(m['es']) - 1:int(m['ee'])].values
    return cause.ravel(), effect.ravel()


def _dataverse(stem, pair_id):
    d = os.path.join(DATA_DIR, 'Dataverse_pairs', stem)
    pairs = pd.read_csv(d + '_pairs.csv')
    target = pd.read_csv(d + '_targets.csv').loc[pair_id - 1, 'Target']
    to_np = lambda s: np.array([float(e) for e in s.strip().split(' ')])
    a, b = to_np(pairs.loc[pair_id - 1, 'A']), to_np(pairs.loc[pair_id - 1, 'B'])
    if target == 1:
        return a, b
    if target == -1:
        return b, a
    raise ValueError(target)


def load(name):
    """Yields (pair_id, cause, effect, weight); cause is always the true cause."""
    entry = next(b for b in BENCHMARKS if b[0] == name)
    _, family, folder, n = entry
    if family == 'anlsmn':
        for i in range(1, n + 1):
            c, e = _anlsmn(folder, i)
            yield i, np.asarray(c, float), np.asarray(e, float), 1.0
    elif family == 'simulated':
        meta = _meta(folder)
        for i in range(1, n + 1):
            c, e = _pairmeta_style(folder, i, meta)
            yield i, c, e, 1.0
    elif family == 'dataverse':
        for i in range(1, n + 1):
            c, e = _dataverse(folder, i)
            yield i, c, e, 1.0
    elif family == 'tuebingen':
        meta = _meta(folder)
        for i in range(1, n + 1):
            if i in TUEBINGEN_BLACKLIST:          # LOCI's own 99-pair subset
                continue
            c, e = _pairmeta_style(folder, i, meta)
            yield i, c, e, float(meta.loc[i, 'weight'])
    else:
        raise ValueError(family)
