#!/usr/bin/env python
"""Benchmark figures: inference time against sample size, optionally with accuracy.

What can be measured
--------------------
Timing a method needs its code, not its published per-pair outputs.  Of the
fifteen columns of the paper's table, these have a runnable implementation in
this repository or in the installed environment:

    Lap^unif / Lap^std / EdgeMass   lap_nlogn.py (O(n log n)) and
                                    laplacian_causality.py (direct O(n^2));
                                    EdgeMass is the same kernel pass as the
                                    energy, so it has no separate cost
    RECI                            cdt.causality.pairwise.RECI, as shipped
    RECI_poly                        RECIPoly, loci_dataset_benchmark/run_reci_poly.py
    IGCI, IGCI_G                    cdt.causality.pairwise.IGCI, the two reference
                                    measures ('uniform' and 'gaussian')
    QPE-k                           qpe_cd/qpek.py (submodule)
    LOCI                            loci/causa/loci.py (vendored)

and these do not, so they are reported as not measurable rather than estimated:

    QCCD, GRCI                      no implementation is vendored with LOCI; the
                                    table's numbers are the authors' released
                                    per-pair outputs in loci/baseline_results
    CAM, RESIT, RESIT_std           same, and both are R packages -- no R
                                    interpreter is present in this environment

`--inventory` prints that check for the machine it is run on, rather than
trusting this docstring.

The figure itself carries no caption text: the caption is written beside it as
`<figure stem>_caption.tex`, a `\caption{...}` block to `\input` into a LaTeX
figure environment.  It is generated from the run, so the stop reasons, the
budget and the machine it was timed on cannot drift from the data.

Protocol
--------
*   One synthetic additive-noise pair per size, the same arrays for every method:
    x ~ N(0,1), y = tanh(2x) + 0.3 eps.  Only the number of points varies.
*   A timed call is one complete inference from the raw arrays -- the method's
    own preprocessing (rank transform, standardization, min-max), both
    directions, and the decision -- exactly as the benchmark runners call it.
*   Cold by default: `lap_nlogn`'s kernel cache is cleared before every repeat,
    so the Toeplitz kernel is built inside the timed region.  The cached path is
    reported as a separate series, because it is what the benchmark actually
    pays when many pairs share one size.
*   Everything runs on the CPU, including the dense estimator, so the columns
    are comparable: no method here has a GPU implementation of its own.
*   Sub-millisecond calls are timed in batches, timeit-style: the warm-up call
    sets how many calls fit in a 50 ms block, that block is timed as a whole and
    divided by the count.  Timing single calls of a few hundred microseconds
    instead charges them whatever transient the previous method left behind --
    on this machine that inflated IGCI at n <= 500 by a factor of two, which is
    visible as a gap between IGCI and IGCI_G that their identical code paths
    cannot produce.
*   The median of several repeats is reported; repeats shrink as the call gets
    slower.  A method stops being timed once one call exceeds `--budget`.

    ./make_benchmark_figures.py               run: results_timing.csv, figure, caption
    ./make_benchmark_figures.py --inventory   only print what can be timed here
    ./make_benchmark_figures.py --summary     re-plot from the saved CSV
    ./make_benchmark_figures.py --budget 30   stop the slow methods sooner
    ./make_benchmark_figures.py --summary --include-accuracy-ranking

The last form adds two panels below the timing one, ranking the same methods by
accuracy with 95% intervals -- on the 12 synthetic suites and on Tuebingen --
read off the benchmark's per-pair outputs in loci_dataset_benchmark/.  Without
that flag the figure is exactly the timing panel, as before.
"""
import argparse
import os
import re
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'loci'))
sys.path.insert(0, os.path.join(HERE, 'qpe_cd'))
sys.path.insert(0, os.path.join(HERE, 'loci_dataset_benchmark'))

M_RANK = 0.07362                 # frozen multiplier, rank transform
SCRATCH = os.environ.get('TMPDIR') or '/tmp'      # where pairs are handed to R
SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
         100000, 200000, 500000, 1000000]

# Categorical slots of the validated palette; one hue per implementation, with
# the variant (device, configuration) carried by the line style instead.
C_LAP = '#2a78d6'        # our estimator; the two implementations share the hue
C_RECI = '#eb6834'
C_QPEK = '#1baf7a'
C_IGCI = '#eda100'
C_LOCI = '#e34948'
C_QCCD = '#008300'
C_RESIT = '#e87ba4'
C_CAM = '#4a3aa7'
C_GRCI = '#7c5b00'       # a darker step of the yellow ramp, distinct from IGCI
INK = '#0b0b0b'
INK2 = '#52514e'
INK3 = '#8a8984'
GRID = '#dcdbd6'


def machine():
    """CPU, memory, OS and interpreter versions of the machine timing this run.

    Every field falls back to something printable, so a run on a machine without
    /proc still produces a caption rather than an exception.
    """
    import platform
    import subprocess

    cpu = platform.processor() or 'unknown CPU'
    cores = None
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name') and cpu == (platform.processor() or
                                                         'unknown CPU'):
                cpu = line.split(':', 1)[1].strip()
            elif line.startswith('cpu cores'):
                cores = int(line.split(':', 1)[1])
    except OSError:
        pass
    threads = os.cpu_count()

    ram = ''
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemTotal'):
                ram = f'{int(line.split()[1]) / 1048576:.1f} GiB'
                break
    except OSError:
        pass

    os_name = platform.platform(terse=True)
    try:
        rel = dict(l.rstrip('\n').split('=', 1) for l in open('/etc/os-release')
                   if '=' in l)
        os_name = rel.get('PRETTY_NAME', os_name).strip('"')
    except OSError:
        pass

    rver = ''
    try:
        rver = subprocess.run([find_rscript(), '-e', 'cat(R.version.string)'],
                              capture_output=True, text=True,
                              timeout=120).stdout.split()[2]
    except Exception:
        pass

    # Vendor decorations read badly in a caption; the model and clock do not.
    cpu = re.sub(r'\s+', ' ', cpu.replace('(R)', '').replace('(TM)', '')
                 .replace(' CPU ', ' ')).strip()
    cpu = re.sub(r'([\d.]+)\s*GHz', r'\1\\,GHz', cpu)
    return dict(cpu=cpu, cores=cores, threads=threads,
                ram=ram.replace(' GiB', r'\,GiB'), os=os_name,
                kernel=platform.release(), python=platform.python_version(),
                r=rver)


def make_pair(n, seed=0):
    """One additive-noise pair of size n; identical arrays for every method."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return x, np.tanh(2 * x) + 0.3 * rng.standard_normal(n)


# --------------------------------------------------------------------------- #
# the methods, each as one complete inference call
# --------------------------------------------------------------------------- #
class Method:
    def __init__(self, key, label, color, style, marker, build, internal=False):
        self.key, self.label = key, label
        self.color, self.style, self.marker = color, style, marker
        self.build = build                    # () -> (call, note) or raises
        self.internal = internal              # the call times itself and returns s
        self.call, self.note, self.error = None, '', ''

    def probe(self):
        try:
            self.call, self.note = self.build()
            return True
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'[:120]
            return False


def build_lap_fast(cached=False):
    """The O(n log n) estimator, with the Toeplitz kernel cold or already built.

    `run` makes one untimed warm-up call per size, so the cached series finds the
    kernel for that n in the lru_cache; the cold series throws it away again at
    the top of every timed call.
    """
    from lap_nlogn import kernel_for, score

    def call(x, y):
        if not cached:
            kernel_for.cache_clear()          # build the kernel inside the call
        return score(x, y, M_RANK)[2]

    return call, ('kernel cache cleared before every repeat' if not cached
                  else 'kernel for this n already built by the warm-up call')


def build_lap_direct():
    """The dense O(n^2) path of run_benchmark.py, on the CPU."""
    import torch
    from scipy.stats import rankdata
    from laplacian_causality import _energy_and_mass, mean_distance, median_bandwidth

    dev = torch.device('cpu')

    def rank(v):
        return rankdata(v, method='ordinal') / (len(v) + 1.0)

    def bandwidth(t):
        s = median_bandwidth(t)
        return s if s > 0 else mean_distance(t)

    def call(x, y):
        X = torch.as_tensor(rank(x), dtype=torch.float64, device=dev).reshape(-1, 1)
        Y = torch.as_tensor(rank(y), dtype=torch.float64, device=dev).reshape(-1, 1)
        ef, mf = _energy_and_mass(X, Y, [M_RANK * bandwidth(X)])
        eb, mb = _energy_and_mass(Y, X, [M_RANK * bandwidth(Y)])
        return ef / mf < eb / mb

    return call, f'CPU, torch threads={torch.get_num_threads()}'


def build_reci(poly=False):
    if poly:
        from run_reci_poly import RECIPoly                     # applies the gelsy patch
        # A fresh instance per call: RECIPoly caches the selected degree so that
        # the two directions of one pair share it, and reusing one instance here
        # would let the repeats of a timed call skip the selection entirely --
        # measuring a cached path no user of the method ever gets.
        return (lambda x, y: RECIPoly().predict_proba((x, y))),\
            "the paper's POLY class, cross-validated degree, min-max"
    import scipy.linalg as sla
    orig = sla.lstsq
    sla.lstsq = lambda a, b, *ar, **kw: orig(
        a, b, *ar, **dict(kw, lapack_driver='gelsy'))
    from cdt.causality.pairwise import RECI
    model = RECI(degree=3)
    return (lambda x, y: model.predict_proba((x, y))),\
        'as shipped (gelsy forced, as in run_reci.py)'


def build_igci(ref_measure):
    from cdt.causality.pairwise import IGCI
    model = IGCI()
    return (lambda x, y: model.predict_proba((x, y), ref_measure=ref_measure,
                                             estimator='entropy')),\
        f'reference measure: {ref_measure}'


def build_qpek():
    from qpek import qpe_k
    basis = [lambda v: np.ones_like(v), lambda v: v, lambda v: v ** 3,
             lambda v: np.tanh(v)]

    def std(v):
        s = v.std()
        return (v - v.mean()) / s if s > 0 else v - v.mean()

    def call(x, y):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            _, f, _, b = qpe_k(std(x).reshape(-1, 1), std(y).reshape(-1, 1),
                               basis_funcs=basis)
        return f > b

    return call, 'the paper default: 20x20 grid, basis [1, y, y^3, tanh y]'


def build_loci():
    import torch
    from causa.loci import loci

    def std(v):
        s = v.std()
        return (v - v.mean()) / s if s > 0 else v - v.mean()

    def call(x, y):
        return float(loci(std(x), std(y))) > 0

    return call, f'packaged defaults, torch threads={torch.get_num_threads()}'


def find_rscript():
    """Rscript to use: $GDECI_RSCRIPT, then PATH, then the gdeci-r conda env."""
    import shutil
    cand = [os.environ.get('GDECI_RSCRIPT'), shutil.which('Rscript'),
            os.path.expanduser('~/anaconda3/envs/gdeci-r/bin/Rscript'),
            os.path.expanduser('~/miniconda3/envs/gdeci-r/bin/Rscript')]
    for c in cand:
        if c and os.path.exists(c):
            return c
    raise RuntimeError('no Rscript found; run baselines/setup_r_baselines.sh')


R_REQUIRES = {'QCCD': ['rvinecopulib', 'statmod', 'quantregForest', 'qrnn'],
              'RESIT': ['gptk', 'kernlab', 'mgcv'],
              'CAM': ['CAM', 'mgcv', 'glmnet', 'mboost'],
              'GRCI': ['RANN', 'DirichletReg']}


def build_r(method):
    """One of the R baselines, timed inside R by baselines/time_baseline.R.

    The call returns its own median seconds over `reps` repeats rather than
    being wall-clocked here, so that R's interpreter start-up -- which no user
    of the method pays per pair -- is not charged to it.  Everything else is
    the same protocol as the Python methods: one untimed warm-up call, then
    repeats inside the one process.
    """
    import subprocess
    rscript = find_rscript()
    driver = os.path.join(HERE, 'baselines', 'time_baseline.R')
    if not os.path.exists(driver):
        raise RuntimeError('baselines/time_baseline.R is missing')
    missing = subprocess.run(
        [rscript, '-e', 'cat(paste(Filter(function(p) !requireNamespace(p, '
         'quietly=TRUE), c(%s)), collapse=","))'
         % ','.join(f'"{p}"' for p in R_REQUIRES[method])],
        capture_output=True, text=True, timeout=300).stdout.strip()
    if missing:
        raise RuntimeError(f'R packages not installed: {missing}')

    def call(x, y, reps, timeout):
        import tempfile
        fd, path = tempfile.mkstemp(suffix='.csv', dir=SCRATCH)
        os.close(fd)
        try:
            np.savetxt(path, np.c_[x, y], delimiter=',', header='x,y', comments='')
            out = subprocess.run([rscript, driver, method, path, str(reps)],
                                 capture_output=True, text=True, timeout=timeout)
            for line in out.stdout.splitlines():
                if line.startswith('seconds '):
                    return float(line.split()[1])
            msg = (out.stderr or out.stdout).strip().splitlines()
            err = next((l for l in reversed(msg) if l.startswith(('Error', 'error'))
                        or 'cannot allocate' in l), msg[-1] if msg else '')
            raise RuntimeError(err[:160] or 'no timing line')
        finally:
            os.path.exists(path) and os.remove(path)

    return call, f'R, {os.path.basename(os.path.dirname(os.path.dirname(rscript)))} env'


def build_heci():
    from causa.heci import HECI
    return (lambda x, y: HECI(x, y)[0]), 'vendored with LOCI, not in the table'


def methods(with_heci=False):
    m = [
        Method('lap_fast', r'GDECI, fast $O(n\log n)$', C_LAP, '-', 'o',
               lambda: build_lap_fast(False)),
        #Method('lap_fast_cached', r'Lap$^{\mathrm{unif}}$, $O(n\log n)$, kernel cached',
        #       C_LAP, ':', 'o', lambda: build_lap_fast(True)),
        Method('lap_direct', r'GDECI, direct $O(n^2)$',
               C_LAP, '--', 's', build_lap_direct),
        Method('reci', 'RECI (CDT default)', C_RECI, '-', '^', lambda: build_reci(False)),
        Method('reci_poly', r'RECI$_{\mathrm{poly}}$', C_RECI, '--', '^',
               lambda: build_reci(True)),
        Method('igci', r'IGCI, IGCI$_G$', C_IGCI, '-', 'D',
               lambda: build_igci('uniform')),
        Method('igci_g', r'IGCI$_G$', C_IGCI, '--', 'D', lambda: build_igci('gaussian')),
        Method('qpek', 'QPE-k', C_QPEK, '-', 'v', build_qpek),
        Method('loci', 'LOCI', C_LOCI, '-', 'P', build_loci),
        Method('qccd', 'QCCD', C_QCCD, '-', '*', lambda: build_r('QCCD'),
               internal=True),
        Method('resit', 'RESIT', C_RESIT, '-', 'h', lambda: build_r('RESIT'),
               internal=True),
        Method('cam', 'CAM', C_CAM, '-', 'X', lambda: build_r('CAM'),
               internal=True),
        Method('grci', 'GRCI', C_GRCI, '-', '<', lambda: build_r('GRCI'),
               internal=True),
    ]
    if with_heci:
        m.append(Method('heci', 'HECI', INK2, '-', 'X', build_heci))
    return m


# Table columns still without a timing here, with the reason.  RESIT_std is the
# same estimator as RESIT on standardized inputs, so it is not timed separately.
NOT_MEASURABLE = []


def inventory(with_heci=False):
    """Print what can and cannot be timed on this machine, and why."""
    ms = methods(with_heci)
    ok = [m for m in ms if m.probe()]
    print(f'{"method":42s} {"timeable":9s} why / how')
    print('-' * 100)
    for m in ms:
        state = 'yes' if m.call else 'NO'
        print(f'{m.key:42s} {state:9s} {m.note if m.call else m.error}')
    for name, why in NOT_MEASURABLE:
        print(f'{name:42s} {"NO":9s} {why}')
    try:
        print(f'\nRscript: {find_rscript()}')
    except Exception as exc:
        print(f'\nRscript: {exc}')
    return ok


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def repeats(seen):
    """How many timed repeats to afford, given what the warm-up call just cost."""
    if seen < 0.02:
        return 7
    if seen < 0.2:
        return 5
    if seen < 2.0:
        return 3
    return 1


def run(ms, sizes, budget, seed):
    rows = []
    for m in ms:
        print(f'\n{m.key}  ({m.note})', flush=True)
        last = None
        for n in sizes:
            x, y = make_pair(n, seed)
            try:
                if m.internal:
                    # The method times itself: one warm-up plus `reps` repeats
                    # inside a single R process, so no interpreter start-up is
                    # charged to it.  Repeats come from the previous size.
                    reps = repeats(last) if last is not None else 3
                    ts = [m.call(x, y, reps, timeout=8 * budget + 300)]
                else:
                    t0 = time.perf_counter()
                    m.call(x, y)                   # warm-up, not timed
                    seen = time.perf_counter() - t0
                    # batch short calls so that one transient cannot dominate
                    inner = 1 if seen >= 0.05 else min(200, max(1, int(0.05 / seen)))
                    ts = []
                    for _ in range(repeats(seen)):
                        t0 = time.perf_counter()
                        for _ in range(inner):
                            m.call(x, y)
                        ts.append((time.perf_counter() - t0) / inner)
            except Exception as exc:
                print(f'  n={n:<8d} failed: {type(exc).__name__}: {exc}'[:110],
                      flush=True)
                rows.append(dict(method=m.key, label=m.label, n=n, seconds=np.nan,
                                 reps=0, budget=budget,
                                 error=f'{type(exc).__name__}: {exc}'[:120]))
                break
            med = float(np.median(ts))
            last = med
            nrep = (repeats(last) if m.internal else len(ts))
            rows.append(dict(method=m.key, label=m.label, n=n, seconds=med,
                             reps=nrep, budget=budget, error=''))
            print(f'  n={n:<8d} {med * 1e3:11.3f} ms   (median of {nrep})',
                  flush=True)
            if med > budget:
                print(f'  stopping: {med:.1f} s exceeds the {budget:g} s budget',
                      flush=True)
                break
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def style(mpl):
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.labelsize': 8,
        'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'axes.edgecolor': INK3, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42,
    })


# Accuracy sources for the optional ranking panels.  A timing curve is an
# implementation; an accuracy row is a method, so the two Lap implementations
# collapse onto one row and IGCI's two reference measures, identical in cost but
# not in accuracy, get one row each.
ACC_KEYS = ['lap_fast', 'loci', 'qpek', 'reci', 'reci_poly', 'igci', 'igci_g',
            'qccd', 'grci', 'cam', 'resit']


def short_labels(keys, ms):
    """Row labels for the ranking panels, taken from the timing labels.

    A timing label names an implementation ('GDECI, fast O(n log n)'); a ranking
    row names the method, so the qualifier after the comma and any trailing
    parenthetical are dropped -- unless that would make two rows read alike, in
    which case both keep the full label.
    """
    short = {k: re.sub(r'\s*\(.*\)$', '', ms[k].label.split(',')[0]).strip()
             for k in keys}
    seen = {}
    for k, v in short.items():
        seen.setdefault(v, []).append(k)
    for v, ks in seen.items():
        if len(ks) > 1:
            for k in ks:
                short[k] = ms[k].label
    return short


def accuracy_table():
    """Per-method accuracy on the synthetic suites and on Tuebingen.

    Read off the benchmark's own per-pair outputs, not off this timing run:
    `results/per_pair.csv` for the proposed score, `results/reci{,_opt}.csv`,
    `results/qpek.csv` and `results_loci/loci.csv` for the methods rerun here,
    and `loci/baseline_results/*.tab` for the published ones.  Intervals follow
    the paper's table: Wilson on Tuebingen, and a stratified bootstrap over
    pairs within each suite for the macro mean.
    """
    bench = os.path.join(HERE, 'loci_dataset_benchmark')
    sys.path.insert(0, bench)
    import datasets as D
    from make_table import baseline_decisions, decisions, positive, wilson

    res = os.path.join(bench, 'results')
    per = pd.read_csv(os.path.join(res, 'per_pair.csv'))
    csvs = {'reci': ('reci.csv', 'reci'), 'reci_poly': ('reci_poly.csv', 'reci_poly'),
            'qpek': ('qpek.csv', 'qpek')}
    frames = {}
    for key, (fn, col) in csvs.items():
        path = os.path.join(res, fn)
        if os.path.exists(path):
            frames[key] = (pd.read_csv(path), col)
    loci_csv = os.path.join(bench, 'results_loci', 'loci.csv')
    if os.path.exists(loci_csv):
        frames['loci'] = (pd.read_csv(loci_csv), 'loci')

    tab = {'igci': 'IGCI', 'igci_g': 'IGCI_G', 'qccd': 'QCCD', 'grci': 'GRCI',
           'cam': 'CAM', 'resit': 'RESIT'}
    keys = list(ACC_KEYS)
    correct = {}
    for name, _, _, _ in D.BENCHMARKS:
        d = per[per.benchmark == name].set_index('pair_id')
        col = {}
        col['lap_fast'] = decisions(d)['Lap_unif'].astype(float)
        base = baseline_decisions(name).reindex(d.index)
        for key, bcol in tab.items():
            col[key] = base[bcol].astype(float)
        for key, (frame, c) in frames.items():
            f = frame[frame.benchmark == name].set_index('pair_id')
            col[key] = positive(f[c]).reindex(d.index)
        correct[name] = pd.DataFrame(col).reindex(columns=keys)

    synth = list(D.SYNTHETIC)
    acc = {b: correct[b].mean(0) for b in correct}
    out = {}
    tue = correct['Tuebingen']
    for key in keys:
        a = float(acc['Tuebingen'][key])
        n = int(tue[key].notna().sum())
        out[key] = dict(tue=(a,) + wilson(a, n) if n else (np.nan,) * 3)

    macro = {k: float(np.nanmean([acc[b][k] for b in synth])) for k in keys}
    rng = np.random.default_rng(0)
    arrs = {b: correct[b].values.astype(float) for b in synth}
    boot = np.empty((2000, len(keys)))
    for i in range(len(boot)):
        boot[i] = np.nanmean([np.nanmean(a[rng.integers(0, len(a), len(a))], axis=0)
                              for a in arrs.values()], axis=0)
    q = np.percentile(boot, [2.5, 97.5], axis=0)
    for j, key in enumerate(keys):
        out[key]['synth'] = (macro[key], q[0, j], q[1, j])
    return out


def draw_ranking(ax, rows, ms, labels, title, xlim):
    """Methods ordered by accuracy, each with its 95% interval."""
    rows = sorted(rows, key=lambda r: -r[1][0])
    for i, (key, (a, lo, hi)) in enumerate(rows):
        m = ms[key]
        ax.plot([lo, hi], [-i, -i], color=m.color, lw=1.1, solid_capstyle='butt',
                zorder=3)
        ax.plot([a], [-i], marker=m.marker, ms=3.4, mew=0, color=m.color, zorder=4)
    ax.axvline(0.5, color=INK3, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks([-i for i in range(len(rows))])
    ax.set_yticklabels([labels[k] for k, _ in rows], fontsize=6.5)
    ax.set_ylim(-len(rows) + 0.4, 0.6)
    ax.set_xlim(*xlim)
    ax.set_xlabel('accuracy', labelpad=1)
    ax.set_title(title, loc='left', pad=3.5, fontsize=7.5, color=INK)
    ax.grid(True, axis='x', color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)


# Reference size for panel (d): the size of most benchmark pairs, so the
# accuracy and the cost shown for a method are the ones a user of these
# benchmarks would actually see.
PARETO_N = 1000


def pareto_front(points):
    """Indices of the non-dominated (time, accuracy) points.

    A point is dominated when some other point is at least as fast and at least
    as accurate, and strictly better in one of the two.  What survives is the
    set of methods for which no other method is a free improvement.
    """
    keep = []
    for i, (t, a) in enumerate(points):
        if not (np.isfinite(t) and np.isfinite(a)):
            continue
        if any(np.isfinite(t2) and np.isfinite(a2)
               and t2 <= t and a2 >= a and (t2 < t or a2 > a)
               for j, (t2, a2) in enumerate(points) if j != i):
            continue
        keep.append(i)
    return keep


def draw_pareto(ax, rows, ms, labels, title, ylim):
    """Accuracy against cost at PARETO_N, with the frontier as a staircase.

    `rows` is (key, seconds, (accuracy, lo, hi)).  The staircase is drawn
    between the non-dominated points: from each one, cost rises to the next
    before accuracy does, so the line traces what the frontier actually
    promises -- no method inside the step is both faster and better.
    """
    points = [(t, a) for _, t, (a, _, _) in rows]
    front = sorted((points[i] for i in pareto_front(points)), key=lambda p: p[0])
    if len(front) > 1:
        xs, ys = [], []
        for t, a in front:
            if xs:
                xs.append(t)
                ys.append(ys[-1])
            xs.append(t)
            ys.append(a)
        ax.plot(xs, ys, color=INK3, lw=0.8, ls=(0, (4, 2)), zorder=2,
                label='Pareto frontier')
        ax.legend(loc='lower right', fontsize=6.0, handlelength=2.0,
                  borderaxespad=0.3, framealpha=0.85, edgecolor='none')
    for key, t, (a, lo, hi) in rows:
        if not (np.isfinite(t) and np.isfinite(a)):
            continue
        m = ms[key]
        ax.plot([t, t], [lo, hi], color=m.color, lw=1.0, solid_capstyle='butt',
                zorder=3)
        ax.plot([t], [a], marker=m.marker, ms=3.4, mew=0, color=m.color, zorder=4)
        ax.annotate(labels[key], (t, a), textcoords='offset points',
                    xytext=(4.0, 3.0), fontsize=5.8, color=m.color, zorder=5)
    ax.set_xscale('log')
    ax.axhline(0.5, color=INK3, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f'time per pair at $n=10^{{{int(np.log10(PARETO_N))}}}$ (s)',
                  labelpad=1)
    ax.set_ylabel('accuracy', labelpad=2)
    ax.set_title(title, loc='left', pad=3.5, fontsize=7.5, color=INK)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def pareto_rows(df, acc, ms, which):
    """(key, seconds at PARETO_N, interval) for every method timed at that size."""
    at_n = (df[(df.n == PARETO_N) & df.seconds.notna()]
            .set_index('method').seconds)
    return [(k, float(at_n[k]), acc[k][which])
            for k in ACC_KEYS
            if k in acc and k in ms and k in at_n.index
            and np.isfinite(acc[k][which][0])]


def stop_facts(df, ms):
    """Why each curve ends: the budget, the size grid, and the two stop lists."""
    d_ok = df[df.seconds.notna()]
    budget = float(df.budget.dropna().iloc[0]) if 'budget' in df else 60.0
    grid_end = int(df.n.max())
    err = df[df.error.fillna('') != ''].set_index('method')

    def sci(v):
        v = int(v)
        e = len(str(v)) - 1
        m = v / 10 ** e
        return f'{m:g}\\times 10^{{{e}}}' if m != 1 else f'10^{{{e}}}'

    over, oom = [], []
    for key in d_ok.method.unique():
        if key not in ms or key == 'igci_g':
            continue
        d = d_ok[d_ok.method == key].sort_values('n')
        if key in err.index:
            oom.append(f'{ms[key].label} at $n={sci(err.loc[key, "n"])}$')
        elif int(d.n.iloc[-1]) != grid_end:
            over.append(f'{ms[key].label} at $n={sci(d.n.iloc[-1])}$')
    return dict(budget=budget, grid_end=sci(grid_end), over=over, oom=oom)


def caption_tex(df, with_heci=False, ranking=False, pareto=False):
    """The figure's caption as LaTeX, with the facts read off the run itself."""
    ms = {m.key: m for m in methods(with_heci)}
    f = stop_facts(df, ms)
    mc = machine()
    r_keys = [k for k in ('qccd', 'resit', 'cam', 'grci') if k in set(df.method)]

    lead = ('(a) Time' if ranking else 'Time')
    text = [lead + r' to decide one causal pair, both directions, against sample '
            r'size. Median of repeated calls, short ones timed in batches, on one '
            r'synthetic additive-noise pair per size, the same arrays for every '
            r"method; a call is a complete inference from the raw arrays, the "
            r"method's own preprocessing included."]
    if f['over']:
        tail = ((' or ran out of memory (' + '; '.join(f['oom']) + ')')
                if f['oom'] else '')
        text.append(f"A curve that stops short of $n={f['grid_end']}$ ends where "
                    f"one call first passed the {f['budget']:g}\\,s budget ("
                    + '; '.join(f['over']) + ')' + tail + '.')
    elif f['oom']:
        text.append(f"A curve that stops short of $n={f['grid_end']}$ ends where "
                    'the method ran out of memory (' + '; '.join(f['oom']) + ').')
    text.append(r'IGCI and IGCI$_G$ share a curve: the same computation after a '
                r'different $O(n)$ rescaling, measured within 10\,\% at every size.')
    if r_keys:
        text.append(', '.join(ms[k].label for k in r_keys)
                    + r" are the authors' own R implementations, timed inside R so "
                    r'that interpreter start-up is not charged to them; '
                    r"GRCI's one \texttt{Rfast} primitive is replaced by its "
                    r'base-R equivalent.')
    if NOT_MEASURABLE:
        text.append('Not timed, no implementation available: '
                    + ', '.join(n for n, _ in NOT_MEASURABLE) + '.')

    if ranking:
        text.append(r'(b, c) The same methods ranked by accuracy, with $95\%$ '
                    r'intervals, on the 12 synthetic suites (macro mean over '
                    r'suites, stratified bootstrap over pairs within each) and on '
                    r'the 99 Tuebingen pairs (Wilson); the dashed line is chance. '
                    r'Accuracies are the per-pair outputs of the benchmark, not of '
                    r'this timing run; the fast and the direct implementation of '
                    r'the proposed score share one row, as they compute the same '
                    r'score.')
    if pareto:
        text.append(r'(d) The accuracies of (b) and (c) against the time per '
                    f'pair at $n=10^{{{int(np.log10(PARETO_N))}}}$ from (a), '
                    r'with vertical bars the same intervals and the dashed line '
                    r'the Pareto frontier: no method below or to the right of it '
                    r'is both faster and more accurate than one on it.')
    core = (f"{mc['cores']} cores, {mc['threads']} threads"
            if mc['cores'] else f"{mc['threads']} threads")
    env = f"Python {mc['python']}" + (f", R {mc['r']}" if mc['r'] else '')
    text.append(f"All timings on one machine, CPU only: {mc['cpu']} ({core}), "
                f"{mc['ram']} RAM, {mc['os']} (kernel {mc['kernel']}), {env}.")

    body = '\n  '.join(t.replace('%', r'\%').replace(r'\\%', r'\%')
                       for t in text)
    return (f'% Generated by {os.path.basename(__file__)} -- do not edit by hand.\n'
            '% Use as:\n'
            '%   \\begin{figure*}[t]\\centering\n'
            '%     \\includegraphics[width=\\textwidth]{figures/gdeci_timing.pdf}\n'
            '%     \\input{figures/gdeci_timing_caption}\n'
            '%   \\end{figure*}\n'
            '\\caption{%\n  ' + body + '%\n}\n'
            '\\label{fig:timing}\n')


def plot(df, out, size, with_heci=False, ranking=False, pareto=False):
    import matplotlib.pyplot as plt
    style(matplotlib)
    ms = {m.key: m for m in methods(with_heci)}
    d_ok = df[df.seconds.notna()]

    if pareto:                       # (a) on top, (b, c) middle, (d) bottom
        fig = plt.figure(figsize=size)
        ax = fig.add_axes([0.075, 0.705, 0.66, 0.270])
    elif ranking:
        fig = plt.figure(figsize=size)
        ax = fig.add_axes([0.075, 0.560, 0.66, 0.385])
    else:
        fig, ax = plt.subplots(figsize=size)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='major', color=GRID, lw=0.5, zorder=0)
    ax.grid(True, which='minor', color=GRID, lw=0.3, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

    # Legend order follows the curves at their right-hand ends: slowest first,
    # so the list reads in the same order as the lines the eye lands on.
    # IGCI's two reference measures are the same computation with a different
    # O(n) scaling step in front of it, and they were measured to agree within
    # 10% at every size (median 3%), so they share one curve rather than being
    # drawn twice; both are kept in the CSV.
    d_ok = d_ok[d_ok.method != 'igci_g']
    ends = d_ok.sort_values('n').groupby('method').seconds.last().sort_values(
        ascending=False)
    handles = []
    for key in ends.index:
        d = d_ok[d_ok.method == key].sort_values('n')
        m = ms.get(key)
        if m is None or d.empty:
            continue
        line, = ax.plot(d.n, d.seconds, color=m.color, ls=m.style, lw=1.2,
                        marker=m.marker, ms=3.2, mew=0, label=m.label, zorder=3)
        handles.append(line)

    # Reference slopes, anchored on the right end of the curve they describe so
    # they lie along the asymptote rather than through the small-n overhead.
    def guide(key, power, label, decades=1.6):
        d = d_ok[d_ok.method == key].sort_values('n')
        if d.empty:
            return
        n1, t1 = float(d.n.iloc[-1]), float(d.seconds.iloc[-1])
        ng = np.array([n1 / 10 ** decades, n1], float)
        ax.plot(ng, t1 * (ng / n1) ** power, color=INK3, lw=0.7, ls=(0, (4, 2)),
                zorder=1)
        ax.text(n1 * 1.15, t1, label, color=INK3, fontsize=6.5, va='center',
                ha='left', zorder=4,
                bbox=dict(facecolor='white', alpha=0.85, pad=1.2, lw=0))

    guide('lap_fast', 1.0, r'$\propto n$')
    guide('lap_direct', 2.0, r'$\propto n^2$')

    ax.set_xlabel('pair sample size $n$')
    ax.set_ylabel('inference time per pair (s)')
    ax.set_xlim(d_ok.n.min() * 0.75, d_ok.n.max() * 1.45)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.015, 1.02),
              fontsize=6.8, handlelength=2.2, labelspacing=0.5, borderaxespad=0)

    if ranking:
        acc = accuracy_table()
        # Only methods the timing panel draws: the ranking panels must not
        # introduce a colour the legend does not explain.
        # A method whose per-pair outputs are not in the repository comes back
        # with a NaN accuracy; drop it rather than drawing an empty row.
        keys = [k for k in ACC_KEYS if k in acc and k in ms
                and np.isfinite(acc[k]['synth'][0]) and np.isfinite(acc[k]['tue'][0])]
        missing = [k for k in ACC_KEYS if k in ms and k not in keys]
        if missing:
            print('note: no per-pair accuracy for '
                  + ', '.join(ms[k].label for k in missing)
                  + '; omitted from the accuracy panels')
        labels = short_labels(keys, ms)
        rows = [(k, acc[k]) for k in keys]
        span = [v for k, a in rows for v in (a['synth'][1:] + a['tue'][1:])]
        xlim = (min(span) - 0.04, max(span) + 0.04)
        band = (0.395, 0.250) if pareto else (0.075, 0.365)
        ax_s = fig.add_axes([0.105, band[0], 0.345, band[1]])
        ax_t = fig.add_axes([0.615, band[0], 0.345, band[1]])
        draw_ranking(ax_s, [(k, a['synth']) for k, a in rows], ms, labels,
                     '(b)  12 synthetic suites, macro mean', xlim)
        draw_ranking(ax_t, [(k, a['tue']) for k, a in rows], ms, labels,
                     '(c)  Tuebingen, 99 pairs', xlim)
        ax.set_title('(a)  inference time', loc='left', pad=3.5, fontsize=7.5,
                     color=INK)

        if pareto:
            # Panel (d) reuses (b) and (c)'s accuracies against the cost the
            # timing panel measured, so the three panels cannot disagree.
            ax_ps = fig.add_axes([0.105, 0.055, 0.345, 0.245])
            ax_pt = fig.add_axes([0.615, 0.055, 0.345, 0.245])
            for axis, which, title in (
                    (ax_ps, 'synth', '(d)  12 synthetic suites, macro mean'),
                    (ax_pt, 'tue', '(d)  Tuebingen, 99 pairs')):
                prows = pareto_rows(df, acc, ms, which)
                if not prows:
                    axis.set_axis_off()
                    continue
                lo = min(r[2][1] for r in prows)
                hi = max(r[2][2] for r in prows)
                pad = 0.08 * (hi - lo) + 0.02
                draw_pareto(axis, prows, ms, labels, title, (lo - pad, hi + pad))
    else:
        fig.subplots_adjust(left=0.075, right=0.735, top=0.97, bottom=0.135)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out)
    png = os.path.splitext(out)[0] + '.png'
    fig.savefig(png, dpi=400)
    tex = os.path.splitext(out)[0] + '_caption.tex'
    with open(tex, 'w') as fh:
        fh.write(caption_tex(df, with_heci, ranking, pareto))
    print(f'wrote {os.path.abspath(out)}\n      {os.path.abspath(png)}'
          f'\n      {os.path.abspath(tex)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--inventory', action='store_true',
                    help='only report which methods can be timed here')
    ap.add_argument('--summary', action='store_true',
                    help='re-plot from the saved CSV without timing anything')
    ap.add_argument('--budget', type=float, default=120.0,
                    help='stop a method once one call takes longer (s, default 120)')
    ap.add_argument('--max-n', type=int, default=1000000)
    ap.add_argument('--csv', default='results_timing.csv')
    ap.add_argument('--out', default=os.path.join('figures', 'gdeci_timing.pdf'))
    ap.add_argument('--size', default='', metavar='WxH',
                    help='figure size in inches (default: full text width, '
                         'taller with --include-accuracy-ranking)')
    ap.add_argument('--include-accuracy-ranking', action='store_true',
                    help='add the accuracy-ranking panels below the timing panel')
    ap.add_argument('--include-pareto', action='store_true',
                    help='also add panel (d), accuracy against time per pair at '
                         f'n={PARETO_N} with the Pareto frontier; implies '
                         '--include-accuracy-ranking.  Together these give '
                         "Figure 1 of the paper")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--only', default='', metavar='KEY[,KEY]',
                    help='time just these methods and merge them into the CSV')
    ap.add_argument('--with-heci', action='store_true',
                    help='also time HECI, which is vendored but not in the table')
    args = ap.parse_args()
    ranking = args.include_accuracy_ranking or args.include_pareto
    size = (tuple(float(t) for t in args.size.lower().split('x'))
            if args.size else
            (7.16, 8.8) if args.include_pareto else
            (7.16, 6.0) if ranking else (7.16, 3.4))

    if args.inventory:
        inventory(args.with_heci)
        return

    if args.summary:
        df = pd.read_csv(args.csv)
        plot(df, args.out, size, args.with_heci, ranking, args.include_pareto)
        return

    print('=== what can be timed on this machine ===')
    ok = inventory(args.with_heci)
    if args.only:
        keys = [k.strip() for k in args.only.split(',')]
        ok = [m for m in ok if m.key in keys]
        if not ok:
            sys.exit(f'none of {keys} is timeable here')

    sizes = [n for n in SIZES if n <= args.max_n]
    print(f'\n=== timing {len(ok)} methods over n = {sizes[0]} ... {sizes[-1]} ===')
    t0 = time.time()
    df = run(ok, sizes, args.budget, args.seed)
    if args.only and os.path.exists(args.csv):
        # merge into what is already measured, replacing only these methods
        old = pd.read_csv(args.csv)
        df = pd.concat([old[~old.method.isin(df.method.unique())], df],
                       ignore_index=True)
    df.to_csv(args.csv, index=False)
    print(f'\nwrote {os.path.abspath(args.csv)} ({len(df)} rows, '
          f'{time.time() - t0:.0f} s)')
    plot(df, args.out, size, args.with_heci, ranking, args.include_pareto)


if __name__ == '__main__':
    main()
