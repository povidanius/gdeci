"""LOCI (Immer et al.) on every benchmark pair, at the packaged defaults.

Calls `causa.loci.loci(x, y)` from the vendored repository with no arguments
beyond the data, i.e. the neural heteroscedastic estimator (5000 epochs, both
directions) followed by the HSIC residual-independence test; a positive score
means x -> y.  Inputs are standardized first, which is what every gin config in
`loci/configs/*.gin` specifies (`<Benchmark>.preprocessor = @StandardScaler()`).

The random seed is set inside `het_fit_nn` (711), so results do not depend on
worker scheduling.

Cost.  About 47 s per pair on one core, so the 1899 pairs are run across a
process pool with `torch.set_num_threads(1)` per worker to avoid
oversubscription.  HSIC is a dense O(n^2) numpy computation holding roughly six
n-by-n float64 matrices -- 12 GB at the largest Tuebingen pair -- so pairs are
grouped by size and the large ones get a smaller pool.  Results are appended
per pair, so the run is resumable and a failure costs one pair rather than all.
"""
import os

# Must precede the first numpy import: HSIC is dense numpy, so without this each
# worker spawns its own MKL/OpenMP thread pool and the 7 processes fight for the
# same 8 cores.  Left unset, the pool ran about 4.7x slower per pair than a
# single unshared process.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'loci'))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, 'results_loci', 'loci.csv')
COLUMNS = ['benchmark', 'pair_id', 'n', 'weight', 'loci', 'seconds', 'error']


def _work(task):
    import torch
    torch.set_num_threads(1)
    from sklearn.preprocessing import StandardScaler
    from causa.loci import loci
    name, pid, cause, effect, weight = task
    prep = lambda v: StandardScaler().fit_transform(
        np.asarray(v, float).reshape(-1, 1)).ravel()
    t0 = time.time()
    try:
        score = float(loci(prep(cause), prep(effect)))
        err = ''
    except Exception as exc:                       # e.g. MemoryError in HSIC
        score, err = float('nan'), f'{type(exc).__name__}: {exc}'[:200]
    return dict(benchmark=name, pair_id=pid, n=len(cause), weight=weight,
                loci=score, seconds=round(time.time() - t0, 1), error=err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=7)
    args = ap.parse_args()

    import datasets as D
    from multiprocessing import Pool

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.benchmark, prev.pair_id))
        print(f'resuming: {len(done)} pairs already scored')
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(OUT, index=False)

    tasks = []
    for name, _, _, _ in D.BENCHMARKS:
        for pid, cause, effect, weight in D.load(name):
            if (name, pid) not in done:
                tasks.append((name, pid, cause, effect, weight))
    tasks.sort(key=lambda t: len(t[2]))

    # smaller pools for the memory-hungry pairs
    groups = [([t for t in tasks if len(t[2]) <= 6000], args.workers),
              ([t for t in tasks if 6000 < len(t[2]) <= 12000], 2),
              ([t for t in tasks if len(t[2]) > 12000], 1)]

    total, k, t0 = len(tasks), 0, time.time()
    print(f'{total} pairs to score')
    for group, workers in groups:
        if not group:
            continue
        print(f'--- {len(group)} pairs, n in [{len(group[0][2])}, '
              f'{len(group[-1][2])}], {workers} workers ---', flush=True)
        with Pool(workers) as pool:
            for row in pool.imap_unordered(_work, group):
                k += 1
                pd.DataFrame([row])[COLUMNS].to_csv(OUT, mode='a', header=False,
                                                    index=False)
                if row['error'] or k % 25 == 0:
                    el = time.time() - t0
                    print(f'[{k}/{total}] {row["benchmark"]}/{row["pair_id"]} '
                          f'n={row["n"]} score={row["loci"]:+.5f} '
                          f'{row["seconds"]}s  elapsed {el/60:.0f}m '
                          f'eta {el/k*(total-k)/60:.0f}m {row["error"]}', flush=True)
    print('done in', round((time.time() - t0) / 60, 1), 'min')


if __name__ == '__main__':
    main()
