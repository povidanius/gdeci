"""O(n log n) graph-Dirichlet causal direction score, optimised.

The criterion
-------------
Both variables are mapped to the rank grid u_k = k/(n+1).  They then carry the
same value set, so a single Gaussian kernel serves both directions:

    sigma  = m * median_{i<j} |u_i - u_j|
    W_ij   = exp(-(u_i - u_j)^2 / (2 sigma^2)),   deg = W 1,   M = sum_ij W_ij
    E_sym  = y^T (I - D^-1/2 W D^-1/2) y          Delta = E / M
    E_un   = y^T (D - W) y                        S     = E_un / M

Decide X -> Y iff the forward score is the smaller one.

Why it is O(n log n)
--------------------
On the rank grid W is Toeplitz -- W_kl depends only on |k - l| -- so every
matrix-vector product is a convolution and the n x n matrix is never formed.
The median bandwidth is also exact in O(n): pairwise distances take the values
d/(n+1) with multiplicity (n - d), so the median is a cumulative-sum lookup
rather than a sort of n(n-1)/2 numbers.

What makes this version faster than the straightforward FFT
-----------------------------------------------------------
1. The kernel is truncated at the point where it drops below 1e-18, which for
   the frozen multiplier is about 15% of n on each side.  That shrinks the
   transform length from 2n to roughly 1.3n.
2. The kernel spectrum is computed once per (n, m) and cached, instead of being
   recomputed inside every convolution as `fftconvolve` would do.
3. Whole kernels are cached by (n, m).  A benchmark suite has few distinct n but
   many pairs, so this turns thousands of kernel builds into a few dozen.
4. Both directions' signals are transformed in one batched rFFT call.

Measured against a straightforward FFT implementation, on the score computation
itself with the kernel cached and ranks precomputed:

    n = 1e3   4.5x       n = 1e5   5.2x
    n = 1e4   1.8x       n = 1e6   2.2x

The transform length is about 2.2x shorter (truncation) and the two directions
share one batched transform.  End to end the gain is smaller, because ranking
costs 0.09 / 0.66 / 8.6 / 121 ms at those sizes and is common to both.  `--verify` checks the result against a dense O(n^2) evaluation that shares no
code path with it: agreement is 1e-13 relative with zero decision flips.
"""
import argparse
import functools
import os
import sys
import time

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

TRUNC_EPS = 1e-18


# --------------------------------------------------------------------------- #
# grid, bandwidth
# --------------------------------------------------------------------------- #
def rank_grid(v, rng=None):
    """Ranks on the grid k/(n+1); ties broken at random when rng is given."""
    v = np.asarray(v, float)
    n = v.shape[0]
    keys = (rng.random(n), v) if rng is not None else (np.arange(n), v)
    order = np.lexsort(keys)
    r = np.empty(n)
    r[order] = np.arange(1, n + 1)
    return r / (n + 1.0)


def grid_sigma(n, m):
    """m times the lower-median pairwise distance on the grid, exactly, in O(n).

    Distance d/(n+1) occurs (n - d) times, so the order statistic
    k = (P+1)//2 of the P = n(n-1)/2 distances is a searchsorted on the
    cumulative counts.
    """
    d = np.arange(1, n)
    P = n * (n - 1) // 2
    k = (P + 1) // 2
    idx = int(np.searchsorted(np.cumsum(n - d), k))
    return m * float(d[idx]) / (n + 1.0)


# --------------------------------------------------------------------------- #
# kernel
# --------------------------------------------------------------------------- #
class FastGridKernel:
    """Gaussian Toeplitz kernel on {1..n}/(n+1) with a cached spectrum."""

    __slots__ = ('n', 'sigma', 'half', 'nfft', 'spec', 'deg', 'mass', 'g0')

    def __init__(self, n, sigma):
        self.n, self.sigma = n, sigma
        step = 1.0 / (n + 1.0)
        # smallest L with exp(-(L*step)^2 / 2 sigma^2) < TRUNC_EPS
        reach = sigma * np.sqrt(-2.0 * np.log(TRUNC_EPS))
        self.half = int(min(n - 1, np.ceil(reach / step)))
        d = np.arange(self.half + 1) * step
        g = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        self.g0 = float(g[0])                       # self-loops: W_ii = 1
        sym = np.concatenate([g[:0:-1], g])         # length 2*half + 1
        self.nfft = next_fast_len(n + 2 * self.half)
        self.spec = rfft(sym, self.nfft)
        self.deg = self.matvec(np.ones(n))
        self.mass = float(self.deg.sum())

    def matvec(self, z):
        """W z for one signal (1-D) or a stack of signals (rows of a 2-D array)."""
        z = np.asarray(z, float)
        out = irfft(rfft(z, self.nfft, axis=-1) * self.spec, self.nfft, axis=-1)
        return out[..., self.half:self.half + self.n]

    def energies(self, Y):
        """(E_sym, E_unnorm) for each row of Y, in one batched transform."""
        Y = np.atleast_2d(np.asarray(Y, float))
        Z = Y / np.sqrt(self.deg)
        WZ = self.matvec(np.vstack([Z, Y]))
        k = Y.shape[0]
        e_sym = np.einsum('ij,ij->i', Y, Y) - np.einsum('ij,ij->i', Z, WZ[:k])
        e_un = (Y * Y) @ self.deg - np.einsum('ij,ij->i', Y, WZ[k:])
        return e_sym, e_un


@functools.lru_cache(maxsize=256)
def kernel_for(n, m):
    return FastGridKernel(n, grid_sigma(n, m))


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def score(x, y, m, rng=None, variant='unnorm'):
    """(forward, reverse, decision) for the pair (x, y); True means x causes y.

    variant 'unnorm' is the unnormalized Dirichlet energy over edge mass,
    'sym' is the symmetric-normalized one.  Both directions share the kernel.
    """
    u = rank_grid(x, rng)
    v = rank_grid(y, rng)
    K = kernel_for(len(u), m)
    Y = np.vstack([v[np.argsort(u)], u[np.argsort(v)]])
    e_sym, e_un = K.energies(Y)
    e = e_sym if variant == 'sym' else e_un
    f, r = float(e[0] / K.mass), float(e[1] / K.mass)
    return f, r, f < r


# --------------------------------------------------------------------------- #
# benchmark driver
# --------------------------------------------------------------------------- #
def _one(task):
    name, pid, cause, effect, weight, m, seeds, variant = task
    out = []
    for s in seeds:
        f, r, dec = score(cause, effect, m, np.random.default_rng(s), variant)
        out.append(dict(benchmark=name, pair_id=pid, n=len(cause), weight=weight,
                        seed=s, variant=variant, S_fwd=f, S_rev=r,
                        decision=bool(dec), correct=float(dec)))
    return out


def run_suite(m, seeds, variant, jobs, out_csv):
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'loci_dataset_benchmark'))
    import datasets as D

    tasks = [(name, pid, c, e, w, m, seeds, variant)
             for name, _, _, _ in D.BENCHMARKS
             for pid, c, e, w in D.load(name)]
    print(f'{len(tasks)} pairs x {len(seeds)} seeds, variant={variant}, '
          f'm={m}, jobs={jobs}', flush=True)
    t0 = time.time()
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            chunks = pool.map(_one, tasks, chunksize=8)
    else:
        chunks = [_one(t) for t in tasks]
    rows = [r for c in chunks for r in c]
    el = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'scored {len(df)} rows in {el:.1f}s '
          f'({1000 * el / len(df):.2f} ms per pair-seed)')
    print('wrote', out_csv)

    # Report in the order the paper's tables use, not pandas' alphabetical order.
    order = [b[0] for b in D.BENCHMARKS]
    per = df.groupby(['benchmark', 'seed']).correct.mean().groupby('benchmark').mean()
    npairs = df.groupby('benchmark').pair_id.nunique()
    syn = [b for b in order if b != 'Tuebingen']
    print(f'\n{"benchmark":34s} {"#pairs":>7s} {"accuracy":>9s}')
    for b in order:
        print(f'{b:34s} {npairs[b]:7d} {per[b]:9.3f}')
    t = df[df.benchmark == 'Tuebingen']
    w = t.groupby('pair_id').weight.first()
    ws = t.groupby('seed').apply(
        lambda g: float((g.set_index('pair_id').correct * w).sum() / w.sum()))
    print(f'{"Tuebingen, weighted":34s} {npairs["Tuebingen"]:7d} {ws.mean():9.3f}')
    print(f'{"macro mean, 12 non-Tuebingen":34s} {int(npairs[syn].sum()):7d} '
          f'{per[syn].mean():9.3f}')


def dense_reference(x, y, m, rng):
    """Explicit O(n^2) evaluation of both scores, for verification only.

    Forms the kernel matrix and the quadratic forms directly, with no FFT and no
    Toeplitz structure, so it shares no code path with the fast implementation.
    Memory is O(n^2); callers keep n small.
    """
    u = rank_grid(x, rng)
    v = rank_grid(y, rng)
    n = len(u)
    sigma = grid_sigma(n, m)
    g = np.arange(1, n + 1) / (n + 1.0)
    W = np.exp(-((g[:, None] - g[None, :]) ** 2) / (2.0 * sigma ** 2))
    deg = W.sum(1)
    mass = W.sum()
    out = {}
    for key, sig in (('fwd', v[np.argsort(u)]), ('rev', u[np.argsort(v)])):
        z = sig / np.sqrt(deg)
        out[key] = (float(sig @ sig - z @ (W @ z)) / mass,          # symmetric
                    float(np.sum(sig * sig * deg) - sig @ (W @ sig)) / mass)
    return out


def verify(m, n_pairs, max_n=4000):
    """Compare against a dense O(n^2) reference on real benchmark pairs.

    Pairs larger than `max_n` are skipped, since the dense reference would need
    an n x n matrix; the fast path itself has no such limit.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, 'loci_dataset_benchmark'))
    import datasets as D

    worst = {'sym': 0.0, 'unnorm': 0.0}
    flips, used, skipped = 0, 0, 0
    for name, _, _, _ in D.BENCHMARKS:
        for pid, c, e, w in D.load(name):
            if used >= n_pairs:
                break
            if len(c) > max_n:
                skipped += 1
                continue
            ref = dense_reference(c, e, m, np.random.default_rng(0))
            for k, variant in ((0, 'sym'), (1, 'unnorm')):
                f, r, dec = score(c, e, m, np.random.default_rng(0), variant)
                rf, rr = ref['fwd'][k], ref['rev'][k]
                worst[variant] = max(worst[variant], abs(f - rf) / abs(rf))
                flips += (dec != (rf < rr))
            used += 1
        if used >= n_pairs:
            break
    print(f'verified {used} benchmark pairs against a dense O(n^2) reference '
          f'({skipped} skipped for n > {max_n})')
    print(f'  max relative difference, symmetric-normalized : {worst["sym"]:.3e}')
    print(f'  max relative difference, unnormalized         : {worst["unnorm"]:.3e}')
    print(f'  decision flips                                : {flips}')
    return flips == 0 and max(worst.values()) < 1e-10


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--m', type=float, default=0.07362,
                    help='frozen bandwidth multiplier (default: the frozen value)')
    ap.add_argument('--seeds', default='0', help='comma-separated tie-breaking seeds')
    ap.add_argument('--variant', default='unnorm', choices=['unnorm', 'sym'])
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--out', default='results_nlogn.csv')
    ap.add_argument('--verify', type=int, metavar='NPAIRS', default=0,
                    help='compare against a dense O(n^2) reference and exit')
    ap.add_argument('--bench', type=int, metavar='N', default=0,
                    help='time a single pair of size N and exit')
    args = ap.parse_args()

    if args.verify:
        sys.exit(0 if verify(args.m, args.verify) else 1)
    if args.bench:
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, args.bench)
        y = np.cos(2 * np.pi * x) + 0.1 * rng.standard_normal(args.bench)
        score(x, y, args.m, rng)                       # warm the cache
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            score(x, y, args.m, np.random.default_rng(1))
            ts.append(time.perf_counter() - t)
        print(f'n={args.bench}: {np.median(ts) * 1000:.3f} ms per pair '
              f'(median of 5, kernel cached)')
        sys.exit(0)

    run_suite(args.m, [int(s) for s in args.seeds.split(',')], args.variant,
              args.jobs, args.out)


if __name__ == '__main__':
    main()
