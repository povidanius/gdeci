import argparse
import numpy as np
import torch
from scipy.stats import rankdata


# --------------------------------------------------------------------------- #
# bandwidth
# --------------------------------------------------------------------------- #
def _count_pairs_leq(xs, t):
    """#{(i<j) : |x_i - x_j| <= t} for a sorted 1-D tensor xs."""
    j = torch.searchsorted(xs, xs + t, right=True)          # first index > x_i + t
    i = torch.arange(xs.numel(), device=xs.device)
    return (j - i - 1).clamp_(min=0).sum().item()           # count of j > i within t


def median_bandwidth(x, exact_max_n=4000):
    """Median of the pairwise Euclidean distances (median heuristic).

    For 1-D data the exact median over all n(n-1)/2 pairs is obtained by
    bisecting the pair-counting function, which costs O(n log n) per step and
    therefore uses every sample even for the largest pairs. For d > 1 (or small
    n) it falls back to an explicit pdist.
    """
    x = x.reshape(x.shape[0], -1).double()
    n = x.shape[0]
    if x.shape[1] > 1 or n <= exact_max_n:
        d = torch.pdist(x)
        return d.median().item()

    xs, _ = torch.sort(x.reshape(-1))
    m = n * (n - 1) // 2
    k = (m + 1) // 2                       # lower median rank, matching torch.median
    lo, hi = 0.0, float(xs[-1] - xs[0])
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _count_pairs_leq(xs, mid) >= k:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-12 * max(hi, 1.0):
            break
    return hi


# --------------------------------------------------------------------------- #
# normalized Laplacian regularizer
# --------------------------------------------------------------------------- #
def _energy_and_mass(x, y, sigmas, block=2048):
    """(y^T L_sym y, edge mass sum_ij W_ij) for W = sum_k exp(-d^2/2 sigma_k^2).

    Computed in row blocks, so the n x n kernel is never materialized.
    Two passes: degrees first, then the quadratic form.
    """
    x = x.reshape(x.shape[0], -1).float()
    y = y.reshape(-1).double()
    n = x.shape[0]
    gammas = [1.0 / (2.0 * float(s) ** 2) for s in sigmas]

    sq = (x * x).sum(1)

    def kernel_block(s, e):
        d2 = (sq[s:e, None] + sq[None, :] - 2.0 * (x[s:e] @ x.T)).clamp_(min=0)
        W = torch.exp(d2 * -gammas[0])
        for g in gammas[1:]:
            W += torch.exp(d2 * -g)
        return W

    deg = torch.empty(n, dtype=torch.float64, device=x.device)
    for s in range(0, n, block):
        e = min(s + block, n)
        deg[s:e] = kernel_block(s, e).sum(1).double()

    z = y / torch.sqrt(deg)                       # D^{-1/2} y ; deg >= K (self-loops)
    zwz = 0.0                                     # z^T W z
    zf = z.float()
    for s in range(0, n, block):
        e = min(s + block, n)
        zwz += (zf[s:e] * (kernel_block(s, e) @ zf)).double().sum().item()

    return (y * y).sum().item() - zwz, deg.sum().item()


def laplacian_score(x, y, sigmas, block=2048, aggregation='sum_scaled', score='raw'):
    """Dirichlet energy of y on the normalized-Laplacian graph of x.

    aggregation (only matters for K > 1 bandwidths):
      'sum_scaled' -- one graph per scale; each scale's energy is divided by that
                      scale's edge mass and the results are summed. The division
                      equalizes the scales; without it the widest kernel carries
                      almost all the edge mass and dominates the sum.
      'kernel_sum' -- the kernels are summed into a single graph first, and the
                      normalized Laplacian is formed on that sum.

    score: 'raw' returns the energy, 'rayleigh' divides by y^T y (scale-invariant
    in y; identical decisions to 'raw' whenever y is standardized).
    """
    y1 = y.reshape(-1).double()
    if aggregation == 'kernel_sum':
        total = _energy_and_mass(x, y, sigmas, block)[0]
    elif aggregation == 'sum_scaled':
        total = sum(e / m for e, m in
                    (_energy_and_mass(x, y, [s], block) for s in sigmas))
    else:
        raise ValueError(aggregation)
    if score == 'rayleigh':
        return total / (y1 * y1).sum().item()
    return total


def mean_distance(x, block=2048):
    """Mean pairwise distance, in row blocks (fallback bandwidth for tied data)."""
    x = x.reshape(x.shape[0], -1)
    n = x.shape[0]
    tot = 0.0
    for s in range(0, n, block):
        tot += torch.cdist(x[s:min(s + block, n)], x).sum().item()
    return tot / (n * (n - 1))


def standardize(v):
    v = v - v.mean()
    s = v.std()
    return v / s if s > 0 else v


def preprocess(v, device, mode='standardize'):
    t = torch.as_tensor(np.asarray(v), dtype=torch.float64, device=device).reshape(-1, 1)
    if mode == 'none':                            # raw data, no rescaling
        return t
    if mode == 'standardize':
        return standardize(t)
    if mode == 'minmax':
        lo, hi = t.min(), t.max()
        return (t - lo) / (hi - lo) if hi > lo else t - lo
    if mode == 'rank':                            # uniform marginals
        r = np.asarray(rankdata(np.asarray(v), method='average')) / (len(v) + 1.0)
        return torch.as_tensor(r, dtype=torch.float64, device=device).reshape(-1, 1)
    raise ValueError(mode)


def bandwidth(v, block=2048):
    """Median heuristic, with a mean-distance fallback when >50% of pairs tie."""
    s = median_bandwidth(v)
    return s if s > 0 else mean_distance(v, block)


def infer_direction(x, y, device, block=2048, mode='standardize', scales=(1.0,),
                    aggregation='sum_scaled', score='raw'):
    """Returns (S(x->y), S(y->x), sigma_med_x, sigma_med_y).

    `scales` are multipliers of the per-graph median bandwidth; more than one
    means a summed multi-scale kernel.
    """
    x = preprocess(x, device, mode)
    y = preprocess(y, device, mode)
    mx = bandwidth(x, block)
    my = bandwidth(y, block)
    return (laplacian_score(x, y, [c * mx for c in scales], block, aggregation, score),
            laplacian_score(y, x, [c * my for c in scales], block, aggregation, score),
            mx, my)


# --------------------------------------------------------------------------- #
# Tuebingen cause-effect pairs
# --------------------------------------------------------------------------- #
# The evaluation protocol used in the paper: of the 108 Tuebingen pairs, six are
# multivariate and three have a categorical or heavily tied variable.  Excluding
# the nine leaves 99, which is the set the LOCI baselines were run on --
# verified identical to the index of loci/baseline_results/Tuebingen.tab.
MULTIVARIATE_PAIRS = [52, 53, 54, 55, 71, 105]
DISCRETE_PAIRS = [47, 70, 107]
TUEBINGEN_BLACKLIST = sorted(MULTIVARIATE_PAIRS + DISCRETE_PAIRS)


def load_tuebingen(folder='./pairs', protocol='99'):
    """(id, x, y, weight) per pair, x always the cause.

    protocol='99'  the paper's protocol: drop the six multivariate pairs and the
                   three discrete ones (47, 70, 107).  This is the default.
    protocol='102' drop only the multivariate pairs.  Retained so that the
                   earlier 102-pair analyses in analysis/ stay reproducible; it
                   is not the protocol any reported result uses.
    """
    if protocol not in ('99', '102'):
        raise ValueError(f"protocol must be '99' or '102', got {protocol!r}")
    drop = TUEBINGEN_BLACKLIST if protocol == '99' else MULTIVARIATE_PAIRS
    meta = np.loadtxt(folder + '/pairmeta.txt')
    out = []
    for row in meta:
        pid, c1, c2, e1, e2, w = row
        if c1 != c2 or e1 != e2:                  # multivariate pair -> skip
            continue
        if int(pid) in drop:
            continue
        d = np.loadtxt(f'{folder}/pair{int(pid):04d}.txt')
        cause = d[:, int(c1) - 1]
        effect = d[:, int(e1) - 1]
        out.append((int(pid), cause, effect, float(w)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', default='./pairs')
    ap.add_argument('--protocol', default='99', choices=['99', '102'],
                    help="Tuebingen evaluation protocol: '99' excludes the six "
                         "multivariate pairs and the three discrete ones "
                         "(47, 70, 107), as in the paper; '102' excludes only "
                         "the multivariate pairs.")
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--block', type=int, default=2048)
    ap.add_argument('--preprocess', default='standardize',
                    choices=['minmax', 'none', 'standardize', 'rank'])
    ap.add_argument('--aggregation', default='sum_scaled',
                    choices=['sum_scaled', 'kernel_sum'],
                    help='at a single scale these are simply the two reported '
                         'scores: sum_scaled = E / edge mass (Lap^std_avg), '
                         'kernel_sum = E alone (Lap^std_raw). The names refer to '
                         'how several scales would be combined, which no '
                         'reported experiment does.')
    ap.add_argument('--score', default='raw', choices=['raw', 'rayleigh'])
    ap.add_argument('--scales', default='0.05535',
                    help='comma-separated multipliers of the median bandwidth. '
                         'Every reported experiment uses a SINGLE scale; the '
                         'default is the frozen multiplier for standardized '
                         'marginals (0.07362 is the one for rank-transformed '
                         'ones). Passing several is supported but is not a '
                         'configuration any published number comes from.')
    args = ap.parse_args()

    device = torch.device(args.device)
    scales = tuple(float(t) for t in args.scales.split(','))
    pairs = load_tuebingen(args.folder, args.protocol)

    correct, w_correct, w_total = 0, 0.0, 0.0
    print(f'{"pair":>5} {"n":>7} {"med_x":>8} {"med_y":>8} '
          f'{"S(x->y)":>12} {"S(y->x)":>12}  ok   acc')
    for k, (pid, cx, cy, w) in enumerate(pairs, 1):
        s_xy, s_yx, sx, sy = infer_direction(cx, cy, device, args.block,
                                             args.preprocess, scales,
                                             args.aggregation, args.score)
        ok = s_xy < s_yx                          # x is the true cause by construction
        correct += ok
        w_total += w
        w_correct += w * ok
        print(f'{pid:5d} {len(cx):7d} {sx:8.4f} {sy:8.4f} '
              f'{s_xy:12.6g} {s_yx:12.6g}  {"+" if ok else "-"}  {correct / k:.3f}')

    n = len(pairs)
    print(f'\npreprocessing    : {args.preprocess}, '
          f'sigma_k = {list(scales)} x median  (K={len(scales)}), '
          f'{args.aggregation}, {args.score}')
    excluded = TUEBINGEN_BLACKLIST if args.protocol == '99' else MULTIVARIATE_PAIRS
    print(f'protocol         : Tuebingen-{args.protocol}, excluded {excluded}')
    print(f'pairs            : {n}')
    print(f'accuracy         : {correct}/{n} = {correct / n:.4f}')
    print(f'weighted accuracy: {w_correct / w_total:.4f}')


if __name__ == '__main__':
    main()
