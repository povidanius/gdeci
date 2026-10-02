"""Test whether Proposition 1's smoothness term carries causal signal.

The experiment uses the unnormalized Laplacian L = D - W and uniform marginals
from Proposition 1, and separates the zero-bandwidth limit of the score from its
positive-bandwidth part by a regression-free bandwidth extrapolation:

    Delta_sigma = E_sigma / M_sigma, with the diagonal (self-loops) included in M,
    is evaluated at sigma = s * sigma_0 for s in {0.1, 0.2, ..., 1.0}, and
    Delta_sigma = B + a*sigma is fitted on s in {0.1, ..., 0.5}.

The intercept B estimates the conditional-variance (Bayes) component, and the
frozen score minus B the positive-bandwidth (smoothness) component.  A
quadratic-in-sigma fit and the older sigma^2-only fit are sensitivity checks.

Protocol, matching Table 1(b):
  * ties are broken at random by `lap_nlogn.rank_grid(v, default_rng(seed))`,
    X ranked before Y from one generator, seeds 0-9; accuracies are means over
    the seeds, and paired counts (corrections, spoils, McNemar) are reported for
    seed 0 and as means over seeds.  Tie-free pairs are seed-invariant;
  * decisions are strict: X -> Y iff the forward value is smaller, so an exact
    tie counts as incorrect.  No numerical tolerance is applied;
  * the full score at s = 1 must reproduce the Table 1(b) GDECI decisions pair by
    pair; this is asserted.

In addition, the accuracy of the full score at every bandwidth s in
{0.2, ..., 1.0} is reported exactly as in `ablation_v2/ablation_v2.py` (same
kernels, same tie seeds, same figure), and checked against that script's
per-pair decisions when its output is present.

The causal labels are never used in the decomposition; they are used only after
all components are fixed, to score decisions.  The linear expansion is not
guaranteed for every benchmark distribution, and at s = 0.1 the lag bandwidth is
h ~ 2 at n = 1000, so the smallest fitted bandwidths are within a few grid
steps.  No extrapolated component is clipped.

`decision`, `wilson` and `accuracy_stats` are shared helpers imported by
`run_nw_reci_experiment.py` and `run_orientation_flip_test.py`; they are kept
unchanged, tolerance included, for those scripts.  This experiment itself uses
strict decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(ROOT, "loci_dataset_benchmark")
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, ROOT)

import datasets as D
from lap_nlogn import FastGridKernel, grid_sigma, kernel_for, rank_grid


FROZEN_M = 0.07362
FRACTIONS = np.round(np.arange(1, 11) / 10.0, 1)       # 0.1, 0.2, ..., 1.0
FIT_FRACTIONS = FRACTIONS[:5]                            # 0.1, ..., 0.5
CURVE_FRACTIONS = FRACTIONS[1:]                          # 0.2, ..., 1.0 (as ablation_v2)
MAIN_SPEC = "sigma_linear"
SPECS = ("sigma_linear", "sigma_quadratic", "sigma2_linear", "pilot")
SPEC_POINTS = {"sigma_linear": 5, "sigma_quadratic": 6, "sigma2_linear": 5}
TIE_SEEDS = tuple(range(10))
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20270917
TAGS = [str(f).replace(".", "p") for f in FRACTIONS]


# --------------------------------------------------------------------------- #
# shared helpers (imported by other experiments; keep unchanged)
# --------------------------------------------------------------------------- #
def decision(margin, scale=1.0, rtol=1e-10, atol=1e-12):
    if not np.isfinite(margin):
        return 0
    tolerance = atol + rtol * max(float(scale), 1.0)
    return 0 if abs(margin) <= tolerance else (1 if margin > 0 else -1)


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return [np.nan, np.nan]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z / denominator * np.sqrt(p * (1 - p) / total + z * z / (4 * total ** 2))
    return [float(max(0, center - half)), float(min(1, center + half))]


def accuracy_stats(margins, scale=None):
    margins = np.asarray(margins, float)
    scale = np.ones_like(margins) if scale is None else np.asarray(scale, float)
    decisions = np.array([decision(m, s) for m, s in zip(margins, scale)])
    covered = decisions != 0
    correct = decisions == 1
    return {
        "accuracy_all": float(correct.mean()),
        "accuracy_covered": float(correct[covered].mean()) if covered.any() else np.nan,
        "coverage": float(covered.mean()),
        "correct": int(correct.sum()),
        "covered": int(covered.sum()),
        "pairs": int(len(margins)),
        "wilson_95": wilson(int(correct.sum()), len(margins)),
    }, decisions


# --------------------------------------------------------------------------- #
# extrapolation
# --------------------------------------------------------------------------- #
def extrapolate(q, fractions, spec):
    """Unconstrained zero-bandwidth intercept, with no causal-label input.

    The predictor is fraction = sigma / sigma_0; for each pair sigma_0 is fixed,
    so this rescaling changes the slope but not the fitted intercept.
      sigma_linear     1, sigma          on the 5 smallest fractions (0.1..0.5)
      sigma_quadratic  1, sigma, sigma^2 on the 6 smallest (sensitivity)
      sigma2_linear    1, sigma^2        on the 5 smallest (legacy comparison)
      pilot            the smallest-bandwidth value itself
    """
    q = np.asarray(q, float)
    fractions = np.asarray(fractions, float)
    if (fractions.ndim != 1 or not len(fractions) or
            q.shape != (2, len(fractions))):
        raise ValueError("q must have shape (2, len(fractions)); fractions must be nonempty")
    if (not np.all(np.isfinite(q)) or not np.all(np.isfinite(fractions)) or
            np.any(fractions <= 0) or np.any(np.diff(fractions) <= 0)):
        raise ValueError("Scores must be finite; fractions must be finite, positive and increasing")
    if spec == "pilot":
        return q[:, 0].copy(), np.full(2, np.nan)
    if spec not in SPEC_POINTS:
        raise ValueError(spec)
    k = SPEC_POINTS[spec]
    if len(fractions) < k:
        raise ValueError(f"{spec} needs at least {k} bandwidths")
    x = fractions[:k]
    if spec == "sigma2_linear":
        design = np.column_stack([np.ones(k), x ** 2])
    elif spec == "sigma_linear":
        design = np.column_stack([np.ones(k), x])
    else:
        design = np.column_stack([np.ones(k), x, x ** 2])
    y = q[:, :k].T
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    ss_res = np.sum(residual ** 2, axis=0)
    ss_tot = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), 1.0)
    return coefficients[0], r2


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def kernel_at(n, fraction):
    """The ablation_v2 kernels: exactly the Table 1 kernel at s = 1."""
    if fraction == 1.0:
        return kernel_for(n, FROZEN_M)
    return FastGridKernel(n, float(fraction) * grid_sigma(n, FROZEN_M))


def score_pair(task):
    """Path scores Delta_s = E_s / M_s (diagonal included) for every tie seed.

    Tie-free pairs are seed-invariant and are scored once (seed 0, marked
    `seed_invariant`); `expand_seeds` replicates them.
    """
    name, pair_id, cause, effect, weight = task
    cause = np.asarray(cause, float)
    effect = np.asarray(effect, float)
    n = len(cause)
    ties = len(np.unique(cause)) < n or len(np.unique(effect)) < n
    kernels = [kernel_at(n, f) for f in FRACTIONS]
    rows = []
    for seed in (TIE_SEEDS if ties else TIE_SEEDS[:1]):
        rng = np.random.default_rng(seed)
        u = rank_grid(cause, rng)
        v = rank_grid(effect, rng)
        signals = np.vstack([v[np.argsort(u)], u[np.argsort(v)]])
        row = dict(benchmark=name, pair_id=pair_id, n=n, weight=weight, seed=seed,
                   seed_invariant=not ties, frozen_multiplier=FROZEN_M)
        for tag, kernel in zip(TAGS, kernels):
            q = kernel.energy(signals) / kernel.mass
            row[f"q_fwd_{tag}"], row[f"q_rev_{tag}"] = float(q[0]), float(q[1])
        rows.append(row)
    return rows


def expand_seeds(paths):
    """Replicate seed-invariant rows so that every pair has all tie seeds."""
    fixed = paths[paths.seed_invariant]
    rest = paths[~paths.seed_invariant]
    reps = [fixed.assign(seed=s) for s in TIE_SEEDS]
    return (pd.concat([rest] + reps, ignore_index=True)
            .sort_values(["benchmark", "pair_id", "seed"]).reset_index(drop=True))


def refresh_extrapolations(df):
    """Components, margins and strict decisions from the stored path scores."""
    columns = [[f"q_{d}_{tag}" for tag in TAGS] for d in ("fwd", "rev")]
    missing = sorted(set(c for cs in columns for c in cs) - set(df.columns))
    if missing:
        raise ValueError("CSV lacks bandwidth-path scores; rerun without --reuse-results: "
                         + ", ".join(missing))
    if "frozen_multiplier" in df and not np.allclose(
            df.frozen_multiplier, FROZEN_M, rtol=0, atol=1e-14):
        raise ValueError("CSV uses a different frozen multiplier; recompute the scores")
    df = df.copy()
    paths = np.stack([df[c].to_numpy(dtype=float) for c in columns], axis=1)
    if not np.all(np.isfinite(paths)):
        raise ValueError("CSV bandwidth-path scores contain nonfinite values")
    df["full_fwd"], df["full_rev"] = paths[:, 0, -1], paths[:, 1, -1]
    df["full_margin"] = df.full_rev - df.full_fwd
    df["full_correct"] = (df.full_margin > 0).astype(float)
    for j, tag in enumerate(TAGS):
        df[f"margin_{tag}"] = paths[:, 1, j] - paths[:, 0, j]
        df[f"correct_{tag}"] = (df[f"margin_{tag}"] > 0).astype(float)
    for spec in SPECS:
        fitted = [extrapolate(q, FRACTIONS, spec) for q in paths]
        bayes = np.array([f[0] for f in fitted])
        r2 = np.array([f[1] for f in fitted])
        smooth = paths[:, :, -1] - bayes
        for direction, index in (("fwd", 0), ("rev", 1)):
            df[f"{spec}_bayes_{direction}"] = bayes[:, index]
            df[f"{spec}_smooth_{direction}"] = smooth[:, index]
            df[f"{spec}_r2_{direction}"] = r2[:, index]
        df[f"{spec}_bayes_margin"] = bayes[:, 1] - bayes[:, 0]
        df[f"{spec}_smooth_margin"] = smooth[:, 1] - smooth[:, 0]
        df[f"{spec}_total_margin"] = df[f"{spec}_bayes_margin"] + df[f"{spec}_smooth_margin"]
        df[f"{spec}_bayes_correct"] = (df[f"{spec}_bayes_margin"] > 0).astype(float)
        df[f"{spec}_smooth_correct"] = (df[f"{spec}_smooth_margin"] > 0).astype(float)
    return df


def correct_columns():
    return (["full_correct"] + [f"correct_{t}" for t in TAGS] +
            [f"{s}_{c}_correct" for s in SPECS for c in ("bayes", "smooth")])


def pair_means(seeds):
    """Per-pair correctness averaged over tie seeds (the Table 1(b) protocol)."""
    return (seeds.groupby(["benchmark", "pair_id"], sort=False)
            .agg({**{c: "mean" for c in correct_columns()}, "weight": "first",
                  "n": "first"}).reset_index())


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def component_summary(means, seed0, prefix):
    """Accuracies (seed means), Wilson on seed 0, and paired changes."""
    out = {}
    for label, col in (("full", "full_correct"), ("bayes_only", f"{prefix}_bayes_correct"),
                       ("smoothness_only", f"{prefix}_smooth_correct")):
        k0 = int(seed0[col].sum())
        out[label] = {"accuracy_all": float(means[col].mean()),
                      "accuracy_seed0": float(seed0[col].mean()),
                      "correct_seed0": k0, "pairs": int(len(means)),
                      "wilson_95": wilson(k0, len(seed0))}
    full0 = seed0.full_correct.to_numpy() == 1
    bayes0 = seed0[f"{prefix}_bayes_correct"].to_numpy() == 1
    corrections = int((full0 & ~bayes0).sum())
    spoils = int((~full0 & bayes0).sum())
    discordant = corrections + spoils
    fm, bm = means.full_correct.to_numpy(), means[f"{prefix}_bayes_correct"].to_numpy()
    smooth0 = seed0[f"{prefix}_smooth_correct"].to_numpy() == 1
    out.update({
        "full_minus_bayes_accuracy": out["full"]["accuracy_all"] - out["bayes_only"]["accuracy_all"],
        "corrections": corrections,
        "spoils": spoils,
        "net_corrections": corrections - spoils,
        "corrections_mean_over_seeds": float(np.clip(fm - bm, 0, None).sum()),
        "spoils_mean_over_seeds": float(np.clip(bm - fm, 0, None).sum()),
        "mcnemar_exact_two_sided_p": (float(binomtest(corrections, discordant, 0.5).pvalue)
                                      if discordant else 1.0),
        "smoothness_above_chance_binomial_p": float(binomtest(
            int(smooth0.sum()), len(smooth0), 0.5, alternative="greater").pvalue),
    })
    return out


def stratified_bootstrap(means, bayes_col, full_col="full_correct", suites=None,
                         pooled=False):
    """Resample pairs within suites; macro averages suite means, pooled averages pairs."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arrays = []
    for name in (D.SYNTHETIC if suites is None else suites):
        subset = means[means.benchmark == name]
        if subset.empty:
            raise ValueError(f"Missing benchmark for the bootstrap: {name}")
        arrays.append(subset[[full_col, bayes_col]].to_numpy())

    def aggregate(samples):
        if pooled:
            total = sum(float(x.sum(axis=0) @ [1, -1]) for x in samples)
            return total / sum(len(x) for x in samples)
        return float(np.mean([x.mean(axis=0) @ [1, -1] for x in samples]))

    diffs = np.empty(BOOTSTRAPS)
    for b in range(BOOTSTRAPS):
        diffs[b] = aggregate([x[rng.integers(0, len(x), len(x))] for x in arrays])
    return {
        "estimate": aggregate(arrays),
        "ci_95": [float(x) for x in np.percentile(diffs, [2.5, 97.5])],
        "bootstrap_probability_positive": float(np.mean(diffs > 0)),
        "repetitions": BOOTSTRAPS,
    }


def fit_window_sensitivity(seeds, seed0):
    """Refit the linear intercept on the k smallest bandwidths, for every k.

    The primary analysis uses k = len(FIT_FRACTIONS). Two points determine the
    line exactly, so k >= 3 is the smallest window with any residual left to
    judge the fit by.
    """
    columns = [[f"q_{d}_{tag}" for tag in TAGS] for d in ("fwd", "rev")]
    stack = np.stack([seeds[c].to_numpy(dtype=float) for c in columns], axis=1)
    full_margin = stack[:, 1, -1] - stack[:, 0, -1]
    full_correct = full_margin > 0
    is_seed0 = (seeds.seed.to_numpy() == 0)
    rows = []
    for k in range(3, len(FRACTIONS) + 1):
        x = FRACTIONS[:k]
        design = np.column_stack([np.ones(k), x])
        y = stack[:, :, :k].reshape(-1, k).T                 # (k, 2 * rows)
        coefficients = np.linalg.pinv(design) @ y
        residual = y - design @ coefficients
        ss_res = np.sum(residual ** 2, axis=0)
        ss_tot = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), 1.0)
        bayes = coefficients[0].reshape(-1, 2)
        bayes_margin = bayes[:, 1] - bayes[:, 0]
        bayes_correct = bayes_margin > 0
        smooth_correct = (full_margin - bayes_margin) > 0
        corrections = int((full_correct & ~bayes_correct)[is_seed0].sum())
        spoils = int((~full_correct & bayes_correct)[is_seed0].sum())
        discordant = corrections + spoils
        rows.append({
            "points": k,
            "largest_fitted_fraction": float(x[-1]),
            "bayes_only": float(bayes_correct.mean()),
            "smoothness_only": float(smooth_correct.mean()),
            "full": float(full_correct.mean()),
            "corrections": corrections,
            "spoils": spoils,
            "net_corrections": corrections - spoils,
            "mcnemar_exact_two_sided_p": (float(binomtest(corrections, discordant, 0.5).pvalue)
                                          if discordant else 1.0),
            "median_r2": float(np.median(r2)),
            "negative_bayes_estimates_seed0": int((bayes[np.repeat(is_seed0, 2)
                                                   .reshape(-1, 2)[:, 0]] < 0).sum()),
        })
    return rows


def suite_sign_test(means, bayes_col, suites):
    """Sign test on the per-suite accuracy gain of the full score."""
    gains = np.array([means.loc[means.benchmark == n, "full_correct"].mean() -
                      means.loc[means.benchmark == n, bayes_col].mean() for n in suites])
    positive, negative = int((gains > 1e-12).sum()), int((gains < -1e-12).sum())
    return {"positive": positive, "negative": negative,
            "ties": int(len(gains) - positive - negative),
            "one_sided_p": (float(binomtest(positive, positive + negative, 0.5,
                                            alternative="greater").pvalue)
                            if positive + negative else 1.0)}


def macro(means, col):
    return float(np.mean([means.loc[means.benchmark == name, col].mean()
                          for name in D.SYNTHETIC]))


def weighted(means, col):
    tue = means[means.benchmark == "Tuebingen"]
    return float((tue[col] * tue.weight).sum() / tue.weight.sum())


def accuracy_vs_sigma(means):
    """Accuracy of the full score at every bandwidth (ablation_v2 covers s >= 0.2)."""
    rows = []
    for f in FRACTIONS:
        col = f"correct_{str(f).replace('.', 'p')}"
        row = {"s": float(f), "macro12": macro(means, col),
               "pooled": float(means[col].mean()),
               "tuebingen": float(means.loc[means.benchmark == "Tuebingen", col].mean())}
        for name in [entry[0] for entry in D.BENCHMARKS]:
            row[name] = float(means.loc[means.benchmark == name, col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(seeds, means, seed0, crosschecks):
    summary = {
        "design": {
            "pairs": int(len(means)),
            "datasets": int(means.benchmark.nunique()),
            "frozen_multiplier": FROZEN_M,
            "bandwidth_fractions": FRACTIONS.tolist(),
            "main_extrapolation": MAIN_SPEC,
            "primary_model": "Delta_sigma = E_sigma/M_sigma = B + a*sigma + o(sigma), "
                             "M including the diagonal",
            "primary_fit_fractions": FIT_FRACTIONS.tolist(),
            "quadratic_sensitivity_fit_fractions": FRACTIONS[:6].tolist(),
            "accuracy_vs_sigma_fractions": FRACTIONS.tolist(),
            "ablation_v2_crosscheck_fractions": CURVE_FRACTIONS.tolist(),
            "legacy_comparison": "sigma2_linear",
            "component_clipping": False,
            "bayes_population_range": [0.0, 1.0 / 12.0],
            "tie_policy": "random tie-breaking, seeds 0-9 (Table 1(b) protocol); "
                          "accuracies are means over seeds",
            "decision_rule": "strict: forward < reverse; exact ties count as incorrect",
            "tie_seeds": list(TIE_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": BOOTSTRAPS,
            "causal_labels_used_for_estimation": False,
        },
        "hardware": {"platform": platform.platform(),
                     "python": platform.python_version(),
                     "numpy": np.__version__},
        "specifications": {},
        "per_dataset": {},
        "crosschecks": crosschecks,
    }
    all_suites = [entry[0] for entry in D.BENCHMARKS]
    for prefix in SPECS:
        summary["specifications"][prefix] = component_summary(means, seed0, prefix)
        summary["specifications"][prefix]["stratified_bootstrap_pooled_all"] = \
            stratified_bootstrap(means, f"{prefix}_bayes_correct",
                                 suites=all_suites, pooled=True)
        summary["specifications"][prefix]["stratified_bootstrap_macro12"] = \
            stratified_bootstrap(means, f"{prefix}_bayes_correct")
    for name in [entry[0] for entry in D.BENCHMARKS]:
        summary["per_dataset"][name] = component_summary(
            means[means.benchmark == name], seed0[seed0.benchmark == name], MAIN_SPEC)
    syn = means.benchmark != "Tuebingen"
    syn0 = seed0.benchmark != "Tuebingen"
    summary["main_synthetic_1800"] = component_summary(means[syn], seed0[syn0], MAIN_SPEC)
    cols = {"bayes_only": f"{MAIN_SPEC}_bayes_correct",
            "smoothness_only": f"{MAIN_SPEC}_smooth_correct", "full": "full_correct"}
    summary["fit_window_sensitivity"] = fit_window_sensitivity(seeds, seed0)
    summary["main_all_pairs"] = dict(summary["specifications"][MAIN_SPEC])
    summary["main_all_pairs"]["suite_level_sign_test"] = suite_sign_test(
        means, cols["bayes_only"], all_suites)
    summary["main_macro12"] = {k: macro(means, c) for k, c in cols.items()}
    summary["main_macro12"]["suite_level_sign_test"] = suite_sign_test(
        means, cols["bayes_only"], D.SYNTHETIC)
    summary["tuebingen_weighted"] = {k: weighted(means, c) for k, c in cols.items()}
    curve = accuracy_vs_sigma(means)
    summary["accuracy_vs_sigma"] = curve[["s", "macro12", "pooled", "tuebingen"]].to_dict("list")

    b = np.concatenate([seed0[f"{MAIN_SPEC}_bayes_fwd"], seed0[f"{MAIN_SPEC}_bayes_rev"]])
    s = np.concatenate([seed0[f"{MAIN_SPEC}_smooth_fwd"], seed0[f"{MAIN_SPEC}_smooth_rev"]])
    summary["diagnostics"] = {
        "negative_smoothness_direction_estimates": int((s < 0).sum()),
        "negative_bayes_direction_estimates": int((b < 0).sum()),
        "bayes_above_uniform_variance_estimates": int((b > 1.0 / 12.0).sum()),
        "minimum_bayes_estimate": float(b.min()),
        "maximum_bayes_estimate": float(b.max()),
        "direction_estimates": int(len(s)),
        "median_small_path_r2": float(np.nanmedian(np.concatenate(
            [seed0[f"{MAIN_SPEC}_r2_fwd"], seed0[f"{MAIN_SPEC}_r2_rev"]]))),
        "lag_bandwidth_at_smallest_fraction": {
            str(int(n)): float(FRACTIONS[0] * grid_sigma(int(n), FROZEN_M) * (n + 1))
            for n in sorted(means.n.unique()) if n in (1000, 1500)},
    }
    return summary, curve


def run_crosschecks(means, seeds):
    """The full score must equal Table 1(b); the curve must equal ablation_v2."""
    out = {}
    table_path = os.path.join(BENCH_DIR, "results", "per_pair.csv")
    if os.path.exists(table_path):
        table = pd.read_csv(table_path)
        if "unif_correct" in table:
            m = means.merge(table[["benchmark", "pair_id", "unif_correct"]],
                            on=["benchmark", "pair_id"], validate="one_to_one")
            mism = int((np.abs(m.full_correct - m.unif_correct) > 1e-12).sum())
            out["table_1b"] = {"matched_pairs": int(len(m)), "mismatches": mism}
            if len(m) == len(means) and mism:
                raise SystemExit(f"full score differs from Table 1(b) on {mism} pairs")
    ab_path = os.path.join(ROOT, "ablation_v2", "results", "per_pair_seeds.csv")
    if os.path.exists(ab_path):
        ab = pd.read_csv(ab_path).rename(columns={"suite": "benchmark"})
        m = seeds.merge(ab, on=["benchmark", "pair_id", "seed"], validate="one_to_one")
        mism = 0
        for f in CURVE_FRACTIONS:
            mism += int((m[f"correct_{str(f).replace('.', 'p')}"] != m[f"Rs_{f:.1f}"]).sum())
        out["ablation_v2_curve"] = {"matched_pair_seeds": int(len(m)),
                                    "fractions": CURVE_FRACTIONS.tolist(),
                                    "mismatches": mism}
        if len(m) == len(seeds) and mism:
            raise SystemExit(f"accuracy-vs-sigma decisions differ from ablation_v2 on "
                             f"{mism} pair-seed-bandwidths")
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
INK, INK2, GRID = "#1f1f1e", "#5d5c56", "#dcdbd6"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def make_figures(means, summary, curve, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    order = [entry[0] for entry in D.BENCHMARKS]
    cols = [(f"{MAIN_SPEC}_bayes_correct", "estimated conditional-variance component", BLUE),
            (f"{MAIN_SPEC}_smooth_correct", "positive-bandwidth component alone", ORANGE),
            ("full_correct", "full GDECI", GREEN)]

    # component accuracy by benchmark (paper Figure 2)
    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    x = np.arange(len(order) + 2)
    width = 0.8 / len(cols)
    for j, (col, label, color) in enumerate(cols):
        vals = ([means.loc[means.benchmark == n, col].mean() for n in order] +
                [macro(means, col), float(means[col].mean())])
        ax.bar(x + (j - 1) * width, vals, width, label=label, color=color,
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--", zorder=2)
    ax.axvline(len(order) - 0.5, color=GRID, lw=1)
    ax.set_xticks(x, order + ["macro-12", f"all {len(means):,}"],
                  rotation=35, ha="right", color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy (mean over tie-break seeds)", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"smoothness_accuracy.{ext}"), dpi=220)
    plt.close(fig)

    # accuracy vs sigma, drawn exactly as panel (b) of ablation_v2's figure
    fig, bx = plt.subplots(figsize=(5.2, 4.3))
    for name in D.SYNTHETIC:
        bx.plot(curve.s, curve[name], color="#9a9992", lw=0.7, zorder=2)
    bx.plot(curve.s, curve.pooled, color=GREEN, lw=2.2, marker="o", ms=5, zorder=4,
            label="all pairs (full GDECI at $s$)")
    bx.plot(curve.s, curve.macro12, color=GREEN, lw=1.4, ls=":", zorder=3, label="macro-12")
    bx.plot(curve.s, curve.tuebingen, color=INK2, lw=1.4, ls="--", zorder=3, label="Tuebingen")
    bx.plot([], [], color="#9a9992", lw=0.7, label="individual synthetic suites")
    bx.set_xlabel(r"bandwidth fraction $s = h_s / h_0$", color=INK)
    bx.set_ylabel("accuracy", color=INK)
    bx.set_ylim(0.3, 1.02)
    bx.grid(color=GRID, lw=0.6, zorder=0)
    bx.legend(loc="lower right", frameon=False, fontsize=8)
    bx.set_title("accuracy vs bandwidth", loc="left", color=INK, fontsize=10)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    bx.tick_params(colors=INK2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"smoothness_bandwidth_path.{ext}"), dpi=220)
    plt.close(fig)

    # corrections and spoils (seed 0)
    main = summary["specifications"][MAIN_SPEC]
    corrected, spoiled = main["corrections"], main["spoils"]
    changed = corrected + spoiled
    cs = corrected / changed if changed else 0.0
    ss = spoiled / changed if changed else 0.0
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh([0], [cs], height=0.48, color="#168a55")
    ax.barh([0], [ss], left=[cs], height=0.48, color="#c43c39")
    ax.text(cs / 2 if changed else 0.25, 0, f"Corrected\n{corrected} ({100 * cs:.1f}%)",
            ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(cs + ss / 2 if changed else 0.75, 0, f"Spoiled\n{spoiled} ({100 * ss:.1f}%)",
            ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel(f"Share of {changed} changed decisions (tie-break seed 0)")
    ax.set_title("Corrections and spoils after adding the positive-bandwidth component")
    ax.text(0.5, -0.34,
            f"{changed}/{summary['design']['pairs']} decisions changed; net "
            f"{corrected - spoiled:+d}; McNemar p={main['mcnemar_exact_two_sided_p']:.2g}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"smoothness_decision_changes.{ext}"), dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_report(summary, curve, path):
    main = summary["specifications"][MAIN_SPEC]
    macro12 = summary["main_macro12"]
    sign = macro12["suite_level_sign_test"]
    sign13 = summary["main_all_pairs"]["suite_level_sign_test"]
    boot = main["stratified_bootstrap_macro12"]
    pooled_boot = main["stratified_bootstrap_pooled_all"]
    syn = summary["main_synthetic_1800"]
    tue_ds = summary["per_dataset"]["Tuebingen"]
    tue = summary["tuebingen_weighted"]
    diag = summary["diagnostics"]
    cc = summary["crosschecks"]
    per_dataset = "\n".join(
        f"| {name} | {v['bayes_only']['accuracy_all']:.3f} | "
        f"{v['smoothness_only']['accuracy_all']:.3f} | {v['full']['accuracy_all']:.3f} | "
        f"{v['corrections']} | {v['spoils']} |"
        for name, v in summary["per_dataset"].items())
    sensitivity = "\n".join(
        f"| {spec} | {v['bayes_only']['accuracy_all']:.3f} | "
        f"{v['smoothness_only']['accuracy_all']:.3f} | {v['full']['accuracy_all']:.3f} | "
        f"{v['corrections']} | {v['spoils']} | {v['mcnemar_exact_two_sided_p']:.3g} | "
        f"{v['stratified_bootstrap_pooled_all']['estimate']:+.3f} "
        f"[{v['stratified_bootstrap_pooled_all']['ci_95'][0]:+.3f}, "
        f"{v['stratified_bootstrap_pooled_all']['ci_95'][1]:+.3f}] |"
        for spec, v in summary["specifications"].items())
    windows = "\n".join(
        f"| {w['points']} (s <= {w['largest_fitted_fraction']:.1f}) | {w['bayes_only']:.3f} | "
        f"{w['smoothness_only']:.3f} | {w['corrections']} | {w['spoils']} | "
        f"{w['mcnemar_exact_two_sided_p']:.3g} | {w['median_r2']:.3f} |"
        for w in summary["fit_window_sensitivity"])
    head = " | ".join(f"{s:.1f}" for s in curve.s)
    curve_rows = "\n".join(
        f"| {label} | " + " | ".join(f"{v:.3f}" for v in curve[col]) + " |"
        for label, col in (("pooled 1899", "pooled"), ("macro-12", "macro12"),
                           ("Tuebingen", "tuebingen")))
    checks = []
    if "table_1b" in cc:
        checks.append(f"full score vs Table 1(b) (`per_pair.csv:unif_correct`): "
                      f"{cc['table_1b']['mismatches']} mismatches over "
                      f"{cc['table_1b']['matched_pairs']} pairs")
    if "ablation_v2_curve" in cc:
        checks.append(f"accuracy-vs-sigma decisions vs `ablation_v2`: "
                      f"{cc['ablation_v2_curve']['mismatches']} mismatches over "
                      f"{cc['ablation_v2_curve']['matched_pair_seeds']} pair-seeds x "
                      f"{len(cc['ablation_v2_curve']['fractions'])} bandwidths")
    h = diag["lag_bandwidth_at_smallest_fraction"]
    text = f"""# Does the GDECI positive-bandwidth part carry causal signal?

## Design

All {summary['design']['pairs']:,} pairs from {summary['design']['datasets']} benchmarks,
unnormalized Laplacian `L = D - W`, frozen rank-space multiplier `m = {FROZEN_M}`.
`Delta_sigma = E_sigma/M_sigma` (self-loops included in `M`) is evaluated at
`s = sigma/sigma_0` in `{', '.join(f'{x:.1f}' for x in FRACTIONS)}`, and
`Delta_sigma = B + a*sigma` is fitted on `s` in `{', '.join(f'{x:.1f}' for x in FIT_FRACTIONS)}`.
The intercept estimates `E Var(V|U)`; the frozen score minus the intercept estimates the
positive-bandwidth component. Ties are broken at random with seeds 0-9 as in Table 1(b), and
accuracies are means over the seeds; paired counts and McNemar tests use seed 0. Decisions
are strict (`forward < reverse`); causal labels are used only for scoring. At `s = 0.1` the
lag bandwidth is h = {', '.join(f'{v:.2f} (n = {k})' for k, v in h.items())}.

Cross-checks: {'; '.join(checks) if checks else 'none available'}.

## Primary result (linear in sigma, s = 0.1..0.5)

| | conditional-variance component | positive-bandwidth component alone | full GDECI |
|---|---:|---:|---:|
| all 1,899 pairs | {main['bayes_only']['accuracy_all']:.3f} | {main['smoothness_only']['accuracy_all']:.3f} | {main['full']['accuracy_all']:.3f} |
| macro, 12 synthetic suites | {macro12['bayes_only']:.3f} | {macro12['smoothness_only']:.3f} | {macro12['full']:.3f} |
| Tuebingen, weighted | {tue['bayes_only']:.3f} | {tue['smoothness_only']:.3f} | {tue['full']:.3f} |

Over all {summary['design']['pairs']:,} pairs, adding the positive-bandwidth component
corrected **{main['corrections']}** decisions and spoiled **{main['spoils']}**
(net {main['net_corrections']:+d}; exact McNemar
`p = {main['mcnemar_exact_two_sided_p']:.3g}`, seed 0; means over seeds
{main['corrections_mean_over_seeds']:.1f} and {main['spoils_mean_over_seeds']:.1f}), split as
{syn['corrections']} against {syn['spoils']} on the {syn['full']['pairs']:,} synthetic pairs
and {tue_ds['corrections']} against {tue_ds['spoils']} on Tuebingen. Pooled accuracy changed by
{pooled_boot['estimate']:+.3f} (stratified bootstrap 95% CI
[{pooled_boot['ci_95'][0]:+.3f}, {pooled_boot['ci_95'][1]:+.3f}]) and the 12-suite macro
accuracy by {boot['estimate']:+.3f} ([{boot['ci_95'][0]:+.3f}, {boot['ci_95'][1]:+.3f}]).
The gain was positive on {sign13['positive']} of the {summary['design']['datasets']} suites,
negative on {sign13['negative']} and zero on {sign13['ties']} (one-sided sign test
`p = {sign13['one_sided_p']:.3g}`; over the 12 synthetic suites alone,
{sign['positive']}/{sign['negative']}/{sign['ties']}, `p = {sign['one_sided_p']:.3g}`).
McNemar treats pairs as independent; the bootstrap resamples pairs within suites.

## Accuracy of the full score vs bandwidth (`ablation_v2` cross-checks s >= 0.2)

| s | {head} |
|---|{'---:|' * len(curve)}
{curve_rows}

## Per dataset (linear in sigma)

| benchmark | conditional variance | positive-bandwidth alone | full | corrections | spoils |
|---|---:|---:|---:|---:|---:|
{per_dataset}

## Sensitivity to the fit window (linear in sigma, k smallest bandwidths)

| k (s <= ) | conditional variance | positive-bandwidth alone | corrections | spoils | McNemar p | median R^2 |
|---|---:|---:|---:|---:|---:|---:|
{windows}

## Sensitivity to the extrapolation

| specification | conditional variance | positive-bandwidth alone | full | corrections | spoils | McNemar p | pooled change [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---|
{sensitivity}

Diagnostics (seed 0, linear fit): {diag['negative_smoothness_direction_estimates']} negative
positive-bandwidth components and {diag['negative_bayes_direction_estimates']} negative /
{diag['bayes_above_uniform_variance_estimates']} above-1/12 intercepts out of
{diag['direction_estimates']} directional estimates (range
[{diag['minimum_bayes_estimate']:.4g}, {diag['maximum_bayes_estimate']:.4g}]); median fit
R^2 {diag['median_small_path_r2']:.4f}. Nothing is clipped.

Caveats. With the self-loops included, `M_sigma` exceeds the off-diagonal edge mass by
roughly the factor `1 + 1/(sqrt(2 pi) h)`, which is not linear in sigma; at `s = 0.1` the lag
bandwidth is only about two grid steps. Both affect the intercept, and hence the split into
the two components, but not the full score. `ablation_v2/` reports the same decomposition
with `E/(M - n)` and an `h >= 5` window for comparison.

## Artifacts

- `figures/smoothness_accuracy.pdf`: component accuracy by benchmark, macro-12 and
  pooled over all pairs.
- `figures/smoothness_bandwidth_path.pdf`: accuracy of the full score vs bandwidth.
- `figures/smoothness_decision_changes.pdf`: corrected and spoiled decisions (seed 0).
- `results/smoothness_signal_per_pair.csv`: one row per pair (seed-0 scores and components,
  plus `*_mean` correctness averaged over tie seeds).
- `results/smoothness_signal_paths.csv`: bandwidth-path scores per tie seed (input of
  `--reuse-results`).
- `results/accuracy_vs_sigma.csv`, `results/smoothness_signal_summary.json`.
"""
    with open(path, "w") as handle:
        handle.write(text)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="development-only limit per dataset")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "smoothness_experiment"))
    parser.add_argument("--reuse-results", action="store_true",
                        help="recompute components, summary and figures from stored path scores")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    results_dir = os.path.join(output_dir, "results")
    figures_dir = os.path.join(output_dir, "figures")
    csv_path = os.path.join(results_dir, "smoothness_signal_per_pair.csv")
    paths_path = os.path.join(results_dir, "smoothness_signal_paths.csv")
    os.makedirs(results_dir, exist_ok=True)
    if args.reuse_results:
        if not os.path.exists(paths_path):
            parser.error(f"--reuse-results requires {paths_path}")
        print("reusing", paths_path, flush=True)
        paths = pd.read_csv(paths_path)
    else:
        tasks = []
        for name, _, _, _ in D.BENCHMARKS:
            pairs = list(D.load(name))
            if args.limit:
                pairs = pairs[:args.limit]
            tasks.extend((name, pair_id, cause, effect, weight)
                         for pair_id, cause, effect, weight in pairs)
        print(f"scoring {len(tasks)} pairs at {len(FRACTIONS)} bandwidths, "
              f"tie seeds {TIE_SEEDS[0]}-{TIE_SEEDS[-1]}", flush=True)
        if args.jobs > 1:
            from multiprocessing import Pool
            with Pool(args.jobs) as pool:
                chunks = list(pool.imap(score_pair, tasks, chunksize=4))
        else:
            chunks = []
            for index, task in enumerate(tasks, 1):
                chunks.append(score_pair(task))
                if index % 100 == 0 or index == len(tasks):
                    print(f"[{index}/{len(tasks)}]", flush=True)
        paths = pd.DataFrame([row for chunk in chunks for row in chunk])
        paths.to_csv(paths_path, index=False)
    if paths.empty:
        parser.error("No benchmark pairs were loaded")

    seeds = refresh_extrapolations(expand_seeds(paths))
    order = {entry[0]: i for i, entry in enumerate(D.BENCHMARKS)}
    means = pair_means(seeds)
    means = means.sort_values(["benchmark", "pair_id"],
                              key=lambda c: c.map(order) if c.name == "benchmark" else c)
    means = means.reset_index(drop=True)
    seed0 = (seeds[seeds.seed == 0].set_index(["benchmark", "pair_id"])
             .loc[list(zip(means.benchmark, means.pair_id))].reset_index())
    crosschecks = run_crosschecks(means, seeds)

    per_pair = seed0.drop(columns=["seed", "seed_invariant"]).merge(
        means[["benchmark", "pair_id"] + correct_columns()].rename(
            columns={c: f"{c}_mean" for c in correct_columns()}),
        on=["benchmark", "pair_id"])
    per_pair.to_csv(csv_path, index=False)
    summary, curve = build_summary(seeds, means, seed0, crosschecks)
    curve.to_csv(os.path.join(results_dir, "accuracy_vs_sigma.csv"), index=False)
    with open(os.path.join(results_dir, "smoothness_signal_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    make_figures(means, summary, curve, figures_dir)
    write_report(summary, curve, os.path.join(output_dir, "SMOOTHNESS_SIGNAL_REPORT.md"))

    m = summary["specifications"][MAIN_SPEC]
    print(json.dumps({
        "bayes_accuracy": m["bayes_only"]["accuracy_all"],
        "smoothness_accuracy": m["smoothness_only"]["accuracy_all"],
        "full_accuracy": m["full"]["accuracy_all"],
        "macro12": summary["main_macro12"],
        "corrections_seed0": m["corrections"],
        "spoils_seed0": m["spoils"],
        "mcnemar_p_seed0": m["mcnemar_exact_two_sided_p"],
        "macro12_change": m["stratified_bootstrap_macro12"],
        "accuracy_vs_sigma_macro12": dict(zip(curve.s.round(1).astype(str),
                                              curve.macro12.round(4))),
        "crosschecks": crosschecks,
    }, indent=2))
    print("wrote", output_dir)


if __name__ == "__main__":
    main()
