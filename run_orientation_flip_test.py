"""Orientation-flip test: is GDECI's decision invariant to negating a variable?

GDECI scores a direction by the UNNORMALIZED Laplacian energy
E = v^T (D - W) v of the effect ranks on the graph of the cause ranks, with
self-loops in W and the frozen multiplier m = 0.07362, and returns the direction
with the smaller energy per unit edge mass.  The rank transform is invariant to
any increasing map of either variable, but negation is decreasing: it reverses
the rank order, v -> 1 - v.

Theory says the unnormalized energy absorbs that.  E = 1/2 sum_ij W_ij (v_i-v_j)^2
depends on the signal only through differences, so it is invariant to adding a
constant and to sign, hence E(1 - v) = E(v); and reversing the ordering of the
*conditioning* variable reverses the signal on a persymmetric Toeplitz kernel,
which leaves the energy unchanged as well.  So no negation of either variable
can move a GDECI score.

The symmetric-normalized Laplacian I - D^{-1/2} W D^{-1/2} does not have this
property: E_sym = v^T v - z^T W z with z = D^{-1/2} v is not a function of the
differences alone, L_sym 1 != 0 whenever the degrees are unequal, and the
degree weighting does not absorb v -> 1 - v.  E_sym(U->V) therefore moves when Y
is negated, and E_sym(V->U) when X is.  This script measures both claims on the
benchmark: that GDECI does not move, and how often the normalized variant --
which an earlier version of this work used -- would have changed the decision.

One caveat is measured rather than assumed.  The invariance of E holds for the
*same* rank vectors, and `rank_grid` breaks ties by original index: negating
a variable reverses the order of its distinct values but keeps the index order
inside a tied block, so a tied sample can receive a different rank and the
signal itself changes.  Invariance is therefore asserted only on pairs whose raw
arrays have no ties (there it holds to rounding, ~1e-13 relative); pairs with
ties are counted and their deviations reported as a finding.

`lap_nlogn.FastGridKernel.energy` returns the published energy (verified there
against a dense reference) and `energy_sym` the normalized contrast.  Nothing is
reimplemented.

Four configurations are built by negating the raw arrays *before* any
preprocessing -- orig (X, Y), flipX (-X, Y), flipY (X, -Y), flipBoth (-X, -Y) --
and each is pushed through the unchanged pipeline, so ranking, tie-breaking and
bandwidth selection behave exactly as in the method.  Three scores are compared:

  (a) gdeci  the published score, E / sum_ij W_ij, one decision per
             configuration;
  (b) sym    the symmetric-normalized energy on the same kernel and bandwidth,
             the variant this work replaced;
  (c) symsum the symmetrization that (b) would have needed: its margin is the
             sum of the four E_sym margins, i.e. one decision per pair rather
             than per configuration.

Energies are divided by the edge mass sum_ij W_ij, which is the score of
Proposition 1.  All four configurations of a pair share one kernel, hence one
edge mass, so this scaling changes no margin sign, no decision and no accuracy.

Causal labels are used only after every score has been computed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(ROOT, "loci_dataset_benchmark")
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, ROOT)

import datasets as D
from lap_nlogn import kernel_for, rank_grid
# decision(), wilson() and accuracy_stats() are reused unchanged so that tie
# handling, interval arithmetic and the "ties count as incorrect" convention are
# identical to the smoothness experiment.
from run_smoothness_experiment import accuracy_stats, decision, wilson

FROZEN_M = 0.07362
# Sign applied to the raw cause and effect arrays before any preprocessing.
CONFIGS = {"orig": (1.0, 1.0), "flipX": (-1.0, 1.0),
           "flipY": (1.0, -1.0), "flipBoth": (-1.0, -1.0)}
FLIPS = ("flipX", "flipY", "flipBoth")
SCORES = ("gdeci", "sym")
# Published GDECI reference values this test must reproduce in `orig`.
PUBLISHED_MACRO12 = 0.776
PUBLISHED_TUEBINGEN = 0.677
# Tolerance for the orientation invariance of the published energy.  The four
# configurations evaluate mathematically identical quadratic forms through
# different FFT operand orders, so agreement is to rounding, not to the bit.
GDECI_RTOL = 1e-6


@dataclass
class Energies:
    """The two energies of one direction, already divided by the edge mass."""

    gdeci: float
    sym: float


def margin_columns():
    """(label, margin column, scale column) for every score that is reported."""
    columns = []
    for score in SCORES:
        for config in CONFIGS:
            columns.append((f"{score}_{config}", f"{score}_{config}_margin",
                            f"{score}_{config}_scale"))
    columns.append(("symsum", "symsum_margin", "symsum_scale"))
    return columns


def score_pair(task):
    """Score one benchmark pair in all four orientations.

    The task tuple matches run_smoothness_experiment.score_pair:
    (benchmark, pair_id, cause, effect, weight), with the true cause first.
    """
    name, pair_id, cause, effect, weight = task
    cause = np.asarray(cause, float)
    effect = np.asarray(effect, float)
    n = len(cause)
    # Ties are a property of the raw arrays and are unchanged by negation; they
    # decide whether the published energy can be expected to be orientation-invariant.
    row = dict(benchmark=name, pair_id=pair_id, n=n, weight=weight,
               frozen_multiplier=FROZEN_M,
               ties_cause=int(n - len(np.unique(cause))),
               ties_effect=int(n - len(np.unique(effect))))

    symsum_margin = 0.0
    symsum_scale = 0.0
    for config, (sign_x, sign_y) in CONFIGS.items():
        # Negate the raw arrays, then run the unchanged pipeline.
        u = rank_grid(sign_x * cause)
        v = rank_grid(sign_y * effect)
        signals = np.vstack([v[np.argsort(u)], u[np.argsort(v)]])
        kernel = kernel_for(n, FROZEN_M)
        gdeci = kernel.energy(signals) / kernel.mass
        sym = kernel.energy_sym(signals) / kernel.mass
        forward = Energies(float(gdeci[0]), float(sym[0]))
        reverse = Energies(float(gdeci[1]), float(sym[1]))
        for score, fwd, rev in (("gdeci", forward.gdeci, reverse.gdeci),
                                ("sym", forward.sym, reverse.sym)):
            row[f"{score}_{config}_fwd"] = fwd
            row[f"{score}_{config}_rev"] = rev
            row[f"{score}_{config}_margin"] = rev - fwd
            row[f"{score}_{config}_scale"] = abs(fwd) + abs(rev)
        symsum_margin += row[f"sym_{config}_margin"]
        symsum_scale += row[f"sym_{config}_scale"]

    # Score (c): one decision per pair from the four orientations together.
    row["symsum_margin"] = symsum_margin
    row["symsum_scale"] = symsum_scale
    return row


def add_decisions(df):
    """Attach a decision column per score, using the shared tie tolerance."""
    for label, margin_col, scale_col in margin_columns():
        df[f"dec_{label}"] = [decision(m, s) for m, s
                              in zip(df[margin_col], df[scale_col])]
    return df


def correct_mask(df, label):
    """Correct-decision indicator; ties count as incorrect, as elsewhere."""
    return df[f"dec_{label}"].to_numpy() == 1


def stats_for(df, label):
    """Accuracy with a Wilson interval for one score on one subset of pairs."""
    stats, _ = accuracy_stats(df[f"{label}_margin"].to_numpy(),
                              df[f"{label}_scale"].to_numpy())
    return stats


def mcnemar(first_correct, second_correct):
    """Exact two-sided McNemar test on paired correct/incorrect outcomes."""
    first_correct = np.asarray(first_correct, bool)
    second_correct = np.asarray(second_correct, bool)
    gains = int((~first_correct & second_correct).sum())
    losses = int((first_correct & ~second_correct).sum())
    discordant = gains + losses
    p = float(binomtest(gains, discordant, 0.5).pvalue) if discordant else 1.0
    return {"first_only_correct": losses, "second_only_correct": gains,
            "discordant": discordant, "exact_two_sided_p": p}


def flip_rates(df, score="gdeci"):
    """Share of pairs whose decision under `score` changes under each negation."""
    base = df[f"dec_{score}_orig"].to_numpy()
    rates = {}
    any_changed = np.zeros(len(df), bool)
    for flip in FLIPS:
        changed = df[f"dec_{score}_{flip}"].to_numpy() != base
        any_changed |= changed
        rates[flip] = {"rate": float(changed.mean()),
                       "changed": int(changed.sum()),
                       "pairs": int(len(df))}
    rates["any"] = {"rate": float(any_changed.mean()),
                    "changed": int(any_changed.sum()), "pairs": int(len(df))}
    return rates


def group_summary(df):
    """Accuracies, flip rates and paired tests for one group of pairs."""
    out = {"pairs": int(len(df)),
           "accuracy": {label: stats_for(df, label)
                        for label, _, _ in margin_columns()},
           "flip_rates": {score: flip_rates(df, score) for score in SCORES},
           "mcnemar_orig_vs_flip": {
               flip: mcnemar(correct_mask(df, "sym_orig"),
                             correct_mask(df, f"sym_{flip}"))
               for flip in FLIPS}}
    base = correct_mask(df, "gdeci_orig")
    for label in ("sym_orig", "symsum"):
        out[f"paired_gdeci_orig_vs_{label}"] = {
            "accuracy_difference": (out["accuracy"]["gdeci_orig"]["accuracy_all"] -
                                    out["accuracy"][label]["accuracy_all"]),
            **mcnemar(base, correct_mask(df, label)),
        }
    return out


def macro_summary(df, names):
    """Macro mean over suites, with the Wilson interval at the pair count.

    Suite accuracies are averaged with equal weight, as in the paper's table;
    the interval is the Wilson interval of that mean at the pooled pair count,
    matching loci_dataset_benchmark/summarize_direct.py.  Paired tests pool the
    pairs, since a paired test needs pairs and not suite means.
    """
    subsets = [df[df.benchmark == name] for name in names]
    pooled = df[df.benchmark.isin(names)]
    total = int(len(pooled))
    out = {"suites": list(names), "pairs": total, "accuracy": {}}
    for label, _, _ in margin_columns():
        macro = float(np.mean([correct_mask(d, label).mean() for d in subsets]))
        out["accuracy"][label] = {
            "accuracy_all": macro,
            "wilson_95": wilson(macro * total, total),
            "per_suite": {name: float(correct_mask(d, label).mean())
                          for name, d in zip(names, subsets)},
        }
    out["flip_rates"] = {
        score: {flip: {"rate": float(np.mean([flip_rates(d, score)[flip]["rate"]
                                              for d in subsets])),
                       "pooled_rate": flip_rates(pooled, score)[flip]["rate"],
                       "changed": flip_rates(pooled, score)[flip]["changed"],
                       "pairs": total}
                for flip in (*FLIPS, "any")}
        for score in SCORES}
    out["mcnemar_orig_vs_flip"] = {
        flip: mcnemar(correct_mask(pooled, "sym_orig"),
                      correct_mask(pooled, f"sym_{flip}")) for flip in FLIPS}
    base = correct_mask(pooled, "gdeci_orig")
    for label in ("sym_orig", "symsum"):
        macro_base = out["accuracy"]["gdeci_orig"]["accuracy_all"]
        out[f"paired_gdeci_orig_vs_{label}"] = {
            "accuracy_difference": macro_base - out["accuracy"][label]["accuracy_all"],
            **mcnemar(base, correct_mask(pooled, label)),
        }
    return out


def tuebingen_summary(df):
    """Tuebingen, unweighted and with the pairmeta weights."""
    tue = df[df.benchmark == "Tuebingen"]
    weights = tue.weight.to_numpy(float)
    effective = float(weights.sum() ** 2 / np.sum(weights ** 2))
    weighted = {"effective_sample_size": effective, "pairs": int(len(tue)),
                "accuracy": {}, "flip_rates": {}}
    for label, _, _ in margin_columns():
        correct = correct_mask(tue, label)
        accuracy = float(np.sum(weights * correct) / weights.sum())
        weighted["accuracy"][label] = {
            "accuracy_all": accuracy,
            "wilson_95": wilson(accuracy * effective, effective)}
    for score in SCORES:
        base = tue[f"dec_{score}_orig"].to_numpy()
        weighted["flip_rates"][score] = {}
        for flip in FLIPS:
            changed = tue[f"dec_{score}_{flip}"].to_numpy() != base
            weighted["flip_rates"][score][flip] = {
                "rate": float(np.sum(weights * changed) / weights.sum()),
                "changed": int(changed.sum()), "pairs": int(len(tue))}
    return {"unweighted": group_summary(tue), "weighted": weighted}


def run_sanity_checks(df):
    """Checks that must fail loudly; returns what they measured."""
    problems = []
    checks = {}

    # 1. the published GDECI results must come back out of `orig`.
    macro12 = float(np.mean([correct_mask(df[df.benchmark == name], "gdeci_orig").mean()
                             for name in D.SYNTHETIC]))
    tuebingen = float(correct_mask(df[df.benchmark == "Tuebingen"], "gdeci_orig").mean())
    checks["published_macro12"] = {"value": macro12, "expected": PUBLISHED_MACRO12,
                                   "passed": round(macro12, 3) == PUBLISHED_MACRO12}
    checks["published_tuebingen"] = {"value": tuebingen, "expected": PUBLISHED_TUEBINGEN,
                                     "passed": round(tuebingen, 3) == PUBLISHED_TUEBINGEN}
    if not checks["published_macro12"]["passed"]:
        problems.append(f"macro accuracy of the published score is {macro12:.4f}, "
                        f"expected {PUBLISHED_MACRO12:.3f}")
    if not checks["published_tuebingen"]["passed"]:
        problems.append(f"Tuebingen accuracy of the published score is {tuebingen:.4f}, "
                        f"expected {PUBLISHED_TUEBINGEN:.3f}")

    # 2. the published energy must not move with orientation, in margin or in
    #    decision -- on pairs without ties.  With ties, rank_grid assigns the
    #    ranks inside a tied block by original index, so negation changes the
    #    rank vectors themselves and every score may legitimately move; those
    #    pairs are reported rather than asserted on.
    tie_free = ((df["ties_cause"].to_numpy() == 0) &
                (df["ties_effect"].to_numpy() == 0))
    base_margin = df["gdeci_orig_margin"].to_numpy(float)
    scale = np.maximum(df["gdeci_orig_scale"].to_numpy(float), 1e-300)
    base_decision = df["dec_gdeci_orig"].to_numpy()
    worst = {True: 0.0, False: 0.0}
    changes = {True: 0, False: 0}
    for flip in FLIPS:
        deviation = np.abs(df[f"gdeci_{flip}_margin"].to_numpy(float) -
                           base_margin) / scale
        differs = df[f"dec_gdeci_{flip}"].to_numpy() != base_decision
        for group in (True, False):
            mask = tie_free if group else ~tie_free
            if mask.any():
                worst[group] = max(worst[group], float(deviation[mask].max()))
                changes[group] += int(differs[mask].sum())
    checks["gdeci_invariance"] = {
        "tie_free_pairs": int(tie_free.sum()),
        "max_relative_margin_deviation": worst[True],
        "tolerance": GDECI_RTOL,
        "decision_changes": changes[True],
        "passed": worst[True] <= GDECI_RTOL and changes[True] == 0,
        "pairs_with_ties": int((~tie_free).sum()),
        "tied_max_relative_margin_deviation": worst[False],
        "tied_decision_changes": changes[False],
        "note": "ties are broken by original index, so negation can change the "
                "rank vectors of a tied pair; invariance is asserted on tie-free "
                "pairs only",
    }
    if not checks["gdeci_invariance"]["passed"]:
        problems.append("the published GDECI energy is not orientation-invariant on "
                        f"tie-free pairs: worst relative margin deviation "
                        f"{worst[True]:.2e}, {changes[True]} decision changes")

    # 2b. the contrast must be real: the normalized energy is expected to move.
    #     A run in which it did not would mean the two scores had been wired to
    #     the same quantity, which this test would otherwise not notice.
    sym_moved = int(flip_rates(df, "sym")["any"]["changed"])
    checks["normalized_variant_moves"] = {
        "decision_changes": sym_moved,
        "pairs": int(len(df)),
        "passed": sym_moved > 0,
        "note": "E_sym is not orientation-invariant; if it did not move, the "
                "two scores would not be measuring different quantities",
    }
    if not checks["normalized_variant_moves"]["passed"]:
        problems.append("the symmetric-normalized energy did not change a single "
                        "decision under negation; expected it to")

    # 3. shape of the table.
    duplicates = int(df.duplicated(["benchmark", "pair_id"]).sum())
    checks["table_shape"] = {"rows": int(len(df)),
                             "suites": int(df.benchmark.nunique()),
                             "duplicate_rows": duplicates,
                             "passed": (len(df) == 1899 and
                                        df.benchmark.nunique() == 13 and
                                        duplicates == 0)}
    if not checks["table_shape"]["passed"]:
        problems.append(f"expected 1899 rows over 13 suites with no duplicates, found "
                        f"{len(df)} rows, {df.benchmark.nunique()} suites, "
                        f"{duplicates} duplicates")

    checks["all_passed"] = not problems
    if problems:
        raise SystemExit("sanity checks failed:\n  - " + "\n  - ".join(problems))
    return checks


def build_summary(df):
    order = [entry[0] for entry in D.BENCHMARKS]
    summary = {
        "design": {
            "pairs": int(len(df)),
            "datasets": int(df.benchmark.nunique()),
            "frozen_multiplier": FROZEN_M,
            "configurations": {name: list(signs) for name, signs in CONFIGS.items()},
            "scores": {
                "gdeci": "E = v^T (D - W) v, the published score, unnormalized "
                         "Laplacian",
                "sym": "E_sym = v^T (I - D^-1/2 W D^-1/2) v on the same kernel "
                       "and bandwidth, the normalized variant this work replaced",
                "symsum": "margin is the sum of the four E_sym margins of a pair",
            },
            "energy_scaling": "energies divided by the edge mass sum_ij W_ij; "
                              "identical kernel per pair, so decisions are "
                              "unaffected",
            "tie_policy": "ties count as incorrect; shared decision() tolerance",
            "negation_applied_to": "raw arrays, before any preprocessing",
            "deterministic": True,
            "causal_labels_used_for_scoring": False,
        },
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "sanity_checks": run_sanity_checks(df),
        "per_dataset": {name: group_summary(df[df.benchmark == name])
                        for name in order},
        "macro12": macro_summary(df, D.SYNTHETIC),
        "tuebingen": tuebingen_summary(df),
        "pooled_1899": group_summary(df),
    }
    return summary


def make_figures(df, summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    order = [entry[0] for entry in D.BENCHMARKS]
    x = np.arange(len(order))

    # Flip rate per suite and per negation, for the published score and for the
    # normalized variant it replaced.
    colors = {"flipX": "#4c78a8", "flipY": "#f2a541", "flipBoth": "#2a9d6f"}
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 8.2), sharex=True)
    titles = {"gdeci": "GDECI, unnormalized Laplacian (the published score)",
              "sym": "symmetric-normalized Laplacian (the replaced variant)"}
    width = 0.27
    for ax, score in zip(axes, SCORES):
        for offset, flip in zip((-width, 0.0, width), FLIPS):
            rates = [summary["per_dataset"][name]["flip_rates"][score][flip]["rate"]
                     for name in order]
            ax.bar(x + offset, rates, width, label=flip, color=colors[flip])
        macro_any = summary["macro12"]["flip_rates"][score]["any"]["rate"]
        ax.axhline(macro_any, color="black", lw=1, ls="--", alpha=0.65,
                   label=f"macro rate, any negation ({macro_any:.3f})")
        ax.set_ylabel("Share of pairs whose\ndecision changes")
        ax.set_title(titles[score])
        ax.legend(ncol=4, loc="upper left")
    axes[-1].set_xticks(x, order, rotation=40, ha="right")
    fig.suptitle("Decision changes under negation of a variable")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"orientation_flip_rates.{ext}"), dpi=220)
    plt.close(fig)

    # Accuracy of the normalized variant in each orientation, beside (a) and (c).
    series = [("gdeci_orig", "(a) GDECI, any orientation")]
    series += [(f"sym_{config}", f"(b) E_sym, {config}") for config in CONFIGS]
    series.append(("symsum", "(c) E_sym symmetrized"))
    palette = ["#2a9d6f", "#4c78a8", "#8fbbdb", "#f2a541", "#f6cf8a", "#8e6bbf"]
    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    width = 0.8 / len(series)
    for index, ((label, text), color) in enumerate(zip(series, palette)):
        values = [summary["per_dataset"][name]["accuracy"][label]["accuracy_all"]
                  for name in order]
        ax.bar(x + (index - (len(series) - 1) / 2) * width, values, width,
               label=text, color=color)
    ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.65)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Causal-direction accuracy")
    ax.set_xticks(x, order, rotation=40, ha="right")
    ax.legend(ncol=3, loc="lower left", fontsize=9)
    ax.set_title("Accuracy by orientation: the invariant score against the "
                 "normalized variant and its symmetrization")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"orientation_accuracy.{ext}"), dpi=220)
    plt.close(fig)


def write_report(summary, path):
    macro = summary["macro12"]
    tue = summary["tuebingen"]
    checks = summary["sanity_checks"]
    lines = [
        "# Orientation-flip test for GDECI",
        "",
        f"{summary['design']['pairs']} benchmark pairs, "
        f"{summary['design']['datasets']} suites, frozen multiplier "
        f"m = {summary['design']['frozen_multiplier']}. Each pair is scored in four "
        "orientations, negating the raw arrays before any preprocessing.",
        "",
        "## Sanity checks",
        "",
        f"- published GDECI reproduced in `orig`: macro12 "
        f"{checks['published_macro12']['value']:.3f} "
        f"(expected {checks['published_macro12']['expected']:.3f}), Tuebingen "
        f"{checks['published_tuebingen']['value']:.3f} "
        f"(expected {checks['published_tuebingen']['expected']:.3f}) -- "
        f"{'PASS' if checks['published_macro12']['passed'] and checks['published_tuebingen']['passed'] else 'FAIL'}",
        f"- GDECI invariant over the four orientations on the "
        f"{checks['gdeci_invariance']['tie_free_pairs']} tie-free pairs: worst "
        f"relative margin deviation "
        f"{checks['gdeci_invariance']['max_relative_margin_deviation']:.2e} "
        f"(tolerance {checks['gdeci_invariance']['tolerance']:.0e}), "
        f"{checks['gdeci_invariance']['decision_changes']} decision changes -- "
        f"{'PASS' if checks['gdeci_invariance']['passed'] else 'FAIL'}",
        f"- the remaining {checks['gdeci_invariance']['pairs_with_ties']} pairs "
        f"have ties in the raw arrays, where negation changes the rank vectors "
        f"themselves: worst relative margin deviation "
        f"{checks['gdeci_invariance']['tied_max_relative_margin_deviation']:.2e}, "
        f"{checks['gdeci_invariance']['tied_decision_changes']} decision changes "
        f"(reported, not asserted)",
        f"- the normalized variant does move, as expected: "
        f"{checks['normalized_variant_moves']['decision_changes']} of "
        f"{checks['normalized_variant_moves']['pairs']} pairs change decision under "
        f"some negation -- "
        f"{'PASS' if checks['normalized_variant_moves']['passed'] else 'FAIL'}",
        f"- table shape: {checks['table_shape']['rows']} rows, "
        f"{checks['table_shape']['suites']} suites, "
        f"{checks['table_shape']['duplicate_rows']} duplicates -- "
        f"{'PASS' if checks['table_shape']['passed'] else 'FAIL'}",
        "",
        "## Flip rates",
        "",
        "`gdeci` is the published score on the unnormalized Laplacian; `sym` is "
        "the symmetric-normalized variant it replaced.",
        "",
        "| score | group | flipX | flipY | flipBoth | any |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for score in SCORES:
        rates = macro["flip_rates"][score]
        lines.append(f"| {score} | macro mean, 12 non-Tuebingen | " + " | ".join(
            f"{rates[flip]['rate']:.3f}" for flip in (*FLIPS, "any")) + " |")
        pooled = summary["pooled_1899"]["flip_rates"][score]
        lines.append(f"| {score} | pooled, 1899 pairs | " + " | ".join(
            f"{pooled[flip]['rate']:.3f}" for flip in (*FLIPS, "any")) + " |")
        tue_rates = tue["unweighted"]["flip_rates"][score]
        lines.append(f"| {score} | Tuebingen | " + " | ".join(
            f"{tue_rates[flip]['rate']:.3f}" for flip in (*FLIPS, "any")) + " |")
    lines += ["", "Per suite:", "",
              "| suite | pairs | " + " | ".join(
                  f"{score} {flip}" for score in SCORES for flip in (*FLIPS, "any"))
              + " |",
              "|---|---:|" + "---:|" * (len(SCORES) * (len(FLIPS) + 1))]
    for name in [entry[0] for entry in D.BENCHMARKS]:
        entry = summary["per_dataset"][name]
        lines.append(f"| {name} | {entry['pairs']} | " + " | ".join(
            f"{entry['flip_rates'][score][flip]['rate']:.3f}"
            for score in SCORES for flip in (*FLIPS, "any")) + " |")

    lines += ["", "## Accuracy", "",
              "| score | macro12 | 95% Wilson | Tuebingen | Tuebingen, weighted |",
              "|---|---:|---|---:|---:|"]
    for label, _, _ in margin_columns():
        macro_entry = macro["accuracy"][label]
        interval = macro_entry["wilson_95"]
        lines.append(
            f"| {label} | {macro_entry['accuracy_all']:.3f} | "
            f"[{interval[0]:.3f}, {interval[1]:.3f}] | "
            f"{tue['unweighted']['accuracy'][label]['accuracy_all']:.3f} | "
            f"{tue['weighted']['accuracy'][label]['accuracy_all']:.3f} |")

    lines += ["", "## Paired tests (macro-12 pairs pooled)", "",
              "| comparison | accuracy difference | discordant | exact two-sided p |",
              "|---|---:|---:|---:|"]
    for flip in FLIPS:
        test = macro["mcnemar_orig_vs_flip"][flip]
        difference = (macro["accuracy"]["sym_orig"]["accuracy_all"] -
                      macro["accuracy"][f"sym_{flip}"]["accuracy_all"])
        lines.append(f"| E_sym orig vs {flip} | {difference:+.3f} | "
                     f"{test['discordant']} | {test['exact_two_sided_p']:.4g} |")
    for label in ("sym_orig", "symsum"):
        test = macro[f"paired_gdeci_orig_vs_{label}"]
        lines.append(f"| GDECI vs {label} | {test['accuracy_difference']:+.3f} | "
                     f"{test['discordant']} | {test['exact_two_sided_p']:.4g} |")
    lines += ["",
              "Ties count as incorrect throughout, with the tie tolerance of "
              "`run_smoothness_experiment.decision`. Macro intervals are Wilson "
              "intervals of the macro mean at the pooled pair count; the weighted "
              "Tuebingen row uses the pairmeta weights at effective sample size "
              f"{tue['weighted']['effective_sample_size']:.1f}.",
              ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="development-only limit per dataset")
    parser.add_argument("--output-dir",
                        default=os.path.join(ROOT, "orientation_flip_test"))
    parser.add_argument("--reuse-results", action="store_true",
                        help="regenerate summary, report and figures from the stored CSV")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    results_dir = os.path.join(output_dir, "results")
    figures_dir = os.path.join(output_dir, "figures")
    csv_path = os.path.join(results_dir, "orientation_flip_per_pair.csv")
    os.makedirs(results_dir, exist_ok=True)

    if args.reuse_results:
        if not os.path.exists(csv_path):
            parser.error(f"--reuse-results requires {csv_path}")
        print("reusing", csv_path, flush=True)
        # round_trip parsing keeps the stored margins bit-exact; the default
        # parser is accurate only to about one ULP, which would make a reuse
        # cycle rewrite the CSV with slightly different digits.
        df = pd.read_csv(csv_path, float_precision="round_trip")
    else:
        tasks = []
        for name, _, _, _ in D.BENCHMARKS:
            pairs = list(D.load(name))
            if args.limit:
                pairs = pairs[:args.limit]
            tasks.extend((name, pair_id, cause, effect, weight)
                         for pair_id, cause, effect, weight in pairs)
        print(f"scoring {len(tasks)} pairs in {len(CONFIGS)} orientations", flush=True)
        if args.jobs > 1:
            from multiprocessing import Pool
            with Pool(args.jobs) as pool:
                rows = list(pool.imap(score_pair, tasks, chunksize=4))
        else:
            rows = []
            for index, task in enumerate(tasks, 1):
                rows.append(score_pair(task))
                if index % 100 == 0 or index == len(tasks):
                    print(f"[{index}/{len(tasks)}]", flush=True)
        df = pd.DataFrame(rows)
    if df.empty:
        parser.error("No benchmark pairs were loaded")

    # Decisions are rebuilt from the stored margins, so a reused CSV and a fresh
    # run apply the same tie tolerance.
    df = add_decisions(df)
    df.to_csv(csv_path, index=False)
    summary = build_summary(df)
    with open(os.path.join(results_dir, "orientation_flip_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    make_figures(df, summary, figures_dir)
    write_report(summary, os.path.join(output_dir, "ORIENTATION_FLIP_REPORT.md"))

    macro = summary["macro12"]
    print(json.dumps({
        "macro12_flip_rate_any": {
            score: macro["flip_rates"][score]["any"]["rate"] for score in SCORES},
        "macro12_flip_rate_by_negation": {
            score: {flip: macro["flip_rates"][score][flip]["rate"] for flip in FLIPS}
            for score in SCORES},
        "macro12_accuracy": {
            "gdeci": macro["accuracy"]["gdeci_orig"]["accuracy_all"],
            "sym_orig": macro["accuracy"]["sym_orig"]["accuracy_all"],
            "symsum": macro["accuracy"]["symsum"]["accuracy_all"]},
        "tuebingen_accuracy": {
            "gdeci": summary["tuebingen"]["unweighted"]["accuracy"]["gdeci_orig"]["accuracy_all"],
            "sym_orig": summary["tuebingen"]["unweighted"]["accuracy"]["sym_orig"]["accuracy_all"],
            "symsum": summary["tuebingen"]["unweighted"]["accuracy"]["symsum"]["accuracy_all"]},
        "sanity_checks_passed": summary["sanity_checks"]["all_passed"],
    }, indent=2))
    print("wrote", output_dir)


if __name__ == "__main__":
    main()
