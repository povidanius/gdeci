"""Accuracy ranking figure: every method's macro-12 and Tuebingen accuracy.

Point estimates are computed from the per-pair outputs in the repository, not
transcribed, so the figure cannot drift from the benchmark it illustrates:

    GDECI, EdgeMass   loci_dataset_benchmark/results/per_pair.csv  (run_benchmark.py)
    QCCD, GRCI, CAM,  loci/baseline_results/*.tab -- the LOCI authors' released
    IGCI, IGCI_G,     per-pair scores, rescored with that repository's own rule
    RESIT, RESIT_std
    LOCI              loci_dataset_benchmark/results_loci/loci.csv  (run_loci.py)
    RECI              loci_dataset_benchmark/results/reci.csv       (run_reci.py)
    QPE-k             loci_dataset_benchmark/results/qpek.csv       (run_qpek.py)

`per_pair.csv` is required; every other source is optional and its row is simply
left out of the figure when the file is absent, with a note on stdout.  That way
the figure can be drawn after a partial re-run instead of failing.

Intervals are Wilson 95% intervals, as the axis label says: at the pooled pair
count for the macro mean, and at 99 for Tuebingen.  Both match
loci_dataset_benchmark/summarize_direct.py.

    python accuracy_ranking_figure.py     -> figures/gdeci_accuracy_ranking.{pdf,png}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "loci_dataset_benchmark")
sys.path.insert(0, BENCH)

import datasets as D
from make_table import baseline_decisions, decisions, positive, wilson

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["FreeSerif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.0,
    "xtick.major.size": 3,
    "pdf.fonttype": 42,
})

# Display label -> column in the frame make_table's helpers produce.  The order
# here is only a tie-break; the panels sort by accuracy.
BASELINE_ROWS = [("QCCD", "QCCD"), ("GRCI", "GRCI"), ("CAM", "CAM"),
                 ("IGCI", "IGCI"), (r"IGCI$_{\mathrm{G}}$", "IGCI_G"),
                 ("RESIT", "RESIT"), (r"RESIT$_{\mathrm{std}}$", "RESIT_std")]
# Display label -> (path relative to loci_dataset_benchmark, score column).
# Correct when the score is positive, the convention of all three scripts.
RERUN_ROWS = [("LOCI", os.path.join("results_loci", "loci.csv"), "loci"),
              ("RECI", os.path.join("results", "reci.csv"), "reci"),
              ("QPE-k", os.path.join("results", "qpek.csv"), "qpek")]
# Display label -> column of make_table.decisions(per_pair).
OURS_ROWS = [("GDECI", "Lap_unif"), ("EdgeMass (control)", "EdgeMass")]


def per_pair_correctness():
    """{label: {suite: Series of per-pair correctness}} from what is on disk."""
    path = os.path.join(BENCH, "results", "per_pair.csv")
    if not os.path.exists(path):
        sys.exit(f"missing {path}\n"
                 "run ./run_laplacian_direct_estimator.sh first")
    per = pd.read_csv(path)

    rerun = {}
    for label, rel, col in RERUN_ROWS:
        full = os.path.join(BENCH, rel)
        if os.path.exists(full):
            rerun[label] = (pd.read_csv(full), col)
        else:
            print(f"note: {rel} absent, omitting {label} from the figure")

    out = {}
    for name in [b[0] for b in D.BENCHMARKS]:
        d = per[per.benchmark == name].set_index("pair_id")
        ours = decisions(d)
        frame = pd.DataFrame(index=ours.index)
        for label, col in OURS_ROWS:
            frame[label] = ours[col].astype(float)
        base = baseline_decisions(name).reindex(ours.index)
        for label, col in BASELINE_ROWS:
            frame[label] = base[col].astype(float)
        for label, (table, col) in rerun.items():
            sub = table[table.benchmark == name].set_index("pair_id")
            frame[label] = positive(sub[col]).reindex(ours.index)
        out[name] = frame
    return out


def build_data():
    """{label: (macro12, lo, hi, tuebingen, lo, hi)}, Wilson intervals."""
    correct = per_pair_correctness()
    synthetic = [b[0] for b in D.BENCHMARKS if b[0] != "Tuebingen"]
    tue = correct["Tuebingen"]
    data = {}
    for label in correct["Tuebingen"].columns:
        # NaN = the pair was not scored by that method; never counted as wrong.
        macro = float(np.nanmean([correct[s][label].mean(skipna=True)
                                  for s in synthetic]))
        n_macro = int(sum(correct[s][label].notna().sum() for s in synthetic))
        t = float(tue[label].mean(skipna=True))
        n_tue = int(tue[label].notna().sum())
        lo1, hi1 = wilson(macro, n_macro)
        lo2, hi2 = wilson(t, n_tue)
        data[label] = (macro, round(lo1, 2), round(hi1, 2),
                       t, round(lo2, 2), round(hi2, 2))
    return data


data = build_data()
print(f"{len(data)} methods, macro-12 and Tuebingen, from per-pair outputs")
for label, v in sorted(data.items(), key=lambda kv: -kv[1][0]):
    print(f"  {label:26s} macro {v[0]:.3f} [{v[1]:.2f},{v[2]:.2f}]   "
          f"Tuebingen {v[3]:.3f} [{v[4]:.2f},{v[5]:.2f}]")

GDECI_C = "#1f6fbf"; OTHER_C = "#333333"; CTRL_C = "#9a9a9a"

def panel(ax, idx, title):
    items = sorted(data.items(), key=lambda kv: kv[1][idx])  # ascending -> best on top
    for y, (name, v) in enumerate(items):
        m, lo, hi = v[idx], v[idx+1], v[idx+2]
        ctrl = "control" in name
        gd = name == "GDECI"
        c = GDECI_C if gd else (CTRL_C if ctrl else OTHER_C)
        if gd:
            ax.axhspan(y-0.45, y+0.45, color=GDECI_C, alpha=0.08, lw=0)
        ax.hlines(y, lo, hi, color=c, lw=1.1 if gd else 0.8)
        ax.plot(m, y, marker="D" if gd else "o",
                ms=5.2 if gd else 4.2,
                mfc="white" if ctrl else c, mec=c, mew=0.9, zorder=3)
        ax.text(hi + 0.012, y, f"{m:.3f}", va="center", ha="left",
                fontsize=7.2, color=c, fontweight="bold" if gd else "normal")
    ax.set_yticks(range(len(items)))
    labels = ax.set_yticklabels([k for k, _ in items])
    for lab, (k, _) in zip(labels, items):
        if k == "GDECI":
            lab.set_color(GDECI_C); lab.set_fontweight("bold")
        elif "control" in k:
            lab.set_color(CTRL_C)
    ax.axvline(0.5, color="#b0b0b0", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(0.15, 1.0)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.text(0.5, len(items)-0.45, "chance", ha="center", va="bottom", fontsize=6.8, color="#8a8a8a")
    ax.set_ylim(-0.7, len(items) + 0.1)
    ax.set_xlabel("accuracy (95% Wilson interval)")
    ax.set_title(title, fontsize=8.8, pad=4)
    ax.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
panel(axes[0], 0, "(a) Synthetic suites, macro mean (1800 pairs)")
panel(axes[1], 3, "(b) Tuebingen, real-world (99 pairs)")
fig.tight_layout(w_pad=2.2)
out_dir = os.path.join(HERE, "figures")
os.makedirs(out_dir, exist_ok=True)
for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
    path = os.path.join(out_dir, f"gdeci_accuracy_ranking.{ext}")
    fig.savefig(path, bbox_inches="tight", **kw)
    print("wrote", path)