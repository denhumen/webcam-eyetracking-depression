"""
Mixed-effects models testing depression & trial_position interactions across session-level gaze metrics
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.multitest import multipletests

BOUNDARY_THRESHOLD = 1e-6

def _at_boundary(result):
    try:
        return bool(np.any(np.diag(result.cov_re.values) < BOUNDARY_THRESHOLD))
    except Exception:
        return True


def _cohens_d(result, param):
    try:
        coef = result.params[param]
        group_var = float(np.diag(result.cov_re.values).sum())
        scale = float(result.scale)
        return coef / np.sqrt(group_var + scale)
    except Exception:
        return np.nan


def _icc(result):
    try:
        group_intercept_var = float(result.cov_re.values[0, 0])
        scale = float(result.scale)
        return group_intercept_var / (group_intercept_var + scale)
    except Exception:
        return np.nan


def fit_one(df, metric, score):
    """
    Fit metric ~ score * trial_norm with a random intercept + slope per user.
    If the slope variance is on the boundary, refit with intercept only
    """
    formula = f"{metric} ~ {score} * trial_norm"
    dsub = df[[metric, score, "trial_norm", "uid"]].dropna()

    try:
        model = smf.mixedlm(formula, dsub, groups=dsub["uid"], re_formula="~trial_norm")
        result = model.fit(reml=True, method="lbfgs")
        if result.converged and not _at_boundary(result):
            return result, "intercept+slope"
    except Exception:
        pass

    model = smf.mixedlm(formula, dsub, groups=dsub["uid"])
    result = model.fit(reml=True, method="lbfgs")
    return result, "intercept_only"


def fit_all(df, metrics, score):
    """
    Fit all metrics for one depression score
    """
    results = {}
    rows = []
    interaction = f"{score}:trial_norm"

    for metric in metrics:
        result, structure = fit_one(df, metric, score)
        results[metric] = result

        rows.append({
            "metric": metric,
            f"{score}_coef": result.params[score],
            f"{score}_pval": result.pvalues[score],
            f"{score}_d": _cohens_d(result, score),
            "trial_norm_coef": result.params["trial_norm"],
            "trial_norm_pval": result.pvalues["trial_norm"],
            f"{interaction}_coef": result.params[interaction],
            f"{interaction}_pval": result.pvalues[interaction],
            "icc": _icc(result),
            "random_structure": structure,
        })

    return results, pd.DataFrame(rows)


def apply_fdr(summary, score):
    """
    Benjamini-Hochberg correction across each p-value column
    """
    interaction = f"{score}:trial_norm"
    for col in [f"{score}_pval", "trial_norm_pval", f"{interaction}_pval"]:
        _, p_fdr, _, _ = multipletests(summary[col].values, method="fdr_bh")
        summary[col + "_fdr"] = p_fdr
    return summary

def plot_trajectories(df, summary_df, score_col, score_label, raw_score_col, metrics):
    """
    Plot attention metric trajectories over trials, split by median depression score.
    One subplot per metric.
    """
    n = len(metrics)
    if n == 0:
        print("No metrics to plot.")
        return

    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    interaction_col = f"{score_col}:trial_norm"
    median_score = df[raw_score_col].median()

    def count_sig(metric):
        row = summary_df[summary_df["metric"] == metric].iloc[0]
        count = 0
        for col in [f"{score_col}_pval", "trial_norm_pval", f"{score_col}:trial_norm_pval"]:
            if row.get(col, 1.0) < 0.05:
                count += 1
        return count
    
    metrics = sorted(metrics, key=count_sig, reverse=True)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for label, subset, color in [
            (f"Low {score_label} (<{median_score:.0f})", df[df[raw_score_col] < median_score], "#2196F3"),
            (f"High {score_label} (>={median_score:.0f})", df[df[raw_score_col] >= median_score], "#F44336"),
        ]:
            subset = subset.copy()
            subset["trial_bin"] = pd.cut(subset["trial_num"], bins=10, labels=False)
            means = subset.groupby("trial_bin")[metric].mean()
            sems = subset.groupby("trial_bin")[metric].sem()
            ax.plot(means.index, means.values, color=color, linewidth=2.5, label=label)
            ax.fill_between(means.index, means.values - sems.values, means.values + sems.values,
                            color=color, alpha=0.15)

        row = summary_df[summary_df["metric"] == metric].iloc[0]
        p_int = row[f"{interaction_col}_pval"]
        n_sig = count_sig(metric)

        bg_colors = {3: "#c8e6c9", 2: "#e8f5e9", 1: "#f1f8e9", 0: "white"}
        ax.set_facecolor(bg_colors[n_sig])

        star = "***" if p_int < 0.001 else "**" if p_int < 0.01 else "*" if p_int < 0.05 else "n.s."
        ax.set_title(f"{metric} ({n_sig}/3 sig, interaction {star})")
        ax.set_xlabel("Trial bin")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
