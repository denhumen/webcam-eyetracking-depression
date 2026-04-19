"""
Mixed-effects models testing depression x valence interactions across multiple gaze outcomes
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


def melt_outcome(df_wide, value_cols, id_vars):
    """
    Reshape one outcome into long format with valence as a factor
    """
    df_long = pd.melt(
        df_wide,
        id_vars=id_vars,
        value_vars=value_cols,
        var_name="valence_col",
        value_name="y",
    )
    df_long["valence"] = df_long["valence_col"].str.rsplit("_", n=1).str[-1]
    return df_long.drop(columns=["valence_col"]).dropna(subset=["y"])


def fit_one(df_long, score):
    """
    Fit y ~ score * valence with a random intercept + slope per user.
    If the slope variance is on the boundary, refit with intercept only
    """
    formula = f"y ~ {score} * C(valence, Treatment(reference='neutral'))"
    dsub = df_long[["y", score, "trial_norm", "valence", "uid"]].dropna()

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


def fit_all(long_dfs, score):
    """
    Fit all outcomes for one depression score
    """
    results = {}
    rows = []
    neg = f"{score}:C(valence, Treatment(reference='neutral'))[T.negative]"
    pos = f"{score}:C(valence, Treatment(reference='neutral'))[T.positive]"

    for outcome, df_long in long_dfs.items():
        result, structure = fit_one(df_long, score)
        results[outcome] = result

        rows.append({
            "outcome": outcome,
            f"{score}_main_coef": result.params[score],
            f"{score}_main_pval": result.pvalues[score],
            "neg_int_coef": result.params[neg],
            "neg_int_pval": result.pvalues[neg],
            "neg_int_d": _cohens_d(result, neg),
            "pos_int_coef": result.params[pos],
            "pos_int_pval": result.pvalues[pos],
            "pos_int_d": _cohens_d(result, pos),
            "icc": _icc(result),
            "random_structure": structure,
        })

    return results, pd.DataFrame(rows)


def apply_fdr(summary):
    """
    Benjamini-Hochberg correction across outcomes, per interaction term
    """
    for col in ["neg_int_pval", "pos_int_pval"]:
        _, p_fdr, _, _ = multipletests(summary[col].values, method="fdr_bh")
        summary[col + "_fdr"] = p_fdr
    return summary

def plot_valence_effects(df, raw_score_col, score_label):
    """
    Plot mean dwell time by valence, split by median depression score.
    """
    median_score = df[raw_score_col].median()
    df = df.copy()
    df["group"] = np.where(df[raw_score_col] < median_score,
                           f"Low {score_label} (<{median_score:.0f})",
                           f"High {score_label} (>={median_score:.0f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    grouped = df.groupby(["valence", "group"])["dwell_time"].mean().unstack()
    grouped.plot(kind="bar", ax=axes[0], edgecolor="white",
                 color=["#2196F3", "#F44336"])
    axes[0].set_title("Mean dwell time by valence and depression")
    axes[0].set_xlabel("Valence")
    axes[0].set_ylabel("Dwell time (ms)")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    valence_order = ["negative", "neutral", "positive"]
    for group_name, color in [(f"Low {score_label} (<{median_score:.0f})", "#2196F3"),
                               (f"High {score_label} (>={median_score:.0f})", "#F44336")]:
        subset = df[df["group"] == group_name]
        means = subset.groupby("valence")["dwell_time"].mean().reindex(valence_order)
        sems = subset.groupby("valence")["dwell_time"].sem().reindex(valence_order)
        axes[1].errorbar(valence_order, means.values, yerr=sems.values,
                         color=color, linewidth=2.5, marker="o", markersize=8,
                         capsize=5, label=group_name)

    axes[1].set_title("Valence x depression interaction")
    axes[1].set_xlabel("Valence")
    axes[1].set_ylabel("Dwell time (ms)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
