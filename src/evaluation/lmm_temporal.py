"""
Mixed-effects models testing depression & trial_position interactions across session-level gaze metrics
"""

import numpy as np
import pandas as pd
import warnings

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
