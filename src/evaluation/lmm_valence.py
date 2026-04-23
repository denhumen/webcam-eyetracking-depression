"""
Mixed-effects models testing depression × valence interactions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
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

def _pick_reference_and_focal(valences):
    """
    For a 2-valence pair, pick which is the reference category and which
    is the focal (interaction) category
    """
    if "negative" in valences:
        focal = "negative"
        reference = next(v for v in valences if v != "negative")
    else:
        focal = "positive"
        reference = "neutral"
    return reference, focal


def melt_one_pair(df_wide, outcome_cols, valences, id_vars):
    """
    Melt a single pair's per-valence outcome columns into long format.
    """
    keep_cols = [c for c in outcome_cols if any(c.endswith(f"_{v}") for v in valences)]
    df_long = pd.melt(
        df_wide,
        id_vars=id_vars,
        value_vars=keep_cols,
        var_name="valence_col",
        value_name="y",
    )
    df_long["valence"] = df_long["valence_col"].str.rsplit("_", n=1).str[-1]
    return df_long.drop(columns=["valence_col"]).dropna(subset=["y"])


def fit_one_pair(df_long, score, reference):
    """
    Fit y ~ score * C(valence, Treatment(reference)) + trial_norm
    with random intercept + trial_norm slope per user. Refit with
    intercept only if slope variance is on the boundary.

    Returns (result, structure) on success, or (None, "failed") if
    both fits error out
    """
    formula = (
        f"y ~ {score} * C(valence, Treatment(reference='{reference}')) "
        f"+ trial_norm"
    )
    dsub = df_long[["y", score, "trial_norm", "valence", "uid"]].dropna()

    try:
        model = smf.mixedlm(formula, dsub, groups=dsub["uid"], re_formula="~trial_norm")
        result = model.fit(reml=True, method="lbfgs")
        if result.converged and not _at_boundary(result):
            return result, "intercept+slope"
    except Exception:
        pass

    try:
        model = smf.mixedlm(formula, dsub, groups=dsub["uid"])
        result = model.fit(reml=True, method="lbfgs")
        return result, "intercept_only"
    except Exception:
        return None, "failed"


def fit_all_per_pair(df_wide, outcomes, score, pairs, id_vars):
    """
    Fit one LMM per (outcome, pair) combination
    """
    results = {}
    rows = []

    for outcome_name, outcome_cols in outcomes.items():
        for pair_name, pair_suffix, valences in pairs:
            reference, focal = _pick_reference_and_focal(valences)

            sub_wide = df_wide[df_wide["scene_valence_pair"] == pair_name]
            df_long = melt_one_pair(sub_wide, outcome_cols, valences, id_vars)

            if len(df_long) < 30:
                continue

            result, structure = fit_one_pair(df_long, score, reference)
            if result is None:
                print(f"[skip] {outcome_name} on {pair_suffix}: fit failed ({structure})")
                continue

            interaction_param = (
                f"{score}:C(valence, Treatment(reference='{reference}'))"
                f"[T.{focal}]"
            )
            valence_main = (
                f"C(valence, Treatment(reference='{reference}'))[T.{focal}]"
            )

            results[(outcome_name, pair_suffix)] = result
            rows.append({
                "outcome": outcome_name,
                "pair_suffix": pair_suffix,
                "reference": reference,
                "focal": focal,
                "interaction_coef": result.params.get(interaction_param, np.nan),
                "interaction_pval": result.pvalues.get(interaction_param, np.nan),
                "interaction_d": _cohens_d(result, interaction_param),
                "valence_main_coef": result.params.get(valence_main, np.nan),
                "valence_main_pval": result.pvalues.get(valence_main, np.nan),
                f"{score}_main_coef": result.params[score],
                f"{score}_main_pval": result.pvalues[score],
                "trial_norm_coef": result.params["trial_norm"],
                "trial_norm_pval": result.pvalues["trial_norm"],
                "icc": _icc(result),
                "random_structure": structure,
                "n_obs": int(result.nobs),
            })

    return results, pd.DataFrame(rows)


def apply_fdr(summary):
    """
    Benjamini-Hochberg correction across all (outcome, pair) fits, per
    interaction term. Only one interaction per row in the new per-pair design.
    """
    if "interaction_pval" in summary.columns:
        _, p_fdr, _, _ = multipletests(summary["interaction_pval"].values,
                                       method="fdr_bh")
        summary["interaction_pval_fdr"] = p_fdr
    return summary


def plot_pair_valence_effect(df_long, raw_score_col, score_label, pair_suffix, y_label="y"):
    """
    Plot the 2-valence contrast for a single pair, split by median
    depression score. Shows mean ± SEM per valence × group.
    """
    median_score = df_long[raw_score_col].median()
    df = df_long.copy()
    df["group"] = np.where(
        df[raw_score_col] < median_score,
        f"Low {score_label} (<{median_score:.0f})",
        f"High {score_label} (>={median_score:.0f})",
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    valences = sorted(df["valence"].unique())
    colors = {
        f"Low {score_label} (<{median_score:.0f})":  "#2196F3",
        f"High {score_label} (>={median_score:.0f})": "#F44336",
    }

    for group_name, color in colors.items():
        subset = df[df["group"] == group_name]
        means = subset.groupby("valence")["y"].mean().reindex(valences)
        sems = subset.groupby("valence")["y"].sem().reindex(valences)
        ax.errorbar(valences, means.values, yerr=sems.values, color=color, linewidth=2.5, marker="o", markersize=8, capsize=5, label=group_name)

    ax.set_title(f"{pair_suffix}: mean {y_label} by valence × {score_label}")
    ax.set_xlabel("Valence")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
