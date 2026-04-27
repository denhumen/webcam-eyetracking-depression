"""
Mixed-effects models testing depression & trial_position effects
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

def fit_one(df, metric, score):
    """
    Fit metric ~ score * trial_norm with random intercept + slope per user.
    If slope variance is on the boundary, refit with intercept only.

    Returns (result, structure) on success, or (None, "failed") if both
    fits fail (e.g. singular matrix on degenerate data).
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

    try:
        model = smf.mixedlm(formula, dsub, groups=dsub["uid"])
        result = model.fit(reml=True, method="lbfgs")
        return result, "intercept_only"
    except Exception:
        return None, "failed"

def _summary_row(metric, result, structure, score, extra_cols=None):
    """
    Build one row of the summary DataFrame from a fit result.
    extra_cols is a dict of additional columns to include (e.g. pair).
    """
    interaction = f"{score}:trial_norm"
    row = {
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
        "n_obs": int(result.nobs),
    }
    if extra_cols:
        row.update(extra_cols)
    return row

def fit_all_stratified(df, pair_invariant_metrics, pair_stratified_spec, score, pairs):
    """
    Fit LMMs for a mix of pair-invariant and pair-stratified metrics.
    """
    results = {}
    rows = []

    for metric in pair_invariant_metrics:
        result, structure = fit_one(df, metric, score)
        if result is None:
            print(f"[skip] {metric}: fit failed ({structure})")
            continue
        results[metric] = result
        rows.append(_summary_row(metric, result, structure, score, extra_cols={"pair_suffix": None}))

    for metric in pair_stratified_spec:
        for pair_name, pair_suffix, _valences in pairs:
            sub = df[df["scene_valence_pair"] == pair_name]
            if sub[metric].notna().sum() < 30:
                continue
            result, structure = fit_one(sub, metric, score)
            if result is None:
                print(f"[skip] {metric} on {pair_suffix}: fit failed ({structure})")
                continue
            key = f"{metric}__{pair_suffix}"
            results[key] = result
            rows.append(_summary_row(metric, result, structure, score, extra_cols={"pair_suffix": pair_suffix}))

    return results, pd.DataFrame(rows)


def apply_fdr(summary, score):
    """
    Benjamini-Hochberg correction across rows of the summary DataFrame.
    Applied independently to each p-value column.
    """
    interaction = f"{score}:trial_norm"
    for col in [f"{score}_pval", "trial_norm_pval", f"{interaction}_pval"]:
        _, p_fdr, _, _ = multipletests(summary[col].values, method="fdr_bh")
        summary[col + "_fdr"] = p_fdr
    return summary


def _plot_single_trajectory(ax, metric, df, summary_row, score_col, score_label, raw_score_col, median_score, n_sig, p_int):
    """
    Render one metric's trajectory onto a given axis
    """
    for label, subset, color in [
        (f"Low {score_label} (<{median_score:.0f})",
         df[df[raw_score_col] < median_score], "#2196F3"),
        (f"High {score_label} (>={median_score:.0f})",
         df[df[raw_score_col] >= median_score], "#F44336"),
    ]:
        subset = subset.copy()
        subset["trial_bin"] = pd.cut(subset["trial_num"], bins=10, labels=False)
        means = subset.groupby("trial_bin")[metric].mean()
        sems = subset.groupby("trial_bin")[metric].sem()
        ax.plot(means.index, means.values, color=color, linewidth=2.5, label=label)
        ax.fill_between(means.index, means.values - sems.values, means.values + sems.values, color=color, alpha=0.15)

    bg_colors = {3: "#c8e6c9", 2: "#e8f5e9", 1: "#f1f8e9", 0: "white"}
    ax.set_facecolor(bg_colors[n_sig])
    star = "***" if p_int < 0.001 else "**" if p_int < 0.01 else "*" if p_int < 0.05 else "n.s."
    ax.set_title(f"{metric} ({n_sig}/3 sig, interaction {star})")
    ax.set_xlabel("Trial bin")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_trajectories(df, summary_df, score_col, score_label, raw_score_col, metrics, pair_filter=None, pair_label=None, save_individual=False, save_grid=False, show=True):
    """
    Plot metric trajectories over trial bins, split by median depression score.
    """
    if pair_filter is not None:
        df = df[df["scene_valence_pair"] == pair_filter]
        rows = summary_df[summary_df.get("pair_suffix", None) == pair_label]
    else:
        rows = summary_df[summary_df.get("pair_suffix", pd.NA).isna()] \
            if "pair_suffix" in summary_df.columns else summary_df

    if rows.empty or len(metrics) == 0:
        print(f"No metrics to plot for pair_filter={pair_filter}.")
        return

    metrics = [m for m in metrics if m in rows["metric"].values]
    interaction_col = f"{score_col}:trial_norm"
    median_score = df[raw_score_col].median()

    def count_sig(metric):
        row = rows[rows["metric"] == metric].iloc[0]
        count = 0
        for col in [f"{score_col}_pval", "trial_norm_pval", f"{interaction_col}_pval"]:
            if row.get(col, 1.0) < 0.05:
                count += 1
        return count

    metrics = sorted(metrics, key=count_sig, reverse=True)

    metric_meta = {}
    for metric in metrics:
        row = rows[rows["metric"] == metric].iloc[0]
        metric_meta[metric] = {
            "summary_row": row,
            "n_sig": count_sig(metric),
            "p_int": row[f"{interaction_col}_pval"],
        }

    if save_individual:
        from src.visualization.io import save_figure
        subfolder_parts = ["lmm_temporal"]
        if pair_label:
            subfolder_parts.append(pair_label)
        subfolder = "/".join(subfolder_parts)

        for metric in metrics:
            meta = metric_meta[metric]
            fig_single, ax_single = plt.subplots(figsize=(7, 5))
            _plot_single_trajectory(
                ax_single, metric, df, meta["summary_row"],
                score_col, score_label, raw_score_col, median_score,
                meta["n_sig"], meta["p_int"],
            )
            fig_single.tight_layout()
            save_figure(fig_single, name=f"{metric}_{score_col}", subfolder=subfolder, close=True)

    n = len(metrics)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        meta = metric_meta[metric]
        _plot_single_trajectory(
            axes[idx], metric, df, meta["summary_row"],
            score_col, score_label, raw_score_col, median_score,
            meta["n_sig"], meta["p_int"],
        )

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    if pair_label:
        fig.suptitle(f"Trajectories for pair = {pair_label}", fontsize=14)
    plt.tight_layout()

    if save_grid:
        from src.visualization.io import save_figure
        grid_subfolder = f"lmm_temporal/{pair_label}" if pair_label else "lmm_temporal"
        grid_name = f"_grid_{score_col}" if pair_label else f"_grid_{score_col}"
        save_figure(fig, name=grid_name, subfolder=grid_subfolder, close=False)

    if show:
        plt.show()
    else:
        plt.close(fig)
