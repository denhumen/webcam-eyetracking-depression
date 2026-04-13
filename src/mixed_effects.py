"""
Utilities for mixed-effects models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def fit_mixed_model(df, metric, score_col, random_slope=True):
    """
    Fit a linear mixed-effects model for one metric.
    Formula: metric ~ score_col * trial_norm + (trial_norm | uid)
    """
    data = df[["uid", score_col, "trial_norm", metric]].dropna()

    if len(data) < 100:
        print(f"Skipping {metric}: only {len(data)} valid rows")
        return None

    formula = f"{metric} ~ {score_col} * trial_norm"

    if random_slope:
        try:
            model = smf.mixedlm(formula, data=data, groups=data["uid"],
                                re_formula="~trial_norm")
            return model.fit(reml=True, method="lbfgs", maxiter=500)
        except Exception as e:
            print(f"Random slope failed for {metric} ({e}), falling back to intercept-only")

    model = smf.mixedlm(formula, data=data, groups=data["uid"])
    return model.fit(reml=True, method="lbfgs", maxiter=500)


def fit_all_metrics(df, metrics, score_col):
    """
    Fit mixed-effects models for all metrics, print results as they run.
    Returns dict of {metric: fitted model}.
    """
    results = {}
    for metric in metrics:
        print(f"--- {metric} ---")
        result = fit_mixed_model(df, metric, score_col)
        if result is not None:
            results[metric] = result
            fe = result.fe_params
            pvals = result.pvalues
            for param in [score_col, "trial_norm", f"{score_col}:trial_norm"]:
                if param in fe.index:
                    star = "***" if pvals[param] < 0.001 else "**" if pvals[param] < 0.01 else "*" if pvals[param] < 0.05 else ""
                    print(f"  {param}: coef={fe[param]:.4f}, p={pvals[param]:.4e} {star}")
        print()
    return results


def build_summary_df(results_dict, score_col):
    """
    Build a summary DataFrame from fitted mixed-effects models.
    """
    interaction_col = f"{score_col}:trial_norm"
    rows = []

    for metric, result in results_dict.items():
        fe = result.fe_params
        pvals = result.pvalues
        row = {"metric": metric}

        for param in [score_col, "trial_norm", interaction_col]:
            if param in fe.index:
                row[f"{param}_coef"] = fe[param]
                row[f"{param}_pval"] = pvals[param]
            else:
                row[f"{param}_coef"] = np.nan
                row[f"{param}_pval"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


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


def compare_scales(phq_summary, bdi_summary):
    """
    Print side-by-side comparison of PHQ-9 and BDI results.
    """
    df_comp = phq_summary[["metric", "phq9_z_coef", "phq9_z_pval"]].merge(
        bdi_summary[["metric", "bdi_z_coef", "bdi_z_pval"]], on="metric"
    )

    df_comp["sig_phq"] = df_comp["phq9_z_pval"] < 0.05
    df_comp["sig_bdi"] = df_comp["bdi_z_pval"] < 0.05

    print(df_comp[["metric", "phq9_z_coef", "phq9_z_pval", "bdi_z_coef", "bdi_z_pval"]]
          .sort_values("phq9_z_pval").to_string(index=False))

    phq_only = (df_comp["sig_phq"] & ~df_comp["sig_bdi"]).sum()
    bdi_only = (~df_comp["sig_phq"] & df_comp["sig_bdi"]).sum()
    both = (df_comp["sig_phq"] & df_comp["sig_bdi"]).sum()

    print()
    print(f"Significant in PHQ-9 only: {phq_only}")
    print(f"Significant in BDI only: {bdi_only}")
    print(f"Significant in both: {both}")