"""
Utilities for mixed-effects models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

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

# Valence

def fit_valence_model(df, score_col, random_slope=True):
    """
    Fit LMM for dwell time with depression x valence interaction.
    Expects long-format df with columns: uid, score_col, trial_norm, valence, dwell_time.
    Formula: dwell_time ~ score_col * C(valence) + (trial_norm | uid)
    """
    data = df[["uid", score_col, "trial_norm", "valence", "dwell_time"]].dropna()

    if len(data) < 100:
        print(f"Too few rows: {len(data)}")
        return None

    formula = f"dwell_time ~ {score_col} * C(valence, Treatment(reference='neutral'))"

    if random_slope:
        try:
            model = smf.mixedlm(formula, data=data, groups=data["uid"],
                                re_formula="~trial_norm")
            return model.fit(reml=True, method="lbfgs", maxiter=500)
        except Exception as e:
            print(f"Random slope failed ({e}), falling back to intercept-only")

    model = smf.mixedlm(formula, data=data, groups=data["uid"])
    return model.fit(reml=True, method="lbfgs", maxiter=500)


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

    # bar chart: mean dwell by valence x depression group
    grouped = df.groupby(["valence", "group"])["dwell_time"].mean().unstack()
    grouped.plot(kind="bar", ax=axes[0], edgecolor="white",
                 color=["#2196F3", "#F44336"])
    axes[0].set_title("Mean dwell time by valence and depression")
    axes[0].set_xlabel("Valence")
    axes[0].set_ylabel("Dwell time (ms)")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # interaction plot: dwell time per valence, lines for depression groups
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


def compare_scales_valence(phq_result, bdi_result):
    """
    Print side-by-side comparison of PHQ-9 and BDI valence model results.
    """
    phq_fe = phq_result.fe_params
    phq_pvals = phq_result.pvalues
    bdi_fe = bdi_result.fe_params
    bdi_pvals = bdi_result.pvalues

    all_params = sorted(set(phq_fe.index) | set(bdi_fe.index))

    rows = []
    for param in all_params:
        rows.append({
            "parameter": param,
            "phq9_coef": phq_fe.get(param, np.nan),
            "phq9_pval": phq_pvals.get(param, np.nan),
            "bdi_coef": bdi_fe.get(param, np.nan),
            "bdi_pval": bdi_pvals.get(param, np.nan),
        })

    df_comp = pd.DataFrame(rows)
    print(df_comp.to_string(index=False))


# session proportion

def fit_proportion_glm(df, outcome_col, score_col):
    """
    Fit a binomial GLM: proportion ~ depression_score.
    For session-level outcomes bounded between 0 and 1.
    """
    data = df[[outcome_col, score_col]].dropna()
    y = data[outcome_col]
    X = sm.add_constant(data[score_col])
    model = sm.GLM(y, X, family=sm.families.Binomial())
    return model.fit()


def plot_proportion(df, outcome_col, raw_score_col, score_label, title):
    """
    Scatter plot of session-level proportion vs depression score with GLM fit line.
    """
    data = df[[outcome_col, raw_score_col]].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data[raw_score_col], data[outcome_col], s=8, alpha=0.3, color="steelblue")

    # fit line
    x_sorted = np.linspace(data[raw_score_col].min(), data[raw_score_col].max(), 100)
    X_pred = sm.add_constant(x_sorted)

    # standardize x for prediction using same params as the model was fit on
    score_z = (data[raw_score_col] - data[raw_score_col].mean()) / data[raw_score_col].std()
    X_fit = sm.add_constant(score_z)
    model = sm.GLM(data[outcome_col], X_fit, family=sm.families.Binomial()).fit()

    x_z = (x_sorted - data[raw_score_col].mean()) / data[raw_score_col].std()
    y_pred = model.predict(sm.add_constant(x_z))
    ax.plot(x_sorted, y_pred, "r-", linewidth=2, alpha=0.8)

    ax.set_xlabel(f"{score_label} Score")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_proportion_models(models_dict):
    """
    Print summary table of all proportion GLM models.
    models_dict: {label: fitted GLM result}
    """
    rows = []
    for label, result in models_dict.items():
        params = result.params
        pvalues = result.pvalues
        # second parameter is the score effect (first is intercept)
        score_param = params.index[1]
        rows.append({
            "model": label,
            "coef": params[score_param],
            "p_value": pvalues[score_param],
            "aic": result.aic,
        })

    df_summary = pd.DataFrame(rows)
    df_summary["significant"] = df_summary["p_value"] < 0.05
    print(df_summary.to_string(index=False))

# binary outcomes

def fit_binary_lpm(df, outcome_col, score_col, random_slope=True):
    """
    Fit a linear probability model (LMM on binary 0/1 outcome).
    Approximation to GLMM since statsmodels lacks native logistic GLMM.
    Formula: outcome ~ score * C(valence) + (trial_norm | uid)
    Expects df with columns: uid, trial_norm, valence, score_col, outcome_col.
    """
    data = df[["uid", score_col, "trial_norm", "valence", outcome_col]].dropna()

    if len(data) < 100:
        print(f"Too few rows: {len(data)}")
        return None

    formula = f"{outcome_col} ~ {score_col} * C(valence, Treatment(reference='neutral'))"

    if random_slope:
        try:
            model = smf.mixedlm(formula, data=data, groups=data["uid"],
                                re_formula="~trial_norm")
            return model.fit(reml=True, method="lbfgs", maxiter=500)
        except Exception as e:
            print(f"Random slope failed ({e}), falling back to intercept-only")

    model = smf.mixedlm(formula, data=data, groups=data["uid"])
    return model.fit(reml=True, method="lbfgs", maxiter=500)


def plot_binary_proportions(df, outcome_col, raw_score_col, score_label, outcome_label):
    """
    Plot proportion of binary outcome by valence, split by median depression score.
    """
    data = df.copy()
    median_score = data[raw_score_col].median()
    data["group"] = np.where(data[raw_score_col] < median_score,
                             f"Low {score_label}", f"High {score_label}")

    grouped = data.groupby(["valence", "group"])[outcome_col].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax, color=["#2196F3", "#F44336"], edgecolor="white")
    ax.set_xlabel("Valence")
    ax.set_ylabel(outcome_label)
    ax.set_title(f"{outcome_label} by valence and {score_label}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_scales_binary(models_dict):
    """
    Print summary of binary LPM models showing depression x valence interaction terms.
    """
    rows = []
    for label, result in models_dict.items():
        fe = result.fe_params
        pvals = result.pvalues

        for param in fe.index:
            if ":" in param:  # interaction terms
                rows.append({
                    "model": label,
                    "parameter": param,
                    "coef": fe[param],
                    "p_value": pvals[param],
                    "significant": pvals[param] < 0.05,
                })

    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))