"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional

def plot_severity_trend(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    score_col: str,
    metric_label: Optional[str] = None,
    score_label: Optional[str] = None,
):
    """
    Plot metric vs depression severity: violin, box+strip, and scatter with Spearman correlation.
    """
    data = df[[score_col, group_col, metric_col]].dropna().copy()

    if len(data) < 20:
        print(f"Skipping {metric_col}: only {len(data)} valid rows")
        return None

    if metric_label is None:
        metric_label = metric_col
    if score_label is None:
        score_label = score_col

    n_groups = data[group_col].nunique()
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    palette = sns.color_palette("flare", n_colors=n_groups)

    sns.violinplot(data=data, x=group_col, y=metric_col, palette=palette, inner="quart", linewidth=1, ax=axes[0])
    axes[0].set_title("By severity group")
    axes[0].set_xlabel("")
    axes[0].set_ylabel(metric_label)

    sns.boxplot(data=data, x=group_col, y=metric_col, palette=palette, showfliers=False, boxprops={"alpha": 0.3}, ax=axes[1])
    sns.stripplot(data=data, x=group_col, y=metric_col, color=".25", size=1.5, alpha=0.3, jitter=True, ax=axes[1])
    axes[1].set_title("By session")
    axes[1].set_xlabel("Severity")
    axes[1].set_ylabel(metric_label)

    axes[2].scatter(data[score_col], data[metric_col], s=8, alpha=0.3, color="steelblue")
    mask = data[[score_col, metric_col]].dropna()
    if len(mask) > 2:
        z = np.polyfit(mask[score_col], mask[metric_col], 1)
        x_ends = [mask[score_col].min(), mask[score_col].max()]
        axes[2].plot(x_ends, np.poly1d(z)(x_ends), "r-", linewidth=2, alpha=0.8)
    axes[2].set_title("Score vs metric")
    axes[2].set_xlabel(score_label)
    axes[2].set_ylabel(metric_label)
    axes[2].grid(True, alpha=0.3)

    corr, p_val = stats.spearmanr(data[score_col], data[metric_col])
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    fig.suptitle(f"{sig} {metric_label} | r = {corr:.3f}; p = {p_val:.2e};", fontsize=13)
    plt.tight_layout()
    plt.show()

    return {"metric": metric_col, "spearman_r": corr, "p_value": p_val, "n": len(data)}
