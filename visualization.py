import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

def visualize_severity_trend(spark_df, variable_col, title=None):
    """
    Takes a Spark DataFrame, converts to Pandas, buckets PHQ-9 into 4 groups,
    and generates a 3-panel dashboard (Violin, Swarm, ECDF).
    
    Parameters:
    - spark_df: The Gold Layer Spark DataFrame
    - variable_col: The string name of the column to analyze (e.g., 'blink_rate')
    - title: (Optional) A nice title for the charts
    """
    
    try:
        pdf = spark_df.select("phq-9_score", variable_col).toPandas()
        
        pdf = pdf.dropna(subset=[variable_col, "phq-9_score"])
        
        bins = [-1, 5, 10, 15, 100]
        labels = ["1. Minimal (0-5)", "2. Mild (6-10)", "3. Moderate (11-15)", "4. Severe (16+)"]
        pdf['severity_group'] = pd.cut(pdf['phq-9_score'], bins=bins, labels=labels)
        
        pdf = pdf.sort_values('severity_group')
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        return

    if title is None:
        title = f"Analysis of {variable_col}"
        
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    colors = sns.color_palette("flare", n_colors=4)

    sns.violinplot(data=pdf, x='severity_group', y=variable_col, palette=colors, 
                   inner="quart", linewidth=1.5, ax=axes[0])
    axes[0].set_title("A. Distribution Shape (Violin)", fontsize=14)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=15)
    
    sns.boxplot(data=pdf, x='severity_group', y=variable_col, palette=colors, 
                showfliers=False, boxprops={'alpha': 0.3}, ax=axes[1])
    sns.swarmplot(data=pdf, x='severity_group', y=variable_col, color=".2", size=2.5, alpha=0.7, ax=axes[1])
    axes[1].set_title("B. Individual Density (Swarm)", fontsize=14)
    axes[1].set_xlabel("Depression Severity")
    axes[1].tick_params(axis='x', rotation=15)

    sns.ecdfplot(data=pdf, x=variable_col, hue='severity_group', palette=colors, 
                 linewidth=3, ax=axes[2])
    axes[2].set_title("C. Cumulative Probability (ECDF)", fontsize=14)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    corr, p_val = stats.spearmanr(pdf['phq-9_score'], pdf[variable_col])
    
    stats_text = f"Spearman Correlation: r={corr:.3f}, p={p_val:.4f}"
    
    plt.suptitle(f"{title}\n{stats_text}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Analysis Complete for {variable_col}")
    print(f"   {stats_text}")
    if p_val < 0.05:
        print("   ⭐ SIGNIFICANT TREND DETECTED")
    else:
        print("   ❌ No significant linear trend")