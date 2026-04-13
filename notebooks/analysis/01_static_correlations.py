# Databricks notebook source
# MAGIC %md
# MAGIC # Static correlations
# MAGIC
# MAGIC Each eye-tracking metric is averaged across all stimulus scenes in a session, producing one value per session per metric. We then check whether these session-level averages correlate with depression severity measured by PHQ-9 and BDI.
# MAGIC
# MAGIC For each metric, three plots are shown: distribution across severity groups (violin), individual sessions (box + strip), and scatter with Spearman correlation.
# MAGIC
# MAGIC **Structure:**
# MAGIC 1. Build session-level dataset
# MAGIC 2. PHQ-9 analysis: plots + correlation summary
# MAGIC 3. BDI analysis: plots + correlation summary
# MAGIC 4. Comparison: PHQ-9 vs BDI

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.session_visualization import plot_severity_trend

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build session-level dataset

# COMMAND ----------

scene_metrics = spark.table("anima.scene_metrics")

df_stimulus = scene_metrics.filter(F.col("scene_type") == "stimulus")

session_metrics = df_stimulus.groupBy("session_id").agg(
    # Fixation metrics
    F.mean("fixation_count").alias("avg_fixation_count"),
    F.mean("mean_fixation_duration_ms").alias("avg_fixation_duration_ms"),
    F.mean("total_fixation_duration_ms").alias("avg_total_fixation_duration_ms"),
    F.mean("fixation_rate_per_sec").alias("avg_fixation_rate"),
    F.mean("fixation_bias").alias("avg_fixation_bias"),

    # Scanpath metrics
    F.mean("scanpath_length").alias("avg_scanpath_length"),
    F.mean("saccade_count").alias("avg_saccade_count"),
    F.mean("saccade_rate_per_sec").alias("avg_saccade_rate"),
    F.mean("mean_saccade_amplitude").alias("avg_saccade_amplitude"),

    # Blink metrics
    F.mean("blink_count").alias("avg_blink_count"),
    F.mean("blink_rate_per_min").alias("avg_blink_rate"),

    # Transition metrics
    F.mean("transition_matrix_density").alias("avg_transition_density"),
    F.mean("gaze_transition_entropy").alias("avg_gaze_entropy"),

    # First / second fixation
    F.mean("first_fixation_duration_ms").alias("avg_first_fixation_duration_ms"),
    F.mean("second_fixation_duration_ms").alias("avg_second_fixation_duration_ms"),

    # First / second fixation probability per valence
    F.avg(F.when(F.col("first_fixation_valence") == "negative", 1).otherwise(0)).alias("first_fix_prob_negative"),
    F.avg(F.when(F.col("first_fixation_valence") == "positive", 1).otherwise(0)).alias("first_fix_prob_positive"),
    F.avg(F.when(F.col("first_fixation_valence") == "neutral", 1).otherwise(0)).alias("first_fix_prob_neutral"),
    F.avg(F.when(F.col("second_fixation_valence") == "negative", 1).otherwise(0)).alias("second_fix_prob_negative"),
    F.avg(F.when(F.col("second_fixation_valence") == "positive", 1).otherwise(0)).alias("second_fix_prob_positive"),
    F.avg(F.when(F.col("second_fixation_valence") == "neutral", 1).otherwise(0)).alias("second_fix_prob_neutral"),

    # Dwell time by valence
    F.mean("dwell_time_ms_negative").alias("avg_dwell_time_negative"),
    F.mean("dwell_time_ms_positive").alias("avg_dwell_time_positive"),
    F.mean("dwell_time_ms_neutral").alias("avg_dwell_time_neutral"),
    F.mean("dwell_time_500ms_negative").alias("avg_dwell_500ms_negative"),
    F.mean("dwell_time_500ms_positive").alias("avg_dwell_500ms_positive"),
    F.mean("dwell_time_500ms_neutral").alias("avg_dwell_500ms_neutral"),

    # Fixation proportion by valence
    F.mean("fixation_proportion_negative").alias("avg_fix_proportion_negative"),
    F.mean("fixation_proportion_positive").alias("avg_fix_proportion_positive"),
    F.mean("fixation_proportion_neutral").alias("avg_fix_proportion_neutral"),

    # Fixation count by valence
    F.mean("fixation_count_negative").alias("avg_fix_count_negative"),
    F.mean("fixation_count_positive").alias("avg_fix_count_positive"),
    F.mean("fixation_count_neutral").alias("avg_fix_count_neutral"),

    # Revisit count by valence
    F.mean("revisit_count_negative").alias("avg_revisit_count_negative"),
    F.mean("revisit_count_positive").alias("avg_revisit_count_positive"),
    F.mean("revisit_count_neutral").alias("avg_revisit_count_neutral"),

    # Time to first fixation by valence
    F.mean("ttff_ms_negative").alias("avg_ttff_negative"),
    F.mean("ttff_ms_positive").alias("avg_ttff_positive"),
    F.mean("ttff_ms_neutral").alias("avg_ttff_neutral"),
)


print(f"Session-level metrics: {session_metrics.count()} sessions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Join with forms data

# COMMAND ----------

df_forms = spark.table("anima.forms").select(
    "session_id", "uid", "phq9_score", "phq9_severity", "bdi_score", "bdi_severity"
)

df_joined = session_metrics.join(df_forms, on="session_id", how="inner")

print(f"Sessions with both metrics and forms: {df_joined.count()}")

df = df_joined.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define severity group order

# COMMAND ----------

PHQ9_ORDER = ["minimal", "mild", "moderate", "moderately_severe", "severe"]
BDI_ORDER = ["normal", "mild", "borderline", "moderate", "severe", "extreme"]

df["phq9_severity"] = pd.Categorical(df["phq9_severity"], categories=PHQ9_ORDER, ordered=True)
df["bdi_severity"] = pd.Categorical(df["bdi_severity"], categories=BDI_ORDER, ordered=True)

print("PHQ-9 severity distribution:")
print(df["phq9_severity"].value_counts().sort_index())
print()
print("BDI severity distribution:")
print(df["bdi_severity"].value_counts().sort_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. PHQ-9: metric plots

# COMMAND ----------

metrics_to_plot = [
    "avg_fixation_count", "avg_fixation_duration_ms", "avg_total_fixation_duration_ms",
    "avg_fixation_rate", "avg_fixation_bias",
    "avg_scanpath_length", "avg_saccade_count", "avg_saccade_rate", "avg_saccade_amplitude",
    "avg_blink_count", "avg_blink_rate",
    "avg_transition_density", "avg_gaze_entropy",
    "avg_first_fixation_duration_ms", "avg_second_fixation_duration_ms",
    "first_fix_prob_negative", "first_fix_prob_positive", "first_fix_prob_neutral",
    "second_fix_prob_negative", "second_fix_prob_positive", "second_fix_prob_neutral",
    "avg_dwell_time_negative", "avg_dwell_time_positive", "avg_dwell_time_neutral",
    "avg_dwell_500ms_negative", "avg_dwell_500ms_positive", "avg_dwell_500ms_neutral",
    "avg_fix_proportion_negative", "avg_fix_proportion_positive", "avg_fix_proportion_neutral",
    "avg_fix_count_negative", "avg_fix_count_positive", "avg_fix_count_neutral",
    "avg_revisit_count_negative", "avg_revisit_count_positive", "avg_revisit_count_neutral",
    "avg_ttff_negative", "avg_ttff_positive", "avg_ttff_neutral",
]

results_phq = []
for metric in metrics_to_plot:
    r = plot_severity_trend(
        df, metric,
        group_col="phq9_severity",
        score_col="phq9_score",
        score_label="PHQ-9 Score",
    )
    if r:
        results_phq.append(r)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. PHQ-9: correlation summary

# COMMAND ----------

df_phq = pd.DataFrame(results_phq).sort_values("p_value")
df_phq["significant"] = df_phq["p_value"] < 0.05

print("PHQ-9 correlation summary (sorted by p-value):\n")
for _, row in df_phq.iterrows():
    star = " * " if row["significant"] else "   "
    print(f"{star} r={row['spearman_r']:+.3f}; p={row['p_value']:.2e}; {row['metric']}")

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 8))

colors = ["#d32f2f" if sig else "#90a4ae" for sig in df_phq["significant"]]

ax.barh(range(len(df_phq)), df_phq["spearman_r"], color=colors, edgecolor="white", height=0.7)

ax.set_yticks(range(len(df_phq)))
ax.set_yticklabels(df_phq["metric"].values, fontsize=10)

ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Spearman r")
ax.set_title("PHQ-9: correlation with attention metrics")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. BDI: metric plots

# COMMAND ----------

results_bdi = []
for metric in metrics_to_plot:
    r = plot_severity_trend(
        df, metric,
        group_col="bdi_severity",
        score_col="bdi_score",
        score_label="BDI Score",
    )
    if r:
        results_bdi.append(r)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. BDI: correlation summary

# COMMAND ----------

df_bdi = pd.DataFrame(results_bdi).sort_values("p_value")
df_bdi["significant"] = df_bdi["p_value"] < 0.05

print("BDI correlation summary (sorted by p-value):")
print()
for _, row in df_bdi.iterrows():
    star = " * " if row["significant"] else "   "
    print(f"{star} r={row['spearman_r']:+.3f}; p={row['p_value']:.2e}; {row['metric']}")

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 8))

colors = ["#d32f2f" if sig else "#90a4ae" for sig in df_bdi["significant"]]

ax.barh(range(len(df_bdi)), df_bdi["spearman_r"], color=colors, edgecolor="white", height=0.7)

ax.set_yticks(range(len(df_bdi)))
ax.set_yticklabels(df_bdi["metric"].values, fontsize=10)

ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Spearman r")
ax.set_title("BDI: correlation with attention metrics (red = p < 0.05)")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Comparison: PHQ-9 vs BDI

# COMMAND ----------

df_comparison = df_phq[["metric", "spearman_r", "p_value"]].merge(
    df_bdi[["metric", "spearman_r", "p_value"]],
    on="metric", suffixes=("_phq", "_bdi")
)

df_comparison["sig_phq"] = df_comparison["p_value_phq"] < 0.05
df_comparison["sig_bdi"] = df_comparison["p_value_bdi"] < 0.05

print(df_comparison[["metric", "spearman_r_phq", "p_value_phq", "spearman_r_bdi", "p_value_bdi"]]
      .sort_values("p_value_phq").to_string(index=False))

phq_only = (df_comparison["sig_phq"] & ~df_comparison["sig_bdi"]).sum()
bdi_only = (~df_comparison["sig_phq"] & df_comparison["sig_bdi"]).sum()
both = (df_comparison["sig_phq"] & df_comparison["sig_bdi"]).sum()

print()
print(f"Significant in PHQ-9 only: {phq_only}")
print(f"Significant in BDI only: {bdi_only}")
print(f"Significant in both: {both}")
