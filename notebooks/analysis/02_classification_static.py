# Databricks notebook source
# MAGIC %md
# MAGIC # Classification: Static Features
# MAGIC
# MAGIC Session-level means of all eye-tracking metrics are used as features to predict depression severity. Each metric is averaged across all stimulus scenes in a session, producing one value per session.
# MAGIC
# MAGIC Three tasks are tested: 
# MAGIC - binary classification (depressed vs not)
# MAGIC - multi-class (5 severity groups)
# MAGIC - regression (predict score directly). 
# MAGIC
# MAGIC Three models are compared: 
# MAGIC - Logistic/Ridge Regression
# MAGIC - Random Forest
# MAGIC - XGBoost 
# MAGIC
# MAGIC Across three feature sets:
# MAGIC - all features
# MAGIC - statistically significant
# MAGIC - theory-driven
# MAGIC
# MAGIC Validation uses GroupKFold grouped by user to prevent data leakage.

# COMMAND ----------

# MAGIC %pip install xgboost -q

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import numpy as np

from src.classification import (
    prepare_data, run_classification_binary, run_classification_multiclass, run_regression,
    plot_best_classification_binary, plot_best_classification_multiclass, plot_best_regression,
    plot_summary, plot_feature_importance,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build session-level dataset

# COMMAND ----------

scene_metrics = spark.table("anima.scene_metrics")
forms = spark.table("anima.forms")

stimulus_scenes = scene_metrics.filter(F.col("scene_type") == "stimulus")

session_metrics = stimulus_scenes.groupBy("session_id").agg(
    F.mean("fixation_count").alias("avg_fixation_count"),
    F.mean("mean_fixation_duration_ms").alias("avg_fixation_duration_ms"),
    F.mean("total_fixation_duration_ms").alias("avg_total_fixation_duration_ms"),
    F.mean("fixation_rate_per_sec").alias("avg_fixation_rate"),
    F.mean("fixation_bias").alias("avg_fixation_bias"),

    F.mean("scanpath_length").alias("avg_scanpath_length"),
    F.mean("saccade_count").alias("avg_saccade_count"),
    F.mean("saccade_rate_per_sec").alias("avg_saccade_rate"),
    F.mean("mean_saccade_amplitude").alias("avg_saccade_amplitude"),

    F.mean("blink_count").alias("avg_blink_count"),
    F.mean("blink_rate_per_min").alias("avg_blink_rate"),

    F.mean("transition_matrix_density").alias("avg_transition_density"),
    F.mean("gaze_transition_entropy").alias("avg_gaze_entropy"),

    F.mean("first_fixation_duration_ms").alias("avg_first_fixation_duration_ms"),
    F.mean("second_fixation_duration_ms").alias("avg_second_fixation_duration_ms"),

    F.avg(F.when(F.col("first_fixation_valence") == "negative", 1).otherwise(0)).alias("first_fix_prob_negative"),
    F.avg(F.when(F.col("first_fixation_valence") == "positive", 1).otherwise(0)).alias("first_fix_prob_positive"),
    F.avg(F.when(F.col("first_fixation_valence") == "neutral", 1).otherwise(0)).alias("first_fix_prob_neutral"),
    F.avg(F.when(F.col("second_fixation_valence") == "negative", 1).otherwise(0)).alias("second_fix_prob_negative"),
    F.avg(F.when(F.col("second_fixation_valence") == "positive", 1).otherwise(0)).alias("second_fix_prob_positive"),
    F.avg(F.when(F.col("second_fixation_valence") == "neutral", 1).otherwise(0)).alias("second_fix_prob_neutral"),

    F.mean("dwell_time_ms_negative").alias("avg_dwell_time_negative"),
    F.mean("dwell_time_ms_positive").alias("avg_dwell_time_positive"),
    F.mean("dwell_time_ms_neutral").alias("avg_dwell_time_neutral"),
    F.mean("dwell_time_500ms_negative").alias("avg_dwell_500ms_negative"),
    F.mean("dwell_time_500ms_positive").alias("avg_dwell_500ms_positive"),
    F.mean("dwell_time_500ms_neutral").alias("avg_dwell_500ms_neutral"),

    F.mean("fixation_proportion_negative").alias("avg_fix_proportion_negative"),
    F.mean("fixation_proportion_positive").alias("avg_fix_proportion_positive"),
    F.mean("fixation_proportion_neutral").alias("avg_fix_proportion_neutral"),

    F.mean("fixation_count_negative").alias("avg_fix_count_negative"),
    F.mean("fixation_count_positive").alias("avg_fix_count_positive"),
    F.mean("fixation_count_neutral").alias("avg_fix_count_neutral"),

    F.mean("revisit_count_negative").alias("avg_revisit_count_negative"),
    F.mean("revisit_count_positive").alias("avg_revisit_count_positive"),
    F.mean("revisit_count_neutral").alias("avg_revisit_count_neutral"),

    F.mean("ttff_ms_negative").alias("avg_ttff_negative"),
    F.mean("ttff_ms_positive").alias("avg_ttff_positive"),
    F.mean("ttff_ms_neutral").alias("avg_ttff_neutral"),
)

# Join with forms
df_joined = session_metrics.join(
    forms.select("session_id", "uid", "phq9_score", "phq9_severity", "bdi_score", "bdi_severity"),
    on="session_id",
    how="inner",
)

df = df_joined.toPandas()
print(f"Dataset: {len(df)} sessions, {df['uid'].nunique()} unique users")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define targets and feature sets

# COMMAND ----------

df["phq9_depressed"] = (df["phq9_score"] >= 10).astype(int)
df["phq9_severity_class"] = df["phq9_severity"].map(
    {"minimal": 0, "mild": 1, "moderate": 2, "moderately_severe": 3, "severe": 4}
)

print(f"Binary (>=10): {df['phq9_depressed'].value_counts().to_dict()}")
print(f"Multi-class: {df['phq9_severity_class'].value_counts().sort_index().to_dict()}")

# COMMAND ----------

ALL_FEATURES = [
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

SIGNIFICANT_FEATURES = [
    "avg_fixation_bias",
    "avg_fix_proportion_positive",
    "avg_fix_count_positive",
    "avg_dwell_time_positive",
    "avg_blink_count",
    "avg_blink_rate",
    "avg_fixation_rate",
    "avg_saccade_rate",
    "avg_fixation_count",
    "avg_saccade_count",
    "avg_transition_density",
    "avg_saccade_amplitude",
    "second_fix_prob_neutral",
    "avg_ttff_negative",
]

THEORY_FEATURES = [
    "avg_fixation_bias",
    "avg_dwell_time_negative",
    "avg_dwell_time_positive",
    "avg_fix_proportion_negative",
    "avg_fix_proportion_positive",
    "first_fix_prob_negative",
    "first_fix_prob_positive",
    "avg_revisit_count_negative",
    "avg_revisit_count_positive",
    "avg_blink_rate",
    "avg_scanpath_length",
]

FEATURE_SETS = {
    "All Features": ALL_FEATURES,
    "Significant Only": SIGNIFICANT_FEATURES,
    "Theory-Driven": THEORY_FEATURES,
}

for name, feats in FEATURE_SETS.items():
    print(f"{name}: {len(feats)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prepare data

# COMMAND ----------

target_cols = ["phq9_score", "phq9_depressed", "phq9_severity_class"]
df_clean, groups = prepare_data(df, FEATURE_SETS, target_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Binary classification (PHQ-9 >= 10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Run classification

# COMMAND ----------

y_binary = df_clean["phq9_depressed"].values
binary_df = run_classification_binary(df_clean, FEATURE_SETS, y_binary, groups)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Results

# COMMAND ----------

print(binary_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Best model

# COMMAND ----------

plot_best_classification_binary(df_clean, FEATURE_SETS, y_binary, groups, binary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Multi-class classification (5 severity groups)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Run classification

# COMMAND ----------

PHQ9_LABELS = ["Minimal", "Mild", "Moderate", "Moderately Severe", "Severe"]
y_multi = df_clean["phq9_severity_class"].values.astype(int)
multi_df = run_classification_multiclass(df_clean, FEATURE_SETS, y_multi, groups)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Results

# COMMAND ----------

print(multi_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Best model

# COMMAND ----------

plot_best_classification_multiclass(df_clean, FEATURE_SETS, y_multi, groups, multi_df, PHQ9_LABELS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Regression (predict PHQ-9 score)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Run regression

# COMMAND ----------

y_reg = df_clean["phq9_score"].values
reg_df = run_regression(df_clean, FEATURE_SETS, y_reg, groups)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Results

# COMMAND ----------

print(reg_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 Best model

# COMMAND ----------

plot_best_regression(df_clean, FEATURE_SETS, y_reg, groups, reg_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Feature importance

# COMMAND ----------

plot_feature_importance(df_clean, ALL_FEATURES, y_binary, title="Feature importance (binary, all features)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary

# COMMAND ----------

feature_order = list(FEATURE_SETS.keys())
plot_summary(binary_df, multi_df, reg_df, feature_order, title="Static features")
