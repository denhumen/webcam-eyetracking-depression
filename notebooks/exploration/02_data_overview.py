# Databricks notebook source
# MAGIC %md
# MAGIC # Data overview
# MAGIC
# MAGIC This notebook explores the dataset before running any analysis
# MAGIC
# MAGIC **Dataset exploration:**
# MAGIC - Eye-tracking recordings from the Anima.help platform, where users watch pairs of images (positive, negative and neutral) while a webcam tracks their gaze
# MAGIC - Each recording is called a session. Each session has ~50 stimulus scenes (image pairs) and ~50 fixation cross scenes (rest periods between image pairs)
# MAGIC - For each session, the user also filled out two depression questionnaires: PHQ-9 (9 questions, score 0-27) and BDI (21 questions, score 0-63)
# MAGIC
# MAGIC **This notebook checks:**
# MAGIC 1. How many sessions, users, and stimuli we have
# MAGIC 2. How depression scores are distributed across users
# MAGIC 3. How each eye-tracking metric is distributed across scenes
# MAGIC 4. Data quality: how many scenes have missing gaze data, excessive blinking, or abnormal durations
# MAGIC 5. Whether data quality is related to depression severity
# MAGIC 6. Which metrics are correlated with each other

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dataset overview

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1 Tables count

# COMMAND ----------

from config import TABLE_STIMULI, TABLE_FORMS, TABLE_SESSIONS, TABLE_SCENE_METRICS

scene_metrics = spark.table(TABLE_SCENE_METRICS)
forms = spark.table(TABLE_FORMS)
sessions = spark.table(TABLE_SESSIONS)
stimuli = spark.table(TABLE_STIMULI)

print(f"{TABLE_SCENE_METRICS}: {scene_metrics.count()} rows")
print(f"{TABLE_FORMS}: {forms.count()} rows")
print(f"{TABLE_SESSIONS}: {sessions.count()} rows") 
print(f"{TABLE_STIMULI}: {stimuli.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2 Session number by folder

# COMMAND ----------

print("Sessions per folder in scene_metrics:")
display(
    scene_metrics
    .select("session_id", "folder")
    .distinct()
    .groupBy("folder")
    .count()
    .orderBy("folder")
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.3 Sessions with form data

# COMMAND ----------

sessions_with_forms = (
    scene_metrics.select("session_id").distinct()
    .join(forms.select("session_id"), on="session_id", how="inner")
    .count()
)
sessions_total = scene_metrics.select("session_id").distinct().count()
print(f"Sessions with forms data: {sessions_with_forms} / {sessions_total}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. PHQ-9 and BDI score distributions

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Distribution by value

# COMMAND ----------

df_forms = forms.select("phq9_score", "bdi_score", "phq9_severity", "bdi_severity").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(df_forms["phq9_score"].dropna(), bins=28, color="steelblue", edgecolor="white")
axes[0].set_xlabel("PHQ-9 Score")
axes[0].set_ylabel("Count")
axes[0].set_title(f"PHQ-9 distribution (n={df_forms['phq9_score'].notna().sum()})")
axes[0].axvline(x=10, color="red", linestyle="--", alpha=0.7, label="Depression threshold")
axes[0].legend()

axes[1].hist(df_forms["bdi_score"].dropna(), bins=40, color="coral", edgecolor="white")
axes[1].set_xlabel("BDI Score")
axes[1].set_ylabel("Count")
axes[1].set_title(f"BDI distribution (n={df_forms['bdi_score'].notna().sum()})")
axes[1].axvline(x=11, color="red", linestyle="--", alpha=0.7, label="Depression threshold")
axes[1].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Distribution by severnity

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

phq_severnity = ["minimal", "mild", "moderate", "moderately_severe", "severe"]
phq_counts = df_forms["phq9_severity"].value_counts().reindex(phq_severnity).fillna(0)
axes[0].bar(phq_counts.index, phq_counts.values, color="steelblue", edgecolor="white")
axes[0].set_title("PHQ-9 severity groups")
axes[0].set_ylabel("Count")
for i, v in enumerate(phq_counts.values):
    axes[0].text(i, v + 5, str(int(v)), ha="center", fontsize=10)

bdi_severnity = ["normal", "mild", "borderline", "moderate", "severe", "extreme"]
bdi_counts = df_forms["bdi_severity"].value_counts().reindex(bdi_severnity).fillna(0)
axes[1].bar(bdi_counts.index, bdi_counts.values, color="coral", edgecolor="white")
axes[1].set_title("BDI severity groups")
axes[1].set_ylabel("Count")
for i, v in enumerate(bdi_counts.values):
    axes[1].text(i, v + 5, str(int(v)), ha="center", fontsize=10)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Scene-level metric distributions

# COMMAND ----------

df_scenes = scene_metrics.filter(F.col("scene_type") == "stimulus").toPandas()

metrics = {
    "fixation_count": "count",
    "mean_fixation_duration_ms": "ms",
    "total_fixation_duration_ms": "ms",
    "fixation_rate_per_sec": "fix/sec",
    "fixation_bias": "ratio (-1 to 1)",
    "first_fixation_duration_ms": "ms",
    "scanpath_length": "px",
    "saccade_count": "count",
    "saccade_rate_per_sec": "sacc/sec",
    "mean_saccade_amplitude": "px",
    "blink_count": "count",
    "blink_rate_per_min": "blinks/min",
    "transition_matrix_density": "ratio (0 to 1)",
    "gaze_transition_entropy": "bits",
    "dwell_time_ms_negative": "ms",
    "dwell_time_ms_positive": "ms",
    "dwell_time_ms_neutral": "ms",
    "dwell_time_500ms_negative": "ms",
    "dwell_time_500ms_positive": "ms",
    "dwell_time_500ms_neutral": "ms",
    "fixation_proportion_negative": "proportion",
    "fixation_proportion_positive": "proportion",
    "fixation_proportion_neutral": "proportion",
    "fixation_count_negative": "count",
    "fixation_count_positive": "count",
    "fixation_count_neutral": "count",
    "revisit_count_negative": "count",
    "revisit_count_positive": "count",
    "revisit_count_neutral": "count",
    "ttff_ms_negative": "ms",
    "ttff_ms_positive": "ms",
    "ttff_ms_neutral": "ms",
}

n_cols = 4
n_rows = (len(metrics) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 5.5))
axes = axes.flatten()

for i, (col, unit) in enumerate(metrics.items()):
    data = df_scenes[col].dropna()

    low, high = data.quantile(0.01), data.quantile(0.99)
    data_trimmed = data[(data >= low) & (data <= high)]

    axes[i].hist(data_trimmed, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[i].set_title(col, fontsize=11)
    axes[i].set_ylabel("Count")
    axes[i].set_xlabel(unit)

    med = data.median()
    axes[i].axvline(x=med, color="red", linestyle="--", alpha=0.7)
    axes[i].text(med, axes[i].get_ylim()[1] * 0.9, f"median={med:.1f}", color="red", fontsize=9)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data quality overview

# COMMAND ----------

df_quality = scene_metrics.select(
    "session_id", "scene_index", "scene_type",
    "n_samples", "duration_ms", "blink_ratio", "missing_gaze_ratio",
    "fixation_count", "scene_quality_valid",
).toPandas()

session_quality = df_quality.groupby("session_id").agg(
    total_scenes=("scene_index", "count"),
    valid_scenes=("scene_quality_valid", "sum"),
    mean_blink_ratio=("blink_ratio", "mean"),
    mean_missing_ratio=("missing_gaze_ratio", "mean"),
    mean_duration_ms=("duration_ms", "mean"),
).reset_index()

session_quality["valid_ratio"] = session_quality["valid_scenes"] / session_quality["total_scenes"]

df_forms_scores = forms.select("session_id", "phq9_score", "bdi_score").toPandas()
session_quality = session_quality.merge(df_forms_scores, on="session_id", how="left")

print(f"Total sessions: {len(session_quality)}")
print(f"Sessions with >=80% valid scenes: {(session_quality['valid_ratio'] >= 0.8).sum()}")
print(f"Sessions with <80% valid scenes: {(session_quality['valid_ratio'] < 0.8).sum()}")

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

axes[0, 0].hist(session_quality["valid_ratio"], bins=30, color="teal", edgecolor="white")
axes[0, 0].axvline(x=0.8, color="red", linestyle="--", label="80% threshold")
axes[0, 0].set_xlabel("Proportion of valid scenes")
axes[0, 0].set_title("Session validity ratio")
axes[0, 0].legend()

axes[0, 1].hist(session_quality["mean_blink_ratio"], bins=30, color="orange", edgecolor="white")
axes[0, 1].set_xlabel("Mean blink ratio")
axes[0, 1].set_title("Blink ratio per session")

axes[0, 2].hist(session_quality["mean_missing_ratio"], bins=30, color="red", edgecolor="white")
axes[0, 2].set_xlabel("Mean missing gaze ratio")
axes[0, 2].set_title("Missing gaze per session")

axes[1, 0].scatter(session_quality["phq9_score"], session_quality["valid_ratio"], s=5, alpha=0.3, color="teal")
axes[1, 0].set_xlabel("PHQ-9 Score")
axes[1, 0].set_ylabel("Valid scene ratio")
axes[1, 0].set_title("Data quality vs depression severity")

session_durations = df_quality.groupby("session_id")["duration_ms"].sum() / 1000
axes[1, 1].hist(session_durations, bins=50, color="orange", edgecolor="white")
axes[1, 1].set_xlabel("seconds")
axes[1, 1].set_title("Total session duration")
axes[1, 1].axvline(x=session_durations.median(), color="red", linestyle="--", label=f"median={session_durations.median():.0f}s")
axes[1, 1].legend()

scene_durations = df_quality[df_quality["scene_type"] == "stimulus"]["duration_ms"]
axes[1, 2].hist(scene_durations, bins=50, color="steelblue", edgecolor="white")
axes[1, 2].set_xlabel("ms")
axes[1, 2].set_title("Stimulus scene duration")
axes[1, 2].axvline(x=scene_durations.median(), color="red", linestyle="--", label=f"median={scene_durations.median():.0f}ms")
axes[1, 2].legend()

plt.suptitle("Data quality overview", fontsize=14)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metric correlation heatmap

# COMMAND ----------

df_scenes = scene_metrics.filter(F.col("scene_type") == "stimulus").toPandas()

corr_cols = list(metrics.keys())

for col in corr_cols:
    df_scenes[col] = pd.to_numeric(df_scenes[col], errors="coerce")

corr_matrix = df_scenes[corr_cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(18, 16))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    square=True, linewidths=0.3, ax=ax,
    vmin=-1, vmax=1,
    annot_kws={"size": 6},
)
ax.set_title("Spearman correlation between attentional metrics", fontsize=14)
ax.tick_params(axis="y", labelsize=8)
ax.tick_params(axis="x", labelsize=8, rotation=90)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.show()
