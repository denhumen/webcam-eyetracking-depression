# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline output audit
# MAGIC
# MAGIC Health check for `anima.scene_metrics` after pipeline 04

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
from config import TABLE_SCENE_METRICS, TABLE_FORMS

VALID_PAIRS = ["negative_vs_positive", "negative_vs_neutral", "neutral_vs_positive"]

scene_metrics = spark.table(TABLE_SCENE_METRICS)
forms = spark.table(TABLE_FORMS)
stim = scene_metrics.filter(F.col("scene_type") == "stimulus")
clean_stim = stim.filter(F.col("scene_valence_pair").isin(VALID_PAIRS))

print(f"Total scene rows: {scene_metrics.count()}")
print(f"Stimulus scenes: {stim.count()}")
print(f"Clean 2-image scenes: {clean_stim.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Volume

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Overall totals

# COMMAND ----------

display(scene_metrics.agg(
    F.countDistinct("session_id").alias("n_sessions"),
    F.countDistinct("folder").alias("n_folders"),
    F.count("*").alias("n_scene_rows"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Sessions per stimulus-set folder

# COMMAND ----------

display(scene_metrics.groupBy("folder").agg(F.countDistinct("session_id").alias("n_sessions"), F.count("*").alias("n_scene_rows")).orderBy("folder"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Stimulus scenes per session

# COMMAND ----------

print("Distribution of stimulus scenes per session:")
stim.groupBy("session_id").count().describe("count").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Forms coverage

# COMMAND ----------

metric_sessions = scene_metrics.select("session_id").distinct()
form_sessions = forms.select("session_id").distinct()
without = metric_sessions.join(form_sessions, "session_id", "left_anti").count()
with_forms = metric_sessions.join(form_sessions, "session_id", "inner").count()
print(f"Sessions with metrics but NO forms: {without}")
print(f"Sessions with BOTH metrics and forms: {with_forms}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Scene-type and valence-pair composition

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Stimulus vs fixation_cross

# COMMAND ----------

display(scene_metrics.groupBy("scene_type").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Full scene_valence_pair distribution for stimulus scenes

# COMMAND ----------

display(stim.groupBy("scene_valence_pair").count().orderBy(F.col("count").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Clean vs unclean summary

# COMMAND ----------

n_stim = stim.count()
n_clean = clean_stim.count()
n_unclean = n_stim - n_clean
print(f"Stimulus scenes: {n_stim}")
print(f"Clean 2-image scenes: {n_clean} ({100*n_clean/n_stim:.1f}%)")
print(f"Unclean (other): {n_unclean} ({100*n_unclean/n_stim:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Per-folder pair-type distribution

# COMMAND ----------

display(clean_stim.groupBy("folder", "scene_valence_pair").count().orderBy("folder", "scene_valence_pair"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Counts per pair type

# COMMAND ----------

display(clean_stim.groupBy("scene_valence_pair").count().orderBy("scene_valence_pair"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Scene-level data quality

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Scene_quality_valid pass rate

# COMMAND ----------

display(stim.groupBy("scene_quality_valid").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Samples per scene

# COMMAND ----------

print("n_samples in stimulus scenes:")
stim.describe("n_samples").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Missing gaze ratio

# COMMAND ----------

print("missing_gaze_ratio:")
stim.describe("missing_gaze_ratio").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Blink ratio

# COMMAND ----------

print("blink_ratio:")
stim.describe("blink_ratio").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Scene duration

# COMMAND ----------

print("duration_ms:")
stim.describe("duration_ms").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 Clean scenes per session

# COMMAND ----------

clean_per_session = clean_stim.groupBy("session_id").count()
print("Clean scenes per session:")
clean_per_session.describe("count").show()

MIN_SCENES = 5
low = clean_per_session.filter(F.col("count") < MIN_SCENES).count()
total = clean_per_session.count()
print(f"Sessions with fewer than {MIN_SCENES} clean scenes: {low} / {total}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. NaN rates per metric

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Per scene pair type

# COMMAND ----------

for pair in VALID_PAIRS:
    sub = clean_stim.filter(F.col("scene_valence_pair") == pair)
    print()
    print(f"Pair: {pair} (n={sub.count()})")
    nan_sub = _nan_rate_table(sub, f"pair={pair}")
    if nan_sub is not None:
        print(nan_sub[nan_sub["nan_rate"] > 0.01].head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metric value distributions

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Summary stats for key metrics

# COMMAND ----------

KEY_METRICS = ["fixation_count", "mean_fixation_duration_ms", "total_fixation_duration_ms", "fixation_rate_per_sec", "scanpath_length", "saccade_count","saccade_rate_per_sec", "mean_saccade_amplitude", "blink_count", "blink_rate_per_min", "gaze_transition_entropy", "transition_matrix_density", "bias_neg_vs_pos", "bias_neg_vs_neu", "bias_pos_vs_neu",
]
present = [c for c in KEY_METRICS if c in clean_stim.columns]
display(clean_stim.select(*present).describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Bias metric bounds check

# COMMAND ----------

for col in ["bias_neg_vs_pos", "bias_neg_vs_neu", "bias_pos_vs_neu"]:
    b = clean_stim.filter(F.col(col).isNotNull()).agg(
        F.min(col).alias("min"),
        F.max(col).alias("max"),
        F.count("*").alias("n")
    ).first()
    ok = (b["min"] is None) or (-1.0 <= b["min"] and b["max"] <= 1.0)
    print(f"{'OK' if ok else 'FAIL':4}  {col}: min={b['min']}, max={b['max']}, n={b['n']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Non-negativity of count columns

# COMMAND ----------

count_cols = [
    "fixation_count", "saccade_count", "blink_count",
    "fixation_count_negative", "fixation_count_positive", "fixation_count_neutral",
    "revisit_count_negative", "revisit_count_positive", "revisit_count_neutral",
]
for c in [col for col in count_cols if col in clean_stim.columns]:
    neg = clean_stim.filter(F.col(c) < 0).count()
    print(f"{'FAIL' if neg else 'OK':4}  {c}: {neg} negative values")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Outliers

# COMMAND ----------

display(stim.filter(F.col("fixation_count") > 100).select("session_id", "scene_index", "fixation_count", "duration_ms", "n_samples").orderBy(F.col("fixation_count").desc()).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 Bias histograms

# COMMAND ----------

import matplotlib.pyplot as plt

pdf_bias = clean_stim.select("bias_neg_vs_pos", "bias_neg_vs_neu", "bias_pos_vs_neu").toPandas()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, pdf_bias.columns):
    vals = pdf_bias[col].dropna()
    ax.hist(vals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title(f"{col} (n={len(vals)})")
    ax.set_xlabel("bias value")
    ax.set_ylabel("scene count")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Participant-level diagnostics

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Sessions with BOTH clean scenes AND forms

# COMMAND ----------

sessions_clean = clean_stim.select("session_id").distinct()
retained = sessions_clean.join(forms, "session_id", "inner")
print(f"Sessions retained (forms + ≥1 clean stim scene): {retained.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 PHQ-9 and BDI distributions

# COMMAND ----------

display(retained.select("phq9_score", "bdi_score").describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 PHQ-9 severity

# COMMAND ----------

if "phq9_severity" in retained.columns:
    display(retained.groupBy("phq9_severity").count().orderBy("phq9_severity"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4 BDI severity

# COMMAND ----------

if "bdi_severity" in retained.columns:
    display(retained.groupBy("bdi_severity").count().orderBy("bdi_severity"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.5 Per-user session counts

# COMMAND ----------

if "uid" in retained.columns:
    per_user = retained.groupBy("uid").count()
    per_user.describe("count").show()
    n_multi = per_user.filter(F.col("count") > 1).count()
    n_total = per_user.count()
    print(f"Users with >1 session: {n_multi} / {n_total} ({100*n_multi/n_total:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Pair-stratified readiness (TODO)

# COMMAND ----------

# Example pattern to fill in later:
#
# session_features = spark.table("anima.session_features")
# for col in [c for c in session_features.columns if "__" in c]:
#     n = session_features.filter(F.col(col).isNotNull()).count()
#     print(f"{col}: {n} sessions")
