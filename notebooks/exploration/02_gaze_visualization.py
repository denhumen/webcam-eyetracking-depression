# Databricks notebook source
# MAGIC %md
# MAGIC # Gaze visualization
# MAGIC
# MAGIC This notebook visualizes how a single user looked at images during their eye-tracking session. It loads the raw gaze data for a selected session and plots what happened on each stimulus scene.
# MAGIC
# MAGIC **What it shows:**
# MAGIC - Scanpath: where the eyes fixated during each scene, shown as colored dots on top of the actual stimulus images. Dot size reflects how long each fixation lasted, and dot color shows which image the fixation belongs to
# MAGIC - Heatmap: gaze density across the scene, showing which areas of the screen received the most attention
# MAGIC - Fixation summary: distribution of fixation durations across the whole session, and how fixations split between image valences
# MAGIC
# MAGIC **How to use:**
# MAGIC Set `SESSION_ID` in the cell below to any session ID from the dataset. Leave it empty if you want to pick a random session. Use `START_SCENE` and `N_SCENES` to control which scenes are plotted.
# MAGIC
# MAGIC **Note on image positions:** The AOI boxes and stimulus images on the plots are placed at approximate screen coordinates estimated from the gaze data. The actual image positions depend on each user's screen size and browser layout and these metrics are not recorded in the data. Therefore, some fixation dots may appear outside of the image boxes even though the platform labeled them as ones that belong to that image.

# COMMAND ----------

import importlib
import src.scene_visualization
importlib.reload(src.scene_visualization)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

from config import VOLUME_BASE, STIMULUS_SETS
from src.data_loading import load_session, load_stimulus_config
from src.preprocessing import (
    get_scene_indices, get_scene_data, extract_fixations,
    compute_session_quality, get_scene_images,
)
from src.scene_metrics import derive_valence
from src.scene_visualization import plot_scene_exploration, VALENCE_COLORS

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Select and load session

# COMMAND ----------

SESSION_ID = "6YBBzBibDY2wbHWLaFNS"

# COMMAND ----------

df_forms = spark.table("anima.forms").toPandas()
df_sessions = spark.table("anima.sessions").toPandas()

if SESSION_ID:
    row = df_sessions[df_sessions["session_id"] == SESSION_ID].iloc[0]
else:
    valid_sids = set(df_forms["session_id"]) & set(df_sessions["session_id"])
    row = df_sessions[df_sessions["session_id"] == np.random.choice(list(valid_sids))].iloc[0]

session_id = row["session_id"]
test_version = row.get("test_version", "depression")

df = load_session(f"{VOLUME_BASE}/set/{test_version}/{session_id}.csv")
stimulus_config = load_stimulus_config(f"{VOLUME_BASE}/set/{STIMULUS_SETS[test_version]}")

quality = compute_session_quality(df)
df_form_row = df_forms[df_forms["session_id"] == session_id]
phq = df_form_row.iloc[0]["phq9_score"] if len(df_form_row) > 0 else "N/A"
bdi = df_form_row.iloc[0]["bdi_score"] if len(df_form_row) > 0 else "N/A"

print(f"Session: {session_id}")
print(f"Folder: {test_version}")
print(f"PHQ-9: {phq}, BDI: {bdi}")
print()
print(f"Duration: {quality['total_duration_ms']/1000:.1f}s, {quality['n_scenes']} scenes, {len(df)} samples")
print(f"Missing gaze: {quality['missing_gaze_ratio']:.1%}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Visualize stimulus scenes (scanpath + heatmap for each scene)

# COMMAND ----------

START_SCENE = 0
N_SCENES = 20

stimulus_scenes = [i for i in get_scene_indices(df) if i % 2 == 1][START_SCENE:N_SCENES]

for scene_idx in stimulus_scenes:
    scene_df = get_scene_data(df, scene_idx)
    fixations = extract_fixations(scene_df)
    plot_scene_exploration(scene_df, scene_idx, fixations, stimulus_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Fixation count and duration distributions for selected session

# COMMAND ----------

all_fixations = []
for scene_idx in get_scene_indices(df):
    if scene_idx % 2 == 1:
        scene_df = get_scene_data(df, scene_idx)
        fixations = extract_fixations(scene_df)
        if len(fixations) > 0:
            fixations["scene_index"] = scene_idx
            all_fixations.append(fixations)

if all_fixations:
    df_all_fixations = pd.concat(all_fixations, ignore_index=True)

    valence_map = {img_id: derive_valence(meta.get("category", ""), meta.get("labels", []))
                   for img_id, meta in stimulus_config.items()}
    df_all_fixations["valence"] = df_all_fixations["image"].map(valence_map).fillna("off_image")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    durations = df_all_fixations["duration_ms"].dropna()
    axes[0].hist(durations, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(x=durations.median(), color="red", linestyle="--",
                    label=f"median={durations.median():.0f}ms")
    axes[0].set_xlabel("Fixation duration (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Duration distribution (n={len(durations)})")
    axes[0].legend()

    val_counts = df_all_fixations["valence"].value_counts()
    colors = [VALENCE_COLORS.get(v, "gray") for v in val_counts.index]
    axes[1].bar(val_counts.index, val_counts.values, color=colors, edgecolor="white")
    axes[1].set_xlabel("Valence")
    axes[1].set_ylabel("Fixation count")
    axes[1].set_title("Fixation count by valence")

    val_dur = df_all_fixations.groupby("valence")["duration_ms"].mean()
    colors2 = [VALENCE_COLORS.get(v, "gray") for v in val_dur.index]
    axes[2].bar(val_dur.index, val_dur.values, color=colors2, edgecolor="white")
    axes[2].set_xlabel("Valence")
    axes[2].set_ylabel("Mean fixation duration (ms)")
    axes[2].set_title("Mean fixation duration by valence")

    plt.tight_layout()
    plt.show()
