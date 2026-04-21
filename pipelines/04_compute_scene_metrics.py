# Databricks notebook source
# MAGIC %md
# MAGIC # Compute scene metrics
# MAGIC
# MAGIC Main computation pipeline. For each session pipeline does the next steps:
# MAGIC 1. Reads the raw CSV
# MAGIC 2. Iterates over scenes
# MAGIC 3. Extracts fixations and blinks
# MAGIC 4. Computes all static attention metrics (using the stimulus schedule as
# MAGIC    the ground truth for which images were shown per scene)
# MAGIC 5. Writes to the TABLE_SCENE_METRICS table (from config.py)

# COMMAND ----------

from config import VOLUME_BASE, STIMULUS_SETS, TABLE_SCENE_METRICS, FOLDERS_TO_PROCESS
import os
import json
import pandas as pd
import numpy as np
from src.data_loading import (
    load_stimulus_config, load_stimulus_schedule,
    load_session, session_id_from_path,
)
from src.preprocessing import get_scene_indices, get_scene_data, compute_session_quality
from src.scene_processor import compute_scene_metrics

# COMMAND ----------

def process_session(csv_path, folder, stimulus_config, stimulus_schedule):
    """
    Process one session CSV, return list of scene metric dicts or None if invalid
    """
    session_id = session_id_from_path(csv_path)
    
    try:
        df = load_session(csv_path)
    except Exception as e:
        print(f"Failed to load {session_id}: {e}")
        return None
    
    quality = compute_session_quality(df)
    if not quality["is_valid"]:
        return None
    
    rows = []
    for scene_idx in get_scene_indices(df):
        scene_df = get_scene_data(df, scene_idx)
        scene_schedule = stimulus_schedule.get(scene_idx)
        metrics = compute_scene_metrics(
            scene_df, scene_idx, stimulus_config, scene_schedule
        )
        
        metrics["session_id"] = session_id
        metrics["folder"] = folder
        
        for key in list(metrics.keys()):
            if isinstance(metrics[key], list):
                metrics[key] = json.dumps(metrics[key])
        
        rows.append(metrics)
    
    return rows

# COMMAND ----------

all_rows = []
processed = 0
failed = 0

for version in FOLDERS_TO_PROCESS:
    print(f"Processing: {version}")
    
    json_path = f"{VOLUME_BASE}/set/{STIMULUS_SETS[version]}"
    try:
        stim_config = load_stimulus_config(json_path)
        stim_schedule = load_stimulus_schedule(json_path)
    except Exception as e:
        print(f"Could not load config/schedule: {e}")
        continue
    
    folder_path = f"{VOLUME_BASE}/set/{version}"
    try:
        csv_files = sorted([
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.endswith(".csv")
        ])
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        continue
    
    print(f"{len(csv_files)} sessions")
    
    for i, csv_path in enumerate(csv_files):
        try:
            rows = process_session(csv_path, version, stim_config, stim_schedule)
            if rows:
                all_rows.extend(rows)
                processed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"Error: {session_id_from_path(csv_path)}: {e}")
        
        if (i + 1) % 100 == 0:
            print(f"... {i+1}/{len(csv_files)}")
    
    print(f"Done: {version}")

print(f"Processed: {processed}, Failed: {failed}, Total rows: {len(all_rows)}")

# COMMAND ----------

if all_rows:
    pdf = pd.DataFrame(all_rows)
    
    for col in pdf.columns:
        if pdf[col].dtype == object:
            pdf[col] = pdf[col].where(pdf[col].notna(), None).astype("string")
    
    sdf = spark.createDataFrame(pdf)
    sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_SCENE_METRICS)
    
    print(f"Table '{TABLE_SCENE_METRICS}' created with {len(all_rows)} rows")
else:
    print("No data to write")

# COMMAND ----------

print("Sample rows:")
display(spark.table(TABLE_SCENE_METRICS).limit(5))

print("By folder:")
display(
    spark.table(TABLE_SCENE_METRICS)
    .groupBy("folder").count()
    .orderBy("folder")
)

print("By scene type:")
display(
    spark.table(TABLE_SCENE_METRICS)
    .groupBy("scene_type").count()
)
