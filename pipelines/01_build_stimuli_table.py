# Databricks notebook source
# MAGIC %md
# MAGIC # Build stimuli table
# MAGIC
# MAGIC Parses all JSON stimulus configs and creates the stimuli table with the TABLE_STIMULI name from the config

# COMMAND ----------

from config import VOLUME_BASE, STIMULUS_SETS, TABLE_STIMULI, NEGATIVE_KEYWORDS, POSITIVE_KEYWORDS
import json
import pandas as pd
from src.data_loading import load_stimulus_config

# COMMAND ----------

def classify_emotion(category, labels):
    """Classify stimulus as negative, positive, neutral, food, or other"""
    labels_lower = [l.lower() for l in labels]
    
    if category.strip() == "neutral":
        return "neutral"
    if any(kw in labels_lower for kw in NEGATIVE_KEYWORDS):
        return "negative"
    if any(kw in labels_lower for kw in POSITIVE_KEYWORDS):
        return "positive"
    if "food" in labels_lower:
        return "food"
    return "other"

# COMMAND ----------

rows = []
for set_name, json_filename in STIMULUS_SETS.items():
    json_path = f"{VOLUME_BASE}/set/{json_filename}"
    config = load_stimulus_config(json_path)
    
    for image_id, meta in config.items():
        category = meta.get("category", "").strip()
        labels = meta.get("labels", [])
        
        rows.append({
            "image_id": image_id,
            "folder": set_name,
            "category": category,
            "valence": classify_emotion(category, labels),
            "labels": json.dumps(labels),
            "url": meta.get("url", ""),
        })

pdf_stimuli = pd.DataFrame(rows)

print(f"Total stimuli: {len(pdf_stimuli)}")
print(f"\nBy folder:\n{pdf_stimuli.groupby('folder').size()}")
print(f"\nBy valence:\n{pdf_stimuli.groupby('valence').size()}")

# COMMAND ----------

sdf = spark.createDataFrame(pdf_stimuli)
sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_STIMULI)
print(f"Table '{TABLE_STIMULI}' created with {len(pdf_stimuli)} rows")

# COMMAND ----------

display(spark.table(TABLE_STIMULI).limit(20))
