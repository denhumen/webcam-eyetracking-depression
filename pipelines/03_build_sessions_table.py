# Databricks notebook source
# MAGIC %md
# MAGIC # Build sessions table
# MAGIC
# MAGIC Scans all session CSV files in folders to create the sessions dimension table. One row per session with metadata: session_id, uid, folder, timestamps.

# COMMAND ----------

import os
from collections import Counter
from pyspark.sql import functions as F
from config import VOLUME_BASE, STIMULUS_SETS, TABLE_SESSIONS, TABLE_FORMS

# COMMAND ----------

sessions = []
for set_name in STIMULUS_SETS:
    folder_path = f"{VOLUME_BASE}/set/{set_name}"
    try:
        for fname in os.listdir(folder_path):
            if fname.endswith(".csv"):
                sessions.append({
                    "session_id": fname.replace(".csv", ""),
                    "test_version": set_name,
                })
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")

print(f"Found {len(sessions)} session CSV files")
for v, c in sorted(Counter(s["test_version"] for s in sessions).items()):
    print(f"  {v}: {c}")

# COMMAND ----------

pdf_sessions = spark.createDataFrame(sessions)

forms = spark.table(TABLE_FORMS).select("session_id", "uid", "createdAt")
df_sessions = pdf_sessions.join(forms, on="session_id", how="left")

# COMMAND ----------

df_sessions.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_SESSIONS)

matched = df_sessions.filter(F.col("uid").isNotNull()).count()
unmatched = df_sessions.filter(F.col("uid").isNull()).count()
print(f"Table '{TABLE_SESSIONS}' created with {df_sessions.count()} rows")
print(f"  With forms data: {matched}")
print(f"  Without forms data: {unmatched}")

# COMMAND ----------

display(spark.table(TABLE_SESSIONS).limit(10))
