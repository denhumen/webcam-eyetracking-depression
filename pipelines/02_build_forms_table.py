# Databricks notebook source
# MAGIC %md
# MAGIC # Build forms table
# MAGIC
# MAGIC Loads forms.csv and creates the forms table with PHQ-9 and BDI severity classifications. Uses the TABLE_FORMS name from the config

# COMMAND ----------

from config import FORMS_PATH, TABLE_FORMS
from pyspark.sql import functions as F

# COMMAND ----------

print(f"Reading forms from: {FORMS_PATH}")

df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(FORMS_PATH)
)

if "_c0" in df_raw.columns:
    df_raw = df_raw.drop("_c0")
if "Unnamed: 0" in df_raw.columns:
    df_raw = df_raw.drop("Unnamed: 0")

print(f"Loaded {df_raw.count()} rows")

# COMMAND ----------

df = df_raw.withColumnRenamed("sid", "session_id")

df = df.withColumn(
    "phq9_severity",
    F.when(F.col("`phq-9_score`") <= 4,  "minimal")
     .when(F.col("`phq-9_score`") <= 9,  "mild")
     .when(F.col("`phq-9_score`") <= 14, "moderate")
     .when(F.col("`phq-9_score`") <= 19, "moderately_severe")
     .otherwise("severe")
)

df = df.withColumn(
    "bdi_severity",
    F.when(F.col("bdi_score") <= 13, "minimal")
     .when(F.col("bdi_score") <= 19, "mild")
     .when(F.col("bdi_score") <= 28, "moderate")
     .otherwise("severe")
)

# COMMAND ----------

for i in range(1, 10):
    df = df.withColumnRenamed(f"phq-9_{i}", f"phq9_q{i}")
df = df.withColumnRenamed("phq-9_score", "phq9_score")

for i in range(1, 22):
    old_name = f"bdi_{i}"
    if old_name in df.columns:
        df = df.withColumnRenamed(old_name, f"bdi_q{i}")

# COMMAND ----------

df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_FORMS)

total_sessions = df.count()
total_users = df.select("uid").distinct().count()
print(f"Table '{TABLE_FORMS}' created")
print(f"  Sessions: {total_sessions}")
print(f"  Users: {total_users}")

print("\nPHQ-9 severity distribution:")
display(df.groupBy("phq9_severity").count().orderBy("phq9_severity"))

print("\nBDI severity distribution:")
display(df.groupBy("bdi_severity").count().orderBy("bdi_severity"))
