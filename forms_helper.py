from pyspark.sql import functions as F

def get_forms_by_uid(spark, uid: str):
    return (
        spark.table("workspace.anima.forms")
             .filter(F.col("uid") == uid)
    )

def get_forms_by_sid(spark, sid: str):
    return (
        spark.table("workspace.anima.forms")
             .filter(F.col("sid") == sid)
    )

def get_results_for_sid(spark, sid: str):
    df = spark.table("workspace.anima.forms")

    row = (
        df.filter(F.col("sid") == sid)
          .select(
              "sid", "uid",
              "phq-9_score", "PHQ9_severity",
              "bdi_score", "BDI_severity"
          )
          .limit(1)
          .collect()
    )

    return row[0] if row else None
