# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Sample notebook

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Aux steps for auto reloading of dependent files

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Example usage of existing code

# COMMAND ----------

from pathlib import Path

import yaml
from pyspark.sql import SparkSession

from spark_forecast.tasks.split import SplitTask

# COMMAND ----------

project_root = Path(".").absolute().parent

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.sql("USE CATALOG `demand-forecast`")

# COMMAND ----------

conf_file = f"{project_root}/conf/tasks/split_config.yml"
conf = yaml.safe_load(Path(project_root, conf_file).read_text())["env"]["dev"]

# COMMAND ----------

task = SplitTask(spark, conf)
task.launch()
