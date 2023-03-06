from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

SalesSchema = StructType(
    [
        StructField("date", DateType(), False),
        StructField("store", IntegerType(), False),
        StructField("item", IntegerType(), False),
        StructField("sales", DoubleType(), False),
    ]
)

InputSchema = StructType(
    [
        StructField("date", DateType(), False),
        StructField("store", IntegerType(), False),
        StructField("item", IntegerType(), False),
        StructField("sales", DoubleType(), False),
    ]
)

SplitSchema = StructType(
    [
        StructField("date", DateType(), False),
        StructField("store", IntegerType(), False),
        StructField("item", IntegerType(), False),
        StructField("sales", DoubleType(), False),
        StructField("split", StringType(), False),
    ]
)

ForecastSchema = StructType(
    [
        StructField("model", StringType(), False),
        StructField("store", IntegerType(), False),
        StructField("item", IntegerType(), False),
        StructField("date", DateType(), False),
        StructField("sales", DoubleType(), False),
    ]
)

MetricsSchema = StructType(
    [
        StructField("model", StringType(), False),
        StructField("store", IntegerType(), False),
        StructField("item", IntegerType(), False),
        StructField("metric", StringType(), False),
        StructField("value", DoubleType(), False),
    ]
)
