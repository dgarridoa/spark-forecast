from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
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
