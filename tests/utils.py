from pyspark.sql import DataFrame


def assert_pyspark_df_equal(
    expected_df: DataFrame, actual_df: DataFrame
) -> None:
    equal_schema = expected_df.dtypes == actual_df.dtypes
    equal_count = expected_df.count() == actual_df.count()
    equal_values = expected_df.join(actual_df, how="anti").count() == 0
    assert equal_schema and equal_count and equal_values
