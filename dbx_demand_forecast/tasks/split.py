from datetime import timedelta

import mlflow
import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

from dbx_demand_forecast.common import Task
from dbx_demand_forecast.schema import SplitSchema
from dbx_demand_forecast.utils import read_delta_table, write_delta_table


class SplitTask(Task):
    def _read_delta_table(self) -> DataFrame:
        df = read_delta_table(self.spark, path=self.conf["input"]["path"])
        return df

    def _write_delta_table(self, df: DataFrame) -> None:
        write_delta_table(
            self.spark,
            df,
            self.conf["output"]["path"],
            SplitSchema,
            self.conf["output"]["database"],
            self.conf["output"]["table"],
        )

    def transform(self, df: DataFrame) -> DataFrame:
        end_test_date = df.agg({"date": "max"}).collect()[0][0]
        start_test_date = end_test_date - timedelta(self.conf["test_size"] - 1)
        df_split = df.withColumn(
            "split",
            F.when(F.col("date") >= start_test_date, "test").otherwise(
                "train"
            ),
        )
        return df_split

    def launch(self):
        self.logger.info(f"Launching {self.__class__.__name__}")

        mlflow.set_experiment(self.conf["experiment"])
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.conf)

        df = self._read_delta_table()
        df_split = self.transform(df)
        self._write_delta_table(df_split)


def entrypoint():
    task = SplitTask()
    task.launch()


if __name__ == "__main__":
    entrypoint()
