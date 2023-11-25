import os

import mlflow
import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame

from spark_forecast.common import Task
from spark_forecast.schema import InputSchema, SalesSchema
from spark_forecast.utils import (
    read_csv,
    set_mlflow_experiment,
    write_delta_table,
)


class IngestionTask(Task):
    def _read_csv(self) -> DataFrame:
        file_path = os.getenv("WORKSPACE_FILE_PATH")
        if file_path:
            self.conf["input"][
                "path"
            ] = "file:{file_path}/{relative_path}".format(
                file_path=file_path, relative_path=self.conf["input"]["path"]
            )
        df = read_csv(
            self.spark,
            self.conf["input"]["path"],
            self.conf["input"]["sep"],
            SalesSchema,
        )

        if "stores" in self.conf:
            df = df.filter(F.col("store").isin(self.conf["stores"]))
        return df

    def _write_delta_table(self, df: DataFrame) -> None:
        write_delta_table(
            self.spark,
            df,
            InputSchema,
            self.conf["output"]["database"],
            self.conf["output"]["table"],
        )

    def launch(self):
        self.logger.info(f"Launching {self.__class__.__name__}")

        set_mlflow_experiment()
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.conf)

        df = self._read_csv()
        self._write_delta_table(df)


def entrypoint():
    task = IngestionTask()
    task.launch()


if __name__ == "__main__":
    entrypoint()
