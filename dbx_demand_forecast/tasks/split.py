from datetime import date, datetime, timedelta
from typing import Optional

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from dbx_demand_forecast.common import Task
from dbx_demand_forecast.schema import SplitSchema
from dbx_demand_forecast.split import Split
from dbx_demand_forecast.utils import read_delta_table, write_delta_table


class SplitTask(Task):
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        init_conf: Optional[dict] = None,
    ) -> None:
        super().__init__(spark, init_conf)
        if "execution_date" not in self.conf:
            execution_date = date.today() - timedelta(
                days=date.today().weekday()
            )
        else:
            execution_date = datetime.strptime(
                self.conf["execution_date"], "%Y-%m-%d"
            ).date()
        self.conf["execution_date"] = execution_date

    def _read_delta_table(self) -> DataFrame:
        df = read_delta_table(
            self.spark,
            self.conf["input"]["database"],
            self.conf["input"]["table"],
        )
        return df

    def _write_delta_table(self, df: DataFrame) -> None:
        write_delta_table(
            self.spark,
            df,
            SplitSchema,
            self.conf["output"]["database"],
            self.conf["output"]["table"],
        )

    def transform(self, df: DataFrame) -> DataFrame:
        split = Split(
            group_columns=self.conf["group_columns"],
            time_column=self.conf["time_column"],
            target_column=self.conf["target_column"],
            test_size=self.conf["test_size"],
            execution_date=self.conf["execution_date"],
            time_delta=self.conf["time_delta"],
            freq=self.conf["freq"],
        )
        df_split = split.transform(df, SplitSchema)
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
