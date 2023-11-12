from datetime import date, datetime, timedelta
from typing import Optional

import mlflow
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType

from spark_forecast.common import Task
from spark_forecast.evaluation import Evaluation
from spark_forecast.schema import ForecastSchema, MetricsSchema
from spark_forecast.utils import (
    read_delta_table,
    set_mlflow_experiment,
    write_delta_table,
)


class EvaluationTask(Task):
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

    def _read_delta_table(self, input_dict: dict[str, str]) -> DataFrame:
        df = read_delta_table(
            self.spark, input_dict["database"], input_dict["table"]
        )
        return df

    def _write_delta_table(
        self, df: DataFrame, schema: StructType, output_dict: dict[str, str]
    ) -> None:
        write_delta_table(
            self.spark,
            df,
            schema,
            output_dict["database"],
            output_dict["table"],
        )

    def launch(self):
        self.logger.info(f"Launching {self.__class__.__name__}")

        evaluation = Evaluation(
            group_columns=self.conf["group_columns"],
            time_column=self.conf["time_column"],
            target_column=self.conf["target_column"],
            metrics=self.conf["metrics"],
            model_selection_metric=self.conf["model_selection_metric"],
            freq=self.conf["freq"],
        )

        set_mlflow_experiment()
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.conf)

            df_test = self._read_delta_table(
                self.conf["input"]["split"]
            ).filter(F.col("split") == "test")
            df_forecast_on_test = self._read_delta_table(
                self.conf["input"]["forecast_on_test"]
            )
            df_metrics = evaluation.get_metrics(
                df_test, df_forecast_on_test, MetricsSchema
            )
            df_best_models = evaluation.get_best_models(df_metrics).cache()

            mlflow.log_metric(
                self.conf["model_selection_metric"],
                df_best_models.agg(F.mean("value")).collect()[0][0],
            )

        df_all_models_forecast = self._read_delta_table(
            self.conf["input"]["all_models_forecast"]
        )
        df_forecast = evaluation.get_forecast(
            df_all_models_forecast, df_best_models
        )

        self._write_delta_table(
            df_metrics, MetricsSchema, self.conf["output"]["metrics"]
        )
        self._write_delta_table(
            df_best_models, MetricsSchema, self.conf["output"]["best_models"]
        )
        self._write_delta_table(
            df_forecast, ForecastSchema, self.conf["output"]["forecast"]
        )


def entrypoint():
    task = EvaluationTask()
    task.launch()


if __name__ == "__main__":
    entrypoint()
