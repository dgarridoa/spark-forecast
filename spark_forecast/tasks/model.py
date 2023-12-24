import argparse
import sys
from datetime import date, datetime, timedelta
from typing import Optional, Type

import mlflow
from darts.models import NaiveMean, NaiveMovingAverage, Prophet
from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.croston import Croston
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.random_forest import RandomForest
from darts.models.forecasting.xgboost import XGBModel
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from spark_forecast.common import Task
from spark_forecast.model import DistributedModel, ModelProtocol
from spark_forecast.schema import ForecastSchema
from spark_forecast.utils import (
    read_delta_table,
    set_mlflow_experiment,
    write_delta_table,
)


class ModelTask(Task):
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

    def _write_delta_table(
        self, df: DataFrame, output_dict: dict[str, str]
    ) -> None:
        write_delta_table(
            self.spark,
            df,
            ForecastSchema,
            output_dict["database"],
            output_dict["table"],
            ["model"],
        )

    def fit_predict(
        self, model_cls: Type[ModelProtocol], df_train: DataFrame, steps: int
    ) -> DataFrame:
        group_columns: list[str] = self.conf["group_columns"]
        time_column: str = self.conf["time_column"]
        target_column: str = self.conf["target_column"]
        model_params: dict = self.conf.get("model_params", {})
        freq: str = self.conf["freq"]

        distributed_model = DistributedModel(
            group_columns=group_columns,
            time_column=time_column,
            target_column=target_column,
            model_cls=model_cls,
            model_params=model_params,
            freq=freq,
        )
        df_predict = distributed_model.fit_predict(
            df_train=df_train,
            forecast_schema=ForecastSchema,
            steps=steps,
        )
        return df_predict

    def launch(self, model_cls: Type[ModelProtocol]):
        run_name = f"{self.__class__.__name__}[{model_cls.__name__}]"
        self.logger.info(f"Launching {run_name}")

        set_mlflow_experiment()
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(self.conf)

        df = self._read_delta_table()
        df_train = df.filter(df["split"] == "train")

        df_forecast_on_test = self.fit_predict(
            model_cls, df_train, self.conf["test_size"]
        )
        df_forecast = self.fit_predict(model_cls, df, self.conf["steps"])

        self._write_delta_table(
            df_forecast_on_test, self.conf["output"]["forecast_on_test"]
        )
        self._write_delta_table(
            df_forecast, self.conf["output"]["all_models_forecast"]
        )


MODELS: list[Type[ModelProtocol]] = [
    ExponentialSmoothing,
    AutoARIMA,
    Prophet,
    XGBModel,
    RandomForest,
    Croston,
    NaiveMean,
    NaiveMovingAverage,
]


def get_model_class(model_name: str) -> Type[ModelProtocol]:
    for model_cls in MODELS:
        if model_cls.__name__ == model_name:
            return model_cls
    raise ValueError(f"Model {model_name} not found")


def entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="ExponentialSmoothing")
    model_name = parser.parse_known_args(sys.argv[1:])[0].model_name
    model = get_model_class(model_name)
    task = ModelTask()
    task.launch(model)


if __name__ == "__main__":
    entrypoint()
