import mlflow
import pandas as pd
from darts.models import NaiveDrift, NaiveEnsembleModel, NaiveSeasonal
from darts.timeseries import TimeSeries
from delta.tables import DeltaTable
from pyspark.sql.dataframe import DataFrame

from dbx_demand_forecast.common import Task
from dbx_demand_forecast.schema import ForecastSchema
from dbx_demand_forecast.utils import read_delta_table


class NaiveModel:
    def __init__(
        self,
        group_columns: list[str],
        time_column: str,
        target_column: str,
        model_params: dict,
    ):
        self.group_columns = group_columns
        self.time_column = time_column
        self.target_column = target_column
        self.model = NaiveEnsembleModel(
            [NaiveDrift(), NaiveSeasonal(**model_params)]
        )

    def set_group_columns_values(self, df: pd.DataFrame) -> None:
        self.group_columns_values = df.loc[0, self.group_columns].to_dict()

    def fit(self, df_train: pd.DataFrame) -> None:
        self.set_group_columns_values(df_train)

        df_train[self.time_column] = pd.to_datetime(df_train[self.time_column])
        df_train = df_train.set_index(self.time_column)
        y_train = TimeSeries.from_series(
            df_train[self.target_column], freq="D"
        )

        self.model.fit(y_train)

    def predict(self, steps: int) -> pd.DataFrame:
        y_predict = self.model.predict(steps).pd_series()
        df = pd.DataFrame(
            {
                "model": self.__class__.__name__,
                **self.group_columns_values,
                self.time_column: y_predict.index,
                self.target_column: y_predict.values,
            }
        )
        return df

    @staticmethod
    def fit_predict(
        df_train: pd.DataFrame,
        steps: int,
        group_columns: list[str],
        time_column: str,
        target_column: str,
        model_params: dict,
    ) -> pd.DataFrame:
        model = NaiveModel(
            group_columns, time_column, target_column, model_params
        )
        model.fit(df_train)
        df_predict = model.predict(steps)
        return df_predict


class NaiveModelTask(Task):
    def _read_delta_table(self) -> DataFrame:
        df = read_delta_table(self.spark, path=self.conf["input"]["path"])
        return df

    def _write_delta_table(
        self, df: DataFrame, output_dict: dict[str, str]
    ) -> None:
        self.spark.sql(
            f"CREATE SCHEMA IF NOT EXISTS {output_dict['database']}"
        )
        (
            DeltaTable.createIfNotExists(self.spark)
            .tableName(f"{output_dict['database']}.{output_dict['table']}")
            .addColumns(ForecastSchema)
            .partitionedBy("model")
            .location(output_dict["path"])
            .execute()
        )
        df.write.format("delta").mode("overwrite").option(
            "partitionOverwriteMode", "dynamic"
        ).save(output_dict["path"])

    def fit_predict(self, df_train: DataFrame, steps: int) -> DataFrame:
        group_columns: list[str] = self.conf["group_columns"]
        time_column: str = self.conf["time_column"]
        target_column: str = self.conf["target_column"]
        model_params: dict[str, str] = self.conf["model_params"]

        df_predict = df_train.groupBy(group_columns).applyInPandas(
            lambda pdf: NaiveModel.fit_predict(
                df_train=pdf,
                steps=steps,
                group_columns=group_columns,
                time_column=time_column,
                target_column=target_column,
                model_params=model_params,
            ),
            schema=ForecastSchema,
        )
        return df_predict

    def launch(self):
        self.logger.info(f"Launching {self.__class__.__name__}")

        mlflow.set_experiment(self.conf["experiment"])
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.conf)

        df = self._read_delta_table()
        df_train = df.filter(df["split"] == "train")

        df_forecast_on_test = self.fit_predict(
            df_train, self.conf["test_size"]
        )
        df_forecast = self.fit_predict(df, self.conf["steps"])

        self._write_delta_table(
            df_forecast_on_test, self.conf["output"]["forecast_on_test"]
        )
        self._write_delta_table(
            df_forecast, self.conf["output"]["all_models_forecast"]
        )


def entrypoint():
    task = NaiveModelTask()
    task.launch()


if __name__ == "__main__":
    entrypoint()
