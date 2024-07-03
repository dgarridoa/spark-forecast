import argparse
import pathlib
import sys
from datetime import date, datetime
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import AfterValidator
from zoneinfo import ZoneInfo

from spark_forecast.evaluation import metric_validator
from spark_forecast.model import model_cls_validator


class CommonParams(BaseModel):
    env: str = "dev"
    tz: str = "America/Santiago"
    execution_date: Annotated[date, Field(default=None)]
    database: str = "dev"

    def model_post_init(self, __context: Any) -> None:
        if self.execution_date is None:
            self.execution_date = datetime.now(tz=ZoneInfo(self.tz)).date()


class IngestionParams(CommonParams):
    path: str = "Chile/SM/PQ0/CHI_SUP/FACT_DAILY_INVENTORY"
    sep: str = ","
    stores: list[int] | None = None


class SplitParams(CommonParams):
    group_columns: list[str] = ["store", "item"]
    time_column: str = "date"
    target_column: str = "sales"
    time_delta: int = Field(2 * 52 * 7, gt=0)
    test_size: int = Field(21, gt=0)
    freq: str = "1D"


class ModelParams(CommonParams):
    group_columns: list[str] = ["store", "item"]
    time_column: str = "date"
    target_column: str = "sales"
    model_cls: str
    model_params: dict = {}
    test_size: int = Field(21, gt=0)
    steps: int = Field(21, gt=0)
    freq: str = "1D"

    model_config = ConfigDict(protected_namespaces=())

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        model_cls_validator(self.model_cls)


def metrics_validator(metrics: list[str]) -> list[str]:
    for metric in metrics:
        _ = metric_validator(metric)
    return metrics


class EvaluationParams(CommonParams):
    group_columns: list[str] = ["store", "item"]
    time_column: str = "date"
    target_column: str = "sales"
    metrics: Annotated[list[str], AfterValidator(metrics_validator)] = [
        "rmse",
        "mae",
        "mape",
    ]
    model_selection_metric: Annotated[
        str, AfterValidator(metric_validator)
    ] = "mape"
    freq: str = "1D"

    model_config = ConfigDict(protected_namespaces=())


class Params(BaseModel):
    create_database: CommonParams
    ingestion: IngestionParams
    split: SplitParams
    models: dict[str, ModelParams]
    evaluation: EvaluationParams


def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-file")
    conf_file = parser.parse_known_args(sys.argv[1:])[0].conf_file
    config = yaml.safe_load(pathlib.Path(conf_file).read_text())
    return config
