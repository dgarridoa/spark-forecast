import mlflow

from dbx_demand_forecast.common import Task


class CreateDataBaseTask(Task):
    def launch(self):
        self.logger.info(f"Launching {self.__class__.__name__}")

        mlflow.set_experiment(self.conf["experiment"])
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.conf)

        self.spark.sql(
            f"CREATE DATABASE IF NOT EXISTS {self.conf['database']}"
        )


def entrypoint():
    task = CreateDataBaseTask()
    task.launch()


if __name__ == "__main__":
    entrypoint()
