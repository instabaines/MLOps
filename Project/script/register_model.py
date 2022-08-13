import argparse
import os
import pickle

import mlflow
from hyperopt import hp, space_eval
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


EXPERIMENT_NAME = "random_forest_pipeline_experiment"

mlflow.set_tracking_uri("sqlite:///backend.db")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def run(log_top=1):
    client = MlflowClient()

    # select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.error ASC"]
    )[0]

    # register the best model
    run_id=best_run.info.run_id
    model_uri= f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri,name="model")
    print(run_id)


if __name__ == '__main__':
    run()
