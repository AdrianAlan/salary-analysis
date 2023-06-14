import argparse
import pickle
import pandas as pd
import numpy as np
import yaml

from coolname import generate
from sklearn.linear_model import (
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
        BayesianRidge,
        QuantileRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        mean_pinball_loss,
        make_scorer,
        median_absolute_error,
        r2_score,
        mean_absolute_percentage_error)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from utils import IsValidFile


def evaluate(y_true: np.array, y_pred: np.array, alpha: float = 0.5):
    evaluation = dict()
    evaluation['r2'] = r2_score(y_true, y_pred)
    evaluation['mse'] = mean_squared_error(y_true, y_pred)
    evaluation['mae'] = mean_absolute_error(y_true, y_pred)
    evaluation['mpl'] = mean_pinball_loss(y_true, y_pred, alpha=alpha)
    evaluation['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    evaluation['medae'] = median_absolute_error(y_true, y_pred)
    return evaluation


def run_grid_search(grid: GridSearchCV,
                    X_train: np.array,
                    y_train: np.array,
                    X_test: np.array,
                    y_test: np.array,
                    quantile: float):
    grid.fit(X_train.reshape(-1, X_train.shape[1]), y_train)
    y_pred = grid.predict(X_test)
    scores = evaluate(y_test, y_pred, alpha=quantile)
    grid.best_params_['no_features'] = X_train.shape[1]
    grid.best_params_['estimator'] = grid.best_estimator_
    return grid, scores


def main(dataset: str = None,
         experiments: list = None,
         log_mlflow: bool = False,
         mlflow_username: str = None,
         mlflow_password: str = None,
         mlflow_exp_name: str = None,
         mlflow_tracking: str = None,
         yoe: bool = True,
         yac: bool = True,
         gender: bool = True,
         location: int = 5,
         cv: int = 5,
         jobs: int = -1,
         reg: bool = True,
         quantile: float = 0.5):

    features = []

    # Setup MLflow
    if log_mlflow:
        import os
        import mlflow
        import mlflow.sklearn
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
        mlflow.set_tracking_uri(mlflow_tracking)
        mlflow.set_experiment(experiment_name=mlflow_exp_name)
        experiment = mlflow.get_experiment_by_name(mlflow_exp_name)

    # Load the dataset
    df_salary = pd.read_csv(dataset)

    # Select input features
    if yoe:
        features.append('yoe')

    if yac:
        features.append('yac')

    if gender:
        genders = pd.get_dummies(df_salary['gender'])
        df_salary = df_salary.drop('gender', axis=1)
        df_salary = df_salary.join(genders)
        features.append('Female')

    if location:
        # Append standard of living as kernel for location
        if location == 5:
            features.append('coli')

        # Map state to country except case 1: encoded state and country
        if location > 1 and location < 5:
            def state_to_country(arg: str):
                return 'United States' if len(arg) == 2 else arg
            df_salary['location'] = df_salary['location'].map(state_to_country)

        # Switch country to continent
        if location == 3:
            import pycountry_convert as pc

            def country_to_cont(name: str):
                ca2 = pc.country_name_to_country_alpha2(name)
                ccc = pc.country_alpha2_to_continent_code(ca2)
                return pc.convert_continent_code_to_continent_name(ccc)
            df_salary['location'] = df_salary['location'].map(country_to_cont)

        # Binary U.S./ Not U.S.
        if location == 4:
            def binary_us(name: str):
                return name if name == 'United States' else 'Not United States'
            df_salary['location'] = df_salary['location'].map(binary_us)

        # One-hot encode the location
        if location > 0 and location < 5:
            locations = pd.get_dummies(df_salary['location'])
            df_salary = df_salary.drop('location', axis=1)
            df_salary = df_salary.join(locations)
            features += list(locations.columns)

    # Select features and target variables
    X = df_salary[features].values
    y = np.log(df_salary['tyc'].values)

    # Dataset split: 0.6:0.2:0.2
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
    df = pd.DataFrame()

    for experiment in experiments:
        name = '-'.join(generate())
        pipeline = Pipeline([("scaler", StandardScaler()),
                             ("model", eval(experiment['model'])())])
        params = experiment['params']
        score = make_scorer(r2_score) if reg else make_scorer(
                mean_pinball_loss, alpha=quantile)
        grid = GridSearchCV(estimator=pipeline,
                            param_grid=params,
                            cv=5,
                            n_jobs=-1,
                            scoring=score,
                            refit=True)

        if log_mlflow:
            with mlflow.start_run(experiment_id=experiment.experiment_id):
                result, details = run_grid_search(
                        grid, X_train, y_train, X_test, y_test, quantile)
                mlflow.log_params(result.best_params_)
                mlflow.log_metrics(details)
                mlflow.sklearn.log_model(result, artifact_path="mod")
        else:
            result, details = run_grid_search(
                    grid, X_train, y_train, X_test, y_test, quantile)
        details['name'] = name
        df = df._append(pd.Series(details, name=experiment['model']))

        with open("../models/{}.pkl".format(name), "wb") as f:
            pickle.dump(result.best_estimator_, f)

    df.columns = df.columns.str.upper()
    df = df.round(decimals=3)
    print(df.to_markdown())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        """Run experiments on salary estimation"""
    )

    parser.add_argument(
        "config",
        help='Provide experimental config file',
        type=str
    )

    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    main(dataset=cfg['dataset'],
         experiments=cfg["experiments"],
         **cfg["track"],
         **cfg["features"],
         **cfg["details"])
