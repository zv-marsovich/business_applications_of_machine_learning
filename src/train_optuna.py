import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import psutil
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .preprocessing import (
    PowerImputer,
    PreprocessingConfig,
    get_columns_types,
    prepare_training_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class TrainingConfig:
    """Конфигурация запуска обучения и трекинга экспериментов."""

    project_root: Path = Path(__file__).resolve().parent.parent
    data_path: Path = project_root / "data" / "autos.csv"
    models_dir: Path = project_root / "models"
    mlruns_dir: Path = project_root / "mlruns"
    experiment_name: str = "car_price_prediction_optuna"
    n_trials: int = 10
    cv_folds: int = 3
    random_state: int = 42


def rmse_scorer(y_true: Any, y_pred: Any) -> float:
    """Расчет метрики RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def log_system_metrics() -> None:
    """Логирование системных ресурсов в MLflow."""
    mlflow.log_metrics(
        {
            "system_cpu_percent": psutil.cpu_percent(interval=0.5),
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_disk_usage": psutil.disk_usage("/").percent,
        }
    )


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Расчет набора регрессионных метрик на отложенной выборке."""
    return {
        "rmse_test": rmse_scorer(y_true, y_pred),
        "mae_test": float(mean_absolute_error(y_true, y_pred)),
        "r2_test": float(r2_score(y_true, y_pred)),
    }


def objective_xgb(
    trial: optuna.Trial,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Pipeline,
    cv_folds: int = 3,
    random_state: int = 42,
) -> float:
    """Целевая функция Optuna для XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": random_state,
        "objective": "reg:squarederror",
        "verbosity": 0,
    }

    pipeline = Pipeline(
        [("preprocessor", clone(preprocessor)), ("model", XGBRegressor(**params))]
    )
    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return float(-scores.mean())


def objective_catboost(
    trial: optuna.Trial,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cat_cols: list[str],
    cv_folds: int = 3,
    random_state: int = 42,
) -> float:
    """Целевая функция Optuna для CatBoost с корректным импьютингом без утечек."""
    params = {
        "iterations": trial.suggest_int("iterations", 500, 1500, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "cat_features": cat_cols,
        "random_seed": random_state,
        "verbose": 0,
        "loss_function": "RMSE",
    }

    pipeline = Pipeline(
        [
            ("power_imputer", PowerImputer(group_col="model", target_col="power")),
            ("model", CatBoostRegressor(**params)),
        ]
    )
    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return float(-scores.mean())


def objective_lgbm(
    trial: optuna.Trial,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Pipeline,
    cv_folds: int = 3,
    random_state: int = 42,
) -> float:
    """Целевая функция Optuna для LightGBM."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": random_state,
        "verbose": -1,
    }

    pipeline = Pipeline(
        [("preprocessor", clone(preprocessor)), ("model", LGBMRegressor(**params))]
    )
    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return float(-scores.mean())


def train_with_optuna(
    train_config: TrainingConfig | None = None,
    prep_config: PreprocessingConfig | None = None,
) -> tuple[Pipeline, dict[str, dict[str, Any]], float]:
    """Запускает полный цикл предобработки, подбора гиперпараметров и логирования."""
    if train_config is None:
        train_config = TrainingConfig()
    if prep_config is None:
        prep_config = PreprocessingConfig()
    train_config.models_dir.mkdir(parents=True, exist_ok=True)
    train_config.mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(train_config.mlruns_dir.as_uri())

    logger.info(
        "Старт AutoML (CPU: %s, RAM: %.1f GB)",
        psutil.cpu_count(),
        psutil.virtual_memory().total / (1024**3),
    )

    X_train, X_test, y_train, y_test, base_preprocessor = prepare_training_dataset(
        train_config.data_path, prep_config
    )
    cat_cols, num_cols = get_columns_types(X_train)
    logger.info(
        "Train: %d, Test: %d | Категориальных: %d, Числовых: %d",
        len(X_train),
        len(X_test),
        len(cat_cols),
        len(num_cols),
    )

    results: dict[str, dict[str, Any]] = {}
    best_pipelines: dict[str, Pipeline] = {}

    mlflow.set_experiment(train_config.experiment_name)
    run_name = f"AutoML_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "n_trials": train_config.n_trials,
                "cv_folds": train_config.cv_folds,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_categorical_features": len(cat_cols),
                "n_numerical_features": len(num_cols),
            }
        )
        log_system_metrics()

        # 1. Baseline: LinearRegression
        logger.info("Обучение baseline модели LinearRegression...")
        start_time = time.perf_counter()
        lr_pipeline = Pipeline(
            [("preprocessor", clone(base_preprocessor)), ("model", LinearRegression())]
        )
        lr_cv_scores = cross_val_score(
            lr_pipeline,
            X_train,
            y_train,
            cv=train_config.cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        lr_pipeline.fit(X_train, y_train)
        lr_time = time.perf_counter() - start_time

        lr_metrics = evaluate_predictions(y_test, lr_pipeline.predict(X_test))
        results["LinearRegression"] = {
            "rmse_cv": float(-lr_cv_scores.mean()),
            **lr_metrics,
            "train_time": lr_time,
            "best_params": None,
        }
        best_pipelines["LinearRegression"] = lr_pipeline

        # 2. Оптимизация градиентных бустингов через Optuna
        sampler = TPESampler(seed=train_config.random_state)
        pruner = MedianPruner()

        # XGBoost
        logger.info("Оптимизация XGBoost (%d trials)...", train_config.n_trials)
        start_time = time.perf_counter()
        study_xgb = optuna.create_study(
            direction="minimize",
            sampler=clone(sampler)
            if hasattr(sampler, "get_params")
            else TPESampler(seed=train_config.random_state),
            pruner=pruner,
        )
        study_xgb.optimize(
            lambda trial: objective_xgb(
                trial,
                X_train,
                y_train,
                base_preprocessor,
                train_config.cv_folds,
                train_config.random_state,
            ),
            n_trials=train_config.n_trials,
        )
        xgb_pipeline = Pipeline(
            [
                ("preprocessor", clone(base_preprocessor)),
                (
                    "model",
                    XGBRegressor(
                        **study_xgb.best_params,
                        random_state=train_config.random_state,
                        objective="reg:squarederror",
                        verbosity=0,
                    ),
                ),
            ]
        )
        xgb_pipeline.fit(X_train, y_train)
        xgb_time = time.perf_counter() - start_time
        results["XGBoost"] = {
            "rmse_cv": float(study_xgb.best_value),
            **evaluate_predictions(y_test, xgb_pipeline.predict(X_test)),
            "train_time": xgb_time,
            "best_params": study_xgb.best_params,
        }
        best_pipelines["XGBoost"] = xgb_pipeline
        # LightGBM
        logger.info("Оптимизация LightGBM (%d trials)...", train_config.n_trials)
        start_time = time.perf_counter()
        study_lgb = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=train_config.random_state),
            pruner=pruner,
        )
        study_lgb.optimize(
            lambda trial: objective_lgbm(
                trial,
                X_train,
                y_train,
                base_preprocessor,
                train_config.cv_folds,
                train_config.random_state,
            ),
            n_trials=train_config.n_trials,
        )
        lgb_pipeline = Pipeline(
            [
                ("preprocessor", clone(base_preprocessor)),
                (
                    "model",
                    LGBMRegressor(
                        **study_lgb.best_params,
                        random_state=train_config.random_state,
                        verbose=-1,
                    ),
                ),
            ]
        )
        lgb_pipeline.fit(X_train, y_train)
        lgb_time = time.perf_counter() - start_time
        results["LightGBM"] = {
            "rmse_cv": float(study_lgb.best_value),
            **evaluate_predictions(y_test, lgb_pipeline.predict(X_test)),
            "train_time": lgb_time,
            "best_params": study_lgb.best_params,
        }
        best_pipelines["LightGBM"] = lgb_pipeline
        # CatBoost
        logger.info("Оптимизация CatBoost (%d trials)...", train_config.n_trials)
        start_time = time.perf_counter()
        study_cb = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=train_config.random_state),
            pruner=pruner,
        )
        study_cb.optimize(
            lambda trial: objective_catboost(
                trial,
                X_train,
                y_train,
                cat_cols,
                train_config.cv_folds,
                train_config.random_state,
            ),
            n_trials=train_config.n_trials,
        )
        cb_pipeline = Pipeline(
            [
                ("power_imputer", PowerImputer(group_col="model", target_col="power")),
                (
                    "model",
                    CatBoostRegressor(
                        **study_cb.best_params,
                        cat_features=cat_cols,
                        random_seed=train_config.random_state,
                        verbose=0,
                        loss_function="RMSE",
                    ),
                ),
            ]
        )
        cb_pipeline.fit(X_train, y_train)
        cb_time = time.perf_counter() - start_time
        results["CatBoost"] = {
            "rmse_cv": float(study_cb.best_value),
            **evaluate_predictions(y_test, cb_pipeline.predict(X_test)),
            "train_time": cb_time,
            "best_params": study_cb.best_params,
        }
        best_pipelines["CatBoost"] = cb_pipeline
    # Логирование метрик всех моделей в MLflow
    for model_name, metrics in results.items():
        prefix = model_name.lower()
        mlflow.log_metrics(
            {
                f"{prefix}_rmse_cv": metrics["rmse_cv"],
                f"{prefix}_rmse_test": metrics["rmse_test"],
                f"{prefix}_mae_test": metrics["mae_test"],
                f"{prefix}_r2_test": metrics["r2_test"],
                f"{prefix}_train_time": metrics["train_time"],
            }
        )
        logger.info(
            "%s -> CV RMSE: %.3f | Test RMSE: %.3f | R2: %.3f | Время: %.2fs",
            model_name,
            metrics["rmse_cv"],
            metrics["rmse_test"],
            metrics["r2_test"],
            metrics["train_time"],
        )
    # Выбор и сохранение лучшего единого Pipeline (препроцессор + модель)
    best_model_name = min(results, key=lambda name: results[name]["rmse_cv"])
    best_pipeline = best_pipelines[best_model_name]
    best_rmse = results[best_model_name]["rmse_cv"]
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_rmse_cv", best_rmse)
    artifact_path = train_config.models_dir / "best_model_pipeline.joblib"
    joblib.dump(best_pipeline, artifact_path)
    mlflow.sklearn.log_model(best_pipeline, artifact_path="best_model")
    logger.info(
        "Лучшая модель: %s (CV RMSE: %.3f). Пайплайн сохранен в %s",
        best_model_name,
        best_rmse,
        artifact_path,
    )
    return best_pipeline, results, best_rmse


if __name__ == "__main__":
    train_with_optuna()
