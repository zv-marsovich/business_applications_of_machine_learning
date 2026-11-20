from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


@dataclass(frozen=True)
class TrainContext:
    """Всё, что нужно objective-функциям: данные + инфраструктура."""

    X_train: pd.DataFrame
    y_train: pd.Series
    preprocessor: Pipeline
    cat_cols: list[str]
    random_state: int
    cv_folds: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_cls: type
    suggest_params: Callable[[optuna.Trial], dict[str, Any]] | None
    fixed_params: Callable[[TrainContext], dict[str, Any]]
    pipeline_kind: Literal["full", "power_only"] = field(default="full")


def _xgb_search(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _xgb_fixed(ctx: TrainContext) -> dict[str, Any]:
    return {
        "random_state": ctx.random_state,
        "objective": "reg:squarederror",
        "verbosity": 0,
    }


def _lgbm_search(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _lgbm_fixed(ctx: TrainContext) -> dict[str, Any]:
    return {"random_state": ctx.random_state, "verbose": -1}


def _cb_search(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 500, 1500, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }


def _cb_fixed(ctx: TrainContext) -> dict[str, Any]:
    return {
        "cat_features": tuple(ctx.cat_cols),
        "random_seed": ctx.random_state,
        "verbose": 0,
        "loss_function": "RMSE",
    }


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "LinearRegression": ModelSpec(
        name="LinearRegression",
        estimator_cls=LinearRegression,
        suggest_params=None,
        fixed_params=lambda ctx: {},
        pipeline_kind="full",
    ),
    "XGBoost": ModelSpec(
        name="XGBoost",
        estimator_cls=XGBRegressor,
        suggest_params=_xgb_search,
        fixed_params=_xgb_fixed,
        pipeline_kind="full",
    ),
    "LightGBM": ModelSpec(
        name="LightGBM",
        estimator_cls=LGBMRegressor,
        suggest_params=_lgbm_search,
        fixed_params=_lgbm_fixed,
        pipeline_kind="full",
    ),
    "CatBoost": ModelSpec(
        name="CatBoost",
        estimator_cls=CatBoostRegressor,
        suggest_params=_cb_search,
        fixed_params=_cb_fixed,
        pipeline_kind="power_only",
    ),
}
