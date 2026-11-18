from collections.abc import Callable
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.features.transformers import PowerImputer
from src.models.registry import ModelSpec, TrainContext

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_pipeline(
    spec: ModelSpec, ctx: TrainContext, params: dict[str, Any]
) -> Pipeline:
    """Единая точка сборки Pipeline: preprocessor-или-power_imputer + model."""
    estimator = spec.estimator_cls(**params)
    if spec.pipeline_kind == "full":
        return Pipeline(
            [
                ("preprocessor", clone(ctx.preprocessor)),
                ("model", estimator),
            ]
        )
    return Pipeline(
        [
            ("power_imputer", PowerImputer(group_col="model", target_col="power")),
            ("model", estimator),
        ]
    )


def cross_val_rmse(spec: ModelSpec, ctx: TrainContext, params: dict[str, Any]) -> float:
    pipeline = build_pipeline(spec, ctx, params)
    scores = cross_val_score(
        pipeline,
        ctx.X_train,
        ctx.y_train,
        cv=ctx.cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return float(-scores.mean())


def make_objective(
    spec: ModelSpec, ctx: TrainContext
) -> Callable[[optuna.Trial], float]:
    if spec.suggest_params is None:
        raise ValueError(f"Model {spec.name} has no search space")

    def objective(trial: optuna.Trial) -> float:
        params = {**spec.fixed_params(ctx), **spec.suggest_params(trial)}  # type: ignore[misc]
        return cross_val_rmse(spec, ctx, params)

    return objective


def run_study(
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    seed: int,
    direction: str = "minimize",
) -> optuna.Study:
    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)
    return study
