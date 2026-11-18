import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.logging_config import get_logger
from src.models.metrics import evaluate_predictions
from src.models.registry import ModelSpec, TrainContext
from src.optimization.optimize import (
    build_pipeline,
    cross_val_rmse,
    make_objective,
    run_study,
)

logger = get_logger(__name__)


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    cv_rmse: float
    test_metrics: dict[str, float]
    best_params: dict[str, Any] | None
    train_time: float


def train_single_model(
    spec: ModelSpec,
    ctx: TrainContext,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int,
) -> ModelResult:
    """Оптимизация (если есть) → fit на полном train → метрики на test."""
    start = time.perf_counter()

    if spec.suggest_params is None:
        params = spec.fixed_params(ctx)
        cv_rmse = cross_val_rmse(spec, ctx, params)
        best_params: dict[str, Any] | None = None
    else:
        study = run_study(make_objective(spec, ctx), n_trials, ctx.random_state)
        params = {**spec.fixed_params(ctx), **study.best_params}
        cv_rmse = float(study.best_value)
        best_params = study.best_params

    pipeline = build_pipeline(spec, ctx, params)
    pipeline.fit(ctx.X_train, ctx.y_train)
    test_metrics = evaluate_predictions(y_test, pipeline.predict(X_test))
    elapsed = time.perf_counter() - start

    logger.info(
        "%s | CV RMSE: %.3f | Test RMSE: %.3f | R2: %.3f | Time: %.2fs",
        spec.name,
        cv_rmse,
        test_metrics["rmse_test"],
        test_metrics["r2_test"],
        elapsed,
    )
    return ModelResult(
        name=spec.name,
        pipeline=pipeline,
        cv_rmse=cv_rmse,
        test_metrics=test_metrics,
        best_params=best_params,
        train_time=elapsed,
    )
