from datetime import datetime

import mlflow
import mlflow.sklearn
from src.config import Settings
from src.features.schema import get_columns_types
from src.logging_config import get_logger
from src.models.persistence import save_model
from src.models.registry import MODEL_REGISTRY, TrainContext
from src.models.trainer import ModelResult, train_single_model
from src.pipeline.prepare import prepare_training_dataset
from src.tracking.mlflow_utils import init_mlflow, log_system_metrics

logger = get_logger(__name__)


def train_pipeline(settings: Settings) -> dict[str, ModelResult]:
    data_path = settings.resolve(settings.paths.raw_data)
    mlruns_dir = settings.resolve(settings.paths.mlruns_dir)
    output_path = settings.resolve(settings.training.model_output_path)
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor = prepare_training_dataset(
        data_path, settings.preprocessing
    )
    cat_cols, num_cols = get_columns_types(X_train)

    ctx = TrainContext(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        cat_cols=cat_cols,
        random_state=settings.training.random_state,
        cv_folds=settings.training.cv_folds,
    )

    init_mlflow(settings.mlflow, mlruns_dir)
    run_name = f"AutoML_{datetime.now():%Y%m%d_%H%M%S}"

    results: dict[str, ModelResult] = {}
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "n_trials": settings.training.n_trials,
                "cv_folds": settings.training.cv_folds,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_categorical_features": len(cat_cols),
                "n_numerical_features": len(num_cols),
            }
        )
        log_system_metrics()

        for name, spec in MODEL_REGISTRY.items():
            logger.info("Training %s...", name)
            results[name] = train_single_model(
                spec, ctx, X_test, y_test, settings.training.n_trials
            )

        # Логирование — внутри with (исправление бага исходника)
        for name, r in results.items():
            prefix = name.lower()
            mlflow.log_metrics(
                {
                    f"{prefix}_rmse_cv": r.cv_rmse,
                    f"{prefix}_rmse_test": r.test_metrics["rmse_test"],
                    f"{prefix}_mae_test": r.test_metrics["mae_test"],
                    f"{prefix}_r2_test": r.test_metrics["r2_test"],
                    f"{prefix}_train_time": r.train_time,
                }
            )

        best_name = min(results, key=lambda n: results[n].cv_rmse)
        best = results[best_name]
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_rmse_cv", best.cv_rmse)

        save_model(best.pipeline, output_path)
        mlflow.sklearn.log_model(best.pipeline, artifact_path="best_model")

    logger.info(
        "Best model: %s (CV RMSE: %.3f) → %s", best_name, best.cv_rmse, output_path
    )
    return results
