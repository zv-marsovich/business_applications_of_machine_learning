import pandas as pd
import pytest
from src.features.pipeline import create_preprocessor
from src.features.schema import get_columns_types, get_features_target
from src.models.registry import MODEL_REGISTRY, TrainContext
from src.models.trainer import train_single_model


@pytest.fixture
def small_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame(
        {
            "model": (
                "golf",
                "astra",
                "focus",
                "polo",
                "golf",
                "astra",
                "focus",
                "polo",
            ),
            "fuel_type": ("petrol", "diesel", "petrol", "diesel") * 2,
            "gearbox": ("manual", "auto") * 4,
            "power": (100, 110, 120, 130, 140, 150, 160, 170),
            "registration_year": (2010, 2012, 2014, 2015, 2011, 2013, 2016, 2009),
            "price": (5000, 6000, 7000, 8000, 5500, 6500, 7500, 8500),
        }
    )
    return get_features_target(df, "price")


@pytest.fixture
def train_ctx(small_dataset: tuple[pd.DataFrame, pd.Series]) -> TrainContext:
    features, target = small_dataset
    cat_cols, num_cols = get_columns_types(features)
    return TrainContext(
        X_train=features,
        y_train=target,
        preprocessor=create_preprocessor(cat_cols, num_cols),
        cat_cols=cat_cols,
        random_state=42,
        cv_folds=2,
    )


def test_train_single_model_baseline(
    train_ctx: TrainContext,
    small_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """LinearRegression: без Optuna, cv_rmse > 0, метрики на месте."""
    features, target = small_dataset
    result = train_single_model(
        MODEL_REGISTRY["LinearRegression"], train_ctx, features, target, n_trials=0
    )

    assert result.name == "LinearRegression"
    assert result.cv_rmse > 0
    assert set(result.test_metrics) == {"rmse_test", "mae_test", "r2_test"}
    assert result.best_params is None
    assert result.pipeline.predict(features).shape == (len(features),)


def test_train_single_model_with_optuna(
    train_ctx: TrainContext,
    small_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """XGBoost с n_trials=1: изучает best_params и собирает pipeline."""
    features, target = small_dataset
    result = train_single_model(
        MODEL_REGISTRY["XGBoost"], train_ctx, features, target, n_trials=1
    )

    assert result.name == "XGBoost"
    assert result.best_params is not None
    assert "n_estimators" in result.best_params
    assert result.cv_rmse > 0
    assert result.train_time > 0


def test_train_single_model_is_reproducible(
    train_ctx: TrainContext,
    small_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """С random_state=42 два прогона LinearRegression дают одинаковый cv_rmse."""
    features, target = small_dataset
    r1 = train_single_model(
        MODEL_REGISTRY["LinearRegression"], train_ctx, features, target, n_trials=0
    )
    r2 = train_single_model(
        MODEL_REGISTRY["LinearRegression"], train_ctx, features, target, n_trials=0
    )
    assert r1.cv_rmse == pytest.approx(r2.cv_rmse)
