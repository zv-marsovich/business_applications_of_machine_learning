from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from src.config import PreprocessingConfig
from src.data.loader import normalize_columns
from src.features.pipeline import create_preprocessor
from src.features.schema import get_columns_types, get_features_target
from src.models.persistence import save_model
from src.pipeline.predict import (
    drop_target_if_present,
    predict_batch,
    predict_from_file,
    prepare_inference_features,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ("golf", "astra", "focus", "polo", "golf", "astra"),
            "fuel_type": ("petrol", "diesel", "petrol", "diesel", "petrol", "diesel"),
            "gearbox": ("manual", "auto", "manual", "auto", "manual", "auto"),
            "power": (100, 110, 120, 130, 140, 150),
            "registration_year": (2010, 2012, 2014, 2015, 2011, 2013),
            "price": (5000, 6000, 7000, 8000, 5500, 6500),
        }
    )


def _fit_toy_pipeline(df: pd.DataFrame) -> Pipeline:
    features, target = get_features_target(df, "price")
    cat_cols, num_cols = get_columns_types(features)
    pipeline = Pipeline(
        [
            ("preprocessor", create_preprocessor(cat_cols, num_cols)),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(features, target)
    return pipeline


# ---------- normalize_columns ----------


def test_normalize_columns_camel_case() -> None:
    df = pd.DataFrame({"VehicleType": ("sedan",), "RegistrationYear": (2010,)})
    out = normalize_columns(df)
    assert tuple(out.columns) == ("vehicle_type", "registration_year")


def test_normalize_columns_idempotent() -> None:
    df = pd.DataFrame({"fuel_type": ("petrol",)})
    out = normalize_columns(df)
    assert tuple(out.columns) == ("fuel_type",)


# ---------- prepare_inference_features ----------


def test_prepare_inference_features_keeps_all_rows() -> None:
    """Ключевое отличие от train: ни одна строка не удаляется."""
    cfg = PreprocessingConfig()
    df = pd.DataFrame(
        {
            "model": ("golf", None, "astra"),  # одна модель = None
            "gearbox": ("manual", "auto", "auto"),
            "power": (100, 0, 120),  # одна мощность = 0
            "registration_year": (2010, 1900, 2012),  # один год вне диапазона
            "price": (5000, 6000, 7000),
            "fuel_type": ("petrol", "gasoline", None),
            "postal_code": (1, 2, 3),  # служебная колонка
        }
    )

    result = prepare_inference_features(df, cfg)

    assert len(result) == 3  # все строки на месте
    assert "postal_code" not in result.columns  # служебные удалены
    assert result["fuel_type"].tolist() == [
        "petrol",
        "petrol",
        "unknown",
    ]  # gasoline→petrol, NaN→unknown


def test_prepare_inference_features_empty_raises() -> None:
    cfg = PreprocessingConfig()
    with pytest.raises(ValueError, match="пуст"):
        prepare_inference_features(pd.DataFrame(), cfg)


# ---------- drop_target_if_present ----------


def test_drop_target_if_present_yes() -> None:
    df = pd.DataFrame({"price": (1, 2), "power": (10, 20)})
    features, actual = drop_target_if_present(df, "price")
    assert "price" not in features.columns
    assert actual is not None
    assert actual.tolist() == [1, 2]


def test_drop_target_if_present_no() -> None:
    df = pd.DataFrame({"power": (10, 20)})
    features, actual = drop_target_if_present(df, "price")
    assert features is df
    assert actual is None


# ---------- predict_batch ----------


def test_predict_batch_preserves_index() -> None:
    """Индекс должен сохраниться — иначе нельзя сматчить с исходным входом."""
    df = _toy_df()
    pipeline = _fit_toy_pipeline(df)
    features, _ = drop_target_if_present(df, "price")
    features.index = pd.Index([10, 20, 30, 40, 50, 60])

    preds = predict_batch(pipeline, features)

    assert isinstance(preds, pd.Series)
    assert preds.index.tolist() == [10, 20, 30, 40, 50, 60]
    assert preds.name == "predicted_price"
    assert not preds.isna().any()


# ---------- predict_from_file (roundtrip) ----------


def test_predict_from_file_roundtrip(tmp_path: Path) -> None:
    """Train → save → load → predict. Проверяем формы и инварианты."""
    df = _toy_df()
    pipeline = _fit_toy_pipeline(df)

    model_path = tmp_path / "model.joblib"
    save_model(pipeline, model_path)

    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    output_csv = tmp_path / "out.csv"
    result = predict_from_file(
        input_path=input_csv,
        output_path=output_csv,
        model_path=model_path,
        config=PreprocessingConfig(),
    )

    assert len(result) == len(df)
    assert "predicted_price" in result.columns
    assert "actual_price" in result.columns  # price был в input
    assert output_csv.exists()
    assert (result["predicted_price"] > 0).all()


def test_predict_from_file_without_target(tmp_path: Path) -> None:
    """Input без target — actual_price в результате быть не должно."""
    df = _toy_df()
    pipeline = _fit_toy_pipeline(df)
    model_path = tmp_path / "model.joblib"
    save_model(pipeline, model_path)

    df_no_target = df.drop(columns=["price"])
    input_csv = tmp_path / "input.csv"
    df_no_target.to_csv(input_csv, index=False)

    result = predict_from_file(
        input_path=input_csv,
        output_path=None,
        model_path=model_path,
        config=PreprocessingConfig(),
    )

    assert len(result) == len(df_no_target)
    assert "actual_price" not in result.columns
    assert "predicted_price" in result.columns


def test_predict_from_file_missing_model(tmp_path: Path) -> None:
    df = _toy_df()
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    with pytest.raises(FileNotFoundError, match="Модель не найдена"):
        predict_from_file(
            input_path=input_csv,
            output_path=None,
            model_path=tmp_path / "nope.joblib",
            config=PreprocessingConfig(),
        )
