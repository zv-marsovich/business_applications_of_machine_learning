from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.preprocessing import (
    PowerImputer,
    PreprocessingConfig,
    clean_training_data,
    create_preprocessor,
    get_columns_types,
    get_features_target,
    load_data,
    prepare_training_dataset,
    split_data,
    standardize_categories,
)


def test_load_data(tmp_path: Path) -> None:
    """Тест загрузки CSV и конвертации CamelCase в snake_case."""
    csv_file = tmp_path / "autos.csv"
    raw_df = pd.DataFrame(
        {
            "Price": (1500, 3200),
            "VehicleType": ("sedan", "suv"),
            "RegistrationYear": (2010, 2015),
        }
    )
    raw_df.to_csv(csv_file, index=False)

    df = load_data(csv_file)
    assert tuple(df.columns) == ("price", "vehicle_type", "registration_year")
    assert len(df) == 2


def test_load_data_file_not_found(tmp_path: Path) -> None:
    """Тест исключения FileNotFoundError при отсутствии файла."""
    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_data(missing_file)


def test_standardize_categories() -> None:
    """Тест унификации категорий, заполнения пропусков и удаления лишних колонок."""
    cfg = PreprocessingConfig()
    df = pd.DataFrame(
        {
            "fuel_type": (None, "gasoline", "petrol"),
            "vehicle_type": (None, "bus", "car"),
            "repaired": (None, "yes", "no"),
            "postal_code": (101000, 102000, 103000),
            "date_crawled": ("2016-03-01", "2016-03-02", "2016-03-03"),
        }
    )

    result = standardize_categories(df, cfg)
    assert tuple(result["fuel_type"]) == ("unknown", "petrol", "petrol")
    assert tuple(result["vehicle_type"]) == ("unknown", "bus", "car")
    assert tuple(result["repaired"]) == ("yes", "yes", "no")
    assert "postal_code" not in result.columns
    assert "date_crawled" not in result.columns


def test_clean_training_data() -> None:
    """Тест фильтрации выбросов по цене, году, мощности и удаления дубликатов."""
    cfg = PreprocessingConfig()
    df = pd.DataFrame(
        {
            "price": (0, 5000, 5000, 10000, 12000, 8000),
            "registration_year": (1900, 2005, 2005, 2020, 2012, 2011),
            "power": (100, 120, 120, 150, 3000, 110),
            "model": ("golf", "astra", "astra", "focus", "civic", "polo"),
            "gearbox": ("manual", "auto", "auto", "manual", "auto", None),
            "fuel_type": ("petrol", "diesel", "diesel", "petrol", "petrol", "diesel"),
            "vehicle_type": ("sedan", "suv", "suv", "wagon", "coupe", "small"),
            "repaired": ("no", "no", "no", "yes", "no", "no"),
        }
    )

    result = clean_training_data(df, cfg)
    assert len(result) == 1
    assert tuple(result["model"]) == ("astra",)
    assert tuple(result["price"]) == (5000,)


def test_power_imputer() -> None:
    """Тест импьютации нулевой мощности без утечки данных и удаления столбца model."""
    train_df = pd.DataFrame(
        {
            "model": ("golf", "golf", "astra", "astra"),
            "power": (100, 200, 110, 130),
            "mileage": (50000, 60000, 70000, 80000),
        }
    )
    test_df = pd.DataFrame(
        {
            "model": ("golf", "astra", "unknown_model"),
            "power": (0, np.nan, 0),
            "mileage": (55000, 75000, 90000),
        }
    )

    imputer = PowerImputer(group_col="model", target_col="power")
    imputer.fit(train_df)
    transformed = imputer.transform(test_df)

    assert "model" not in transformed.columns
    assert "mileage" in transformed.columns
    assert tuple(transformed["power"]) == (150.0, 120.0, 120.0)


def test_get_features_target() -> None:
    """Тест разделения датафрейма на признаки и таргет."""
    df = pd.DataFrame(
        {
            "price": (1000, 2000, 3000),
            "model": ("golf", "astra", "focus"),
            "power": (100, 120, 140),
        }
    )

    features, target = get_features_target(df, target_col="price")
    assert "price" not in features.columns
    assert features.shape == (3, 2)
    assert tuple(target) == (1000, 2000, 3000)


def test_get_columns_types() -> None:
    """Тест разделения признаков на категориальные и числовые с исключением model."""
    df = pd.DataFrame(
        {
            "model": ("golf", "astra"),
            "fuel_type": ("petrol", "diesel"),
            "gearbox": ("manual", "auto"),
            "power": (100, 120),
            "registration_year": (2010, 2012),
        }
    )

    cat_cols, num_cols = get_columns_types(df)
    assert tuple(cat_cols) == ("fuel_type", "gearbox")
    assert tuple(num_cols) == ("power", "registration_year")


def test_create_preprocessor_pipeline() -> None:
    """Тест сборки и обучения полного sklearn Pipeline (PowerImputer + ColumnTransformer)."""
    X_train = pd.DataFrame(
        {
            "model": ("golf", "astra", "focus", "polo"),
            "fuel_type": ("petrol", "diesel", "petrol", "unknown"),
            "power": (100, 0, 140, 90),
            "registration_year": (2010, 2012, 2015, 2008),
        }
    )
    X_test = pd.DataFrame(
        {
            "model": ("golf", "new_brand_model"),
            "fuel_type": ("diesel", "electric"),
            "power": (0, 110),
            "registration_year": (2011, 2016),
        }
    )

    cat_cols, num_cols = get_columns_types(X_train)
    preprocessor = create_preprocessor(cat_cols, num_cols)

    assert isinstance(preprocessor, Pipeline)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    assert X_train_transformed.shape == (4, 4)
    assert X_test_transformed.shape == (2, 4)
    assert not np.isnan(X_test_transformed).any()


def test_split_data() -> None:
    """Тест разбиения выборки на train и test с заданной пропорцией."""
    cfg = PreprocessingConfig(test_size=0.25, random_state=42)
    features = pd.DataFrame({"power": (100, 110, 120, 130, 140, 150, 160, 170)})
    target = pd.Series((1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700))

    X_train, X_test, y_train, y_test = split_data(features, target, cfg)
    assert len(X_train) == 6
    assert len(X_test) == 2
    assert len(y_train) == 6
    assert len(y_test) == 2


def test_prepare_training_dataset(tmp_path: Path) -> None:
    """Сквозной тест функции prepare_training_dataset от CSV до готового Pipeline."""
    csv_file = tmp_path / "full_autos.csv"
    raw_df = pd.DataFrame(
        {
            "Price": (3000, 4500, 6000, 7500, 9000, 0),
            "RegistrationYear": (2008, 2010, 2012, 2014, 2015, 2011),
            "Power": (100, 0, 130, 150, 170, 120),
            "Model": ("golf", "golf", "astra", "focus", "polo", "civic"),
            "Gearbox": ("manual", "auto", "manual", "auto", "manual", "auto"),
            "FuelType": ("petrol", "gasoline", None, "diesel", "petrol", "petrol"),
            "VehicleType": ("sedan", "wagon", "suv", None, "small", "coupe"),
            "Repaired": ("no", None, "yes", "no", "no", "no"),
            "PostalCode": (10000, 20000, 30000, 40000, 50000, 60000),
        }
    )
    raw_df.to_csv(csv_file, index=False)

    cfg = PreprocessingConfig(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test, preprocessor = prepare_training_dataset(
        csv_file, cfg
    )

    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    assert X_train_proc.shape[0] == 4
    assert X_test_proc.shape == (1, X_train_proc.shape[1])
