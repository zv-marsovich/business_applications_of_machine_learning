import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Конфигурация параметров предобработки данных."""

    target_col: str = "price"
    min_year: int = 1920
    max_year: int = 2017
    max_power: int = 2301
    test_size: float = 0.2
    random_state: int = 42
    drop_columns: tuple[str, ...] = (
        "date_crawled",
        "registration_month",
        "date_created",
        "number_of_pictures",
        "postal_code",
        "last_seen",
    )
    categorical_fill_values: dict[str, str] = field(
        default_factory=lambda: {
            "fuel_type": "unknown",
            "vehicle_type": "unknown",
            "repaired": "yes",
        }
    )


class PowerImputer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Заполняет нулевые и пропущенные значения мощности медианой по модели авто.

    Обучается строго на train-выборке для предотвращения утечки данных (data leakage),
    после чего удаляет вспомогательный столбец `model`.
    """

    def __init__(self, group_col: str = "model", target_col: str = "power") -> None:
        self.group_col = group_col
        self.target_col = target_col
        self.group_medians_: pd.Series = pd.Series(dtype=float)
        self.global_median_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PowerImputer":
        power_clean = X[self.target_col].replace(0, np.nan)
        self.group_medians_ = power_clean.groupby(X[self.group_col]).median()
        self.global_median_ = float(power_clean.median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        power_clean = df[self.target_col].replace(0, np.nan)
        mapped_medians = df[self.group_col].map(self.group_medians_)
        df[self.target_col] = power_clean.fillna(mapped_medians).fillna(
            self.global_median_
        )
        return df.drop(columns=[self.group_col])


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Загружает CSV-файл и приводит имена столбцов к формату snake_case."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл с данными не найден: {path}")

    df = pd.read_csv(path)
    df = df.rename(columns=lambda x: re.sub(r"(?<!^)([A-Z])", r"_\1", x).lower())
    logger.info("Загружено %d строк и %d столбцов из %s", *df.shape, path)
    return df


def standardize_categories(
    df: pd.DataFrame, config: PreprocessingConfig
) -> pd.DataFrame:
    """Унифицирует категории и заполняет допустимые пропуски.

    Безопасно для использования как на этапе обучения, так и при инференсе.
    """
    df = df.copy()
    for col, fill_val in config.categorical_fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)

    if "fuel_type" in df.columns:
        df["fuel_type"] = df["fuel_type"].replace("gasoline", "petrol")

    cols_to_drop = [c for c in config.drop_columns if c in df.columns]
    return df.drop(columns=cols_to_drop)


def clean_training_data(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Очищает обучающую выборку от выбросов, дубликатов и битых записей."""
    initial_rows = len(df)
    df = standardize_categories(df, config)

    df = df.dropna(subset=["model", "gearbox"])
    df = df.drop_duplicates()
    df = df[df[config.target_col] > 0]
    df = df[
        (df["registration_year"] >= config.min_year)
        & (df["registration_year"] < config.max_year)
    ]
    df = df[df["power"] < config.max_power]
    df = df.dropna().reset_index(drop=True)

    logger.info(
        "Очистка завершена: оставлено %d из %d строк (удалено %d)",
        len(df),
        initial_rows,
        initial_rows - len(df),
    )
    return df


def get_features_target(
    df: pd.DataFrame, target_col: str = "price"
) -> tuple[pd.DataFrame, pd.Series]:
    """Разделяет датафрейм на матрицу признаков и целевую переменную."""
    features = df.drop(columns=[target_col])
    target = df[target_col]
    return features, target


def get_columns_types(
    features: pd.DataFrame, exclude_cols: tuple[str, ...] = ("model",)
) -> tuple[list[str], list[str]]:
    """Возвращает списки категориальных и числовых признаков."""
    filtered = features.drop(columns=list(exclude_cols), errors="ignore")
    categorical_features = filtered.select_dtypes(
        include=["object", "string", "category"]
    ).columns.to_list()
    numerical_features = filtered.select_dtypes(include=["number"]).columns.to_list()
    return categorical_features, numerical_features


def create_preprocessor(
    categorical_features: list[str], numerical_features: list[str]
) -> Pipeline:
    """Создает полный sklearn-пайплайн предобработки без утечки данных."""
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("power_imputer", PowerImputer(group_col="model", target_col="power")),
            ("encoder_scaler", column_transformer),
        ]
    )


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    config: PreprocessingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разделяет данные на обучающую и тестовую выборки."""
    cfg = config or PreprocessingConfig()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )
    return (
        cast(pd.DataFrame, X_train),
        cast(pd.DataFrame, X_test),
        cast(pd.Series, y_train),
        cast(pd.Series, y_test),
    )


def prepare_training_dataset(
    filepath: str | Path,
    config: PreprocessingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """Точка входа для подготовки данных и инициализации пайплайна обучения."""
    cfg = config or PreprocessingConfig()
    raw_df = load_data(filepath)
    clean_df = clean_training_data(raw_df, cfg)
    features, target = get_features_target(clean_df, cfg.target_col)
    X_train, X_test, y_train, y_test = split_data(features, target, cfg)

    cat_cols, num_cols = get_columns_types(X_train)
    preprocessor = create_preprocessor(cat_cols, num_cols)

    return X_train, X_test, y_train, y_test, preprocessor
