import pandas as pd
from src.config import PreprocessingConfig
from src.logging_config import get_logger

logger = get_logger(__name__)


def standardize_categories(
    df: pd.DataFrame, config: PreprocessingConfig
) -> pd.DataFrame:
    """Унификация категорий и заполнение допустимых пропусков.

    Безопасна и для train, и для inference.
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
    """Очистка только обучающей выборки: выбросы, дубликаты, битые записи."""
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
