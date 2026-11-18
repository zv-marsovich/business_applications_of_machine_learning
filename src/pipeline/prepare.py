from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from src.config import PreprocessingConfig
from src.data.loader import load_data
from src.data.validation import validate_clean_data, validate_raw_schema
from src.features.cleaning import clean_training_data
from src.features.pipeline import create_preprocessor
from src.features.schema import get_columns_types, get_features_target
from src.features.split import split_data


def prepare_training_dataset(
    filepath: str | Path,
    config: PreprocessingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """Оркестратор подготовки train/test и sklearn-пайплайна."""
    cfg = config or PreprocessingConfig()

    raw_df = load_data(filepath)
    validate_raw_schema(raw_df)

    clean_df = clean_training_data(raw_df, cfg)
    validate_clean_data(clean_df, cfg)

    features, target = get_features_target(clean_df, cfg.target_col)
    X_train, X_test, y_train, y_test = split_data(features, target, cfg)

    cat_cols, num_cols = get_columns_types(X_train)
    preprocessor = create_preprocessor(cat_cols, num_cols)

    return X_train, X_test, y_train, y_test, preprocessor
