from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import PreprocessingConfig


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    return (
        cast(pd.DataFrame, X_train),
        cast(pd.DataFrame, X_test),
        cast(pd.Series, y_train),
        cast(pd.Series, y_test),
    )
