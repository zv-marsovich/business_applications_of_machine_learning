import pandas as pd


def get_features_target(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target_col]), df[target_col]


def get_columns_types(
    features: pd.DataFrame, exclude_cols: tuple[str, ...] = ("model",)
) -> tuple[list[str], list[str]]:
    filtered = features.drop(columns=list(exclude_cols), errors="ignore")
    categorical = filtered.select_dtypes(
        include=["object", "string", "category"]
    ).columns.to_list()
    numerical = filtered.select_dtypes(include=["number"]).columns.to_list()
    return categorical, numerical
