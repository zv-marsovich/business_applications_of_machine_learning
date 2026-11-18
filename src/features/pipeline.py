from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features.transformers import PowerImputer


def create_preprocessor(
    categorical_features: list[str], numerical_features: list[str]
) -> Pipeline:
    """Sklearn-пайплайн без утечки: PowerImputer → ColumnTransformer."""
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
