import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PowerImputer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Заполняет нулевые/пропущенные power медианой по модели авто.

    fit — только на train, иначе data leakage. transform удаляет group_col.
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
        mapped = df[self.group_col].map(self.group_medians_)
        df[self.target_col] = power_clean.fillna(mapped).fillna(self.global_median_)
        return df.drop(columns=[self.group_col])
