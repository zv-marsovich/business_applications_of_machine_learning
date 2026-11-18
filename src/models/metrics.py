import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "rmse_test": rmse(y_true, y_pred),
        "mae_test": float(mean_absolute_error(y_true, y_pred)),
        "r2_test": float(r2_score(y_true, y_pred)),
    }
