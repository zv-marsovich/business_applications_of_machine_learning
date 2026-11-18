import numpy as np
import pandas as pd
import pytest
from src.models.metrics import evaluate_predictions, rmse

# ---------- rmse ----------


def test_rmse_positive() -> None:
    """RMSE на неидеальном предсказании: > 0 и тип float."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    result = rmse(y_true, y_pred)

    assert result > 0
    assert isinstance(result, float)


def test_rmse_perfect_prediction() -> None:
    """RMSE на идеальном предсказании равен 0."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    assert rmse(y_true, y_pred) == 0.0


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([1, 2, 3], [2, 3, 4], 1.0),  # постоянная ошибка 1
        ([0, 0, 0], [3, 4, 0], (25 / 3) ** 0.5),  # проверка формулы
        ([-5, -4, -3], [-4, -3, -2], 1.0),  # отрицательные значения
    ],
)
def test_rmse_formula(y_true: list[int], y_pred: list[int], expected: float) -> None:
    assert rmse(np.array(y_true), np.array(y_pred)) == pytest.approx(expected)


def test_rmse_accepts_pandas_series() -> None:
    """Реальный вход в проде — pd.Series, а не np.ndarray."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 2.0, 4.0])

    assert rmse(y_true, y_pred) == pytest.approx(1.0 / (3**0.5))


# ---------- evaluate_predictions ----------


def test_evaluate_predictions_returns_all_metrics() -> None:
    """Возвращает ровно три ключа: rmse_test, mae_test, r2_test."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.0, 4.2])

    metrics = evaluate_predictions(y_true, y_pred)

    assert set(metrics) == {"rmse_test", "mae_test", "r2_test"}
    assert all(isinstance(v, float) for v in metrics.values())


def test_evaluate_predictions_perfect() -> None:
    """Идеальное предсказание: RMSE=0, MAE=0, R2=1."""
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    metrics = evaluate_predictions(y, y.to_numpy())

    assert metrics["rmse_test"] == pytest.approx(0.0)
    assert metrics["mae_test"] == pytest.approx(0.0)
    assert metrics["r2_test"] == pytest.approx(1.0)


def test_evaluate_predictions_consistency_with_rmse() -> None:
    """evaluate_predictions.rmse_test совпадает с отдельным вызовом rmse."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.2, 1.8, 3.1, 3.9, 5.3])

    assert evaluate_predictions(y_true, y_pred)["rmse_test"] == pytest.approx(
        rmse(y_true, y_pred)
    )
