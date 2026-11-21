from unittest.mock import MagicMock

import pytest
from src.models.selection import select_best_model
from src.models.trainer import ModelResult


def _make_result(name: str, cv_rmse: float, train_time: float) -> ModelResult:
    return ModelResult(
        name=name,
        pipeline=MagicMock(),
        cv_rmse=cv_rmse,
        test_metrics={"rmse_test": cv_rmse, "mae_test": 0.0, "r2_test": 0.0},
        best_params=None,
        train_time=train_time,
    )


def test_selects_fastest_within_tolerance() -> None:
    """Модель на 2% хуже, но в 10 раз быстрее — должна победить при tolerance=3%."""
    results = {
        "slow_but_best": _make_result("slow_but_best", 100.0, 100.0),
        "fast_almost_best": _make_result("fast_almost_best", 102.0, 10.0),
        "very_fast_but_worse": _make_result("very_fast_but_worse", 120.0, 1.0),
    }
    assert select_best_model(results, tolerance=0.03) == "fast_almost_best"


def test_zero_tolerance_falls_back_to_quality() -> None:
    """При tolerance=0 поведение как раньше — побеждает лучший RMSE."""
    results = {
        "best": _make_result("best", 100.0, 100.0),
        "faster_but_worse": _make_result("faster_but_worse", 101.0, 10.0),
    }
    assert select_best_model(results, tolerance=0.0) == "best"


def test_all_outside_tolerance_picks_quality_leader() -> None:
    """Если ни одна не укладывается в tolerance — берём лучшую по RMSE."""
    results = {
        "best": _make_result("best", 100.0, 100.0),
        "way_worse": _make_result("way_worse", 150.0, 1.0),
    }
    assert select_best_model(results, tolerance=0.03) == "best"


def test_empty_results_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_best_model({}, tolerance=0.03)


def test_selection_is_deterministic() -> None:
    results = {
        "a": _make_result("a", 100.0, 50.0),
        "b": _make_result("b", 101.0, 50.0),  # same time
    }
    # При равном времени min() берёт по ключу cv_rmse, потом по порядку
    r1 = select_best_model(results, tolerance=0.03)
    r2 = select_best_model(results, tolerance=0.03)
    assert r1 == r2
