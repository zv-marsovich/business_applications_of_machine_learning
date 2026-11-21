from src.logging_config import get_logger
from src.models.trainer import ModelResult

logger = get_logger(__name__)


def select_best_model(
    results: dict[str, ModelResult],
    tolerance: float = 0.03,
) -> str:
    """Выбирает модель с учётом качества и времени обучения.

    1. Находит лучший CV RMSE.
    2. Отбирает модели, чей CV RMSE не хуже лучшего более чем на `tolerance`.
    3. Среди отобранных выбирает самую быструю по train_time.

    Args:
        results: {model_name: ModelResult}
        tolerance: доля (0.03 = 3%). 0 — только качество, 0.5 — половина.

    Returns:
        Имя выбранной модели (ключ в results).
    """
    if not results:
        raise ValueError("results is empty")

    best_by_rmse = min(results, key=lambda n: results[n].cv_rmse)
    best_rmse = results[best_by_rmse].cv_rmse
    threshold = best_rmse * (1 + tolerance)

    candidates = {name: r for name, r in results.items() if r.cv_rmse <= threshold}
    selected_name = min(candidates, key=lambda n: candidates[n].train_time)
    selected = results[selected_name]

    delta_pct = (selected.cv_rmse - best_rmse) / best_rmse * 100
    speedup = (
        results[best_by_rmse].train_time / selected.train_time
        if selected.train_time > 0
        else float("inf")
    )

    logger.info(
        "Selection: best CV RMSE = %.2f (%s). Candidates within %.1f%%: %s",
        best_rmse,
        best_by_rmse,
        tolerance * 100,
        sorted(candidates),
    )
    logger.info(
        "Selected %s — CV RMSE %.2f (+%.2f%%), train_time %.2fs (%.2fx faster than %s)",
        selected_name,
        selected.cv_rmse,
        delta_pct,
        selected.train_time,
        speedup,
        best_by_rmse,
    )
    return selected_name
