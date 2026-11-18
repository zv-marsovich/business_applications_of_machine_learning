import pandas as pd
from src.config import PreprocessingConfig
from src.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "model",
    "gearbox",
    "registration_year",
    "power",
    "price",
)


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Структурная валидация до очистки: колонки и непустота."""
    if df.empty:
        raise ValueError("Входной DataFrame пуст")
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
    logger.info("Схема сырых данных валидна (%d колонок)", df.shape[1])


def validate_clean_data(df: pd.DataFrame, config: PreprocessingConfig) -> None:
    """Инварианты после очистки — должны гарантированно выполняться."""
    if df.empty:
        raise ValueError("После очистки не осталось строк")
    if (df[config.target_col] <= 0).any():
        raise ValueError(f"В '{config.target_col}' остались неположительные значения")
    if not df["registration_year"].between(config.min_year, config.max_year - 1).all():
        raise ValueError(
            "Остались строки с registration_year вне допустимого диапазона"
        )
    logger.info("Инварианты очищенных данных выполняются (%d строк)", len(df))
