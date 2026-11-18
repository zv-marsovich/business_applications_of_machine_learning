import re
from pathlib import Path

import pandas as pd
from src.logging_config import get_logger

logger = get_logger(__name__)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CamelCase → snake_case. Идемпотентна, безопасна для inference."""
    return df.rename(columns=lambda x: re.sub(r"(?<!^)([A-Z])", r"_\1", x).lower())


def load_data(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл с данными не найден: {path}")

    df = pd.read_csv(path)
    df = normalize_columns(df)
    logger.info("Загружено %d строк и %d столбцов из %s", *df.shape, path)
    return df
