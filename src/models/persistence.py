from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.logging_config import get_logger

logger = get_logger(__name__)


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
