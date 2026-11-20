from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from src.config import PreprocessingConfig
from src.data.loader import load_data, normalize_columns
from src.features.cleaning import standardize_categories
from src.logging_config import get_logger
from src.models.persistence import load_model

logger = get_logger(__name__)


def prepare_inference_features(
    df: pd.DataFrame, config: PreprocessingConfig
) -> pd.DataFrame:
    """Подготовка признаков для inference — БЕЗ удаления строк.

    Отличия от clean_training_data:
    - нет dropna / drop_duplicates / фильтров по price, year, power;
    - строки сохраняются один-в-один, чтобы ответ был детерминирован.

    Унификация категорий (fillna, gasoline→petrol, drop служебных колонок)
    повторяет train — иначе OneHotEncoder не узнает значения.
    """
    if df.empty:
        raise ValueError("Входной DataFrame пуст")

    df = normalize_columns(df)
    df = standardize_categories(df, config)
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    df[cat_cols] = df[cat_cols].fillna("unknown")

    return df


def drop_target_if_present(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Отделяет target от признаков, если target пришёл (batch-валидация)."""
    if target_col in df.columns:
        return df.drop(columns=[target_col]), df[target_col]
    return df, None


def predict_batch(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Predict с сохранением исходного индекса — чтобы сматчить с входом."""
    predictions = model.predict(features)
    predictions = predictions.clip(min=0.0)
    return pd.Series(predictions, index=features.index, name="predicted_price")


def predict_from_file(
    input_path: str | Path,
    output_path: str | Path | None,
    model_path: str | Path,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    """End-to-end: CSV → предсказания → (опционально) CSV.

    Если в input есть колонка target (price), она попадёт в результат
    как `actual_price` — удобно для batch-валидации модели в проде.
    """
    cfg = config or PreprocessingConfig()

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {model_file}. Сначала запустите scripts/train.py"
        )

    model = load_model(model_file)
    raw_df = load_data(input_path)
    features = prepare_inference_features(raw_df, cfg)
    features, actual = drop_target_if_present(features, cfg.target_col)

    predictions = predict_batch(model, features)

    result = pd.DataFrame(index=features.index)
    if actual is not None:
        result["actual_price"] = actual
    result["predicted_price"] = predictions

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        logger.info("Predictions saved: %s (%d rows)", out, len(result))

    return result
