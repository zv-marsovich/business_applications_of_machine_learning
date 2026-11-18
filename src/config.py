from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    target_col: str = "price"
    min_year: int = 1920
    max_year: int = 2017
    max_power: int = 2301
    test_size: float = Field(0.2, gt=0, lt=1)
    random_state: int = 42
    drop_columns: list[str] = [
        "date_crawled",
        "registration_month",
        "date_created",
        "number_of_pictures",
        "postal_code",
        "last_seen",
    ]
    categorical_fill_values: dict[str, str] = {
        "fuel_type": "unknown",
        "vehicle_type": "unknown",
        "repaired": "yes",
    }


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    n_trials: int = 10
    cv_folds: int = 3
    random_state: int = 42
    model_output_path: Path = Path("models/best_model_pipeline.joblib")


class MLflowConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    experiment_name: str = "car_price_prediction_optuna"


class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    raw_data: Path = Path("data/raw/autos.csv")
    models_dir: Path = Path("models")
    mlruns_dir: Path = Path("mlruns")


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    project_root: Path = Path.cwd()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    training: TrainingConfig = TrainingConfig()
    mlflow: MLflowConfig = MLflowConfig()
    paths: PathsConfig = PathsConfig()

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else self.project_root / p


def load_settings(
    path: str | Path = "config.yaml",
    project_root: Path | None = None,
) -> Settings:
    root = project_root or Path.cwd()
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return Settings(project_root=root, **(data or {}))
