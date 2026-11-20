from pathlib import Path

import mlflow
import psutil

from src.config import MLflowConfig


def init_mlflow(cfg: MLflowConfig, project_root: Path) -> None:
    """Инициализирует MLflow с database backend."""
    tracking_uri = cfg.tracking_uri or "sqlite:///mlflow.db"
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith(
        "sqlite:////"
    ):
        db_name = tracking_uri.replace("sqlite:///", "")
        tracking_uri = f"sqlite:///{(project_root / db_name).resolve()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)


def log_system_metrics() -> None:
    mlflow.log_metrics(
        {
            "system_cpu_percent": psutil.cpu_percent(interval=0.5),
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_disk_usage": psutil.disk_usage("/").percent,
        }
    )
