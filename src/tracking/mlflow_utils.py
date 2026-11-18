from pathlib import Path

import mlflow
import psutil

from src.config import MLflowConfig


def init_mlflow(cfg: MLflowConfig, mlruns_dir: Path) -> None:
    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment(cfg.experiment_name)


def log_system_metrics() -> None:
    mlflow.log_metrics(
        {
            "system_cpu_percent": psutil.cpu_percent(interval=0.5),
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_disk_usage": psutil.disk_usage("/").percent,
        }
    )
