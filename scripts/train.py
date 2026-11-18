from pathlib import Path

from src.config import load_settings
from src.logging_config import setup_logging
from src.pipeline.train import train_pipeline


def main() -> None:
    setup_logging()
    root = Path(__file__).resolve().parent.parent
    settings = load_settings(root / "config.yaml", project_root=root)
    train_pipeline(settings)


if __name__ == "__main__":
    main()
