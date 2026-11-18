import argparse
from pathlib import Path

from src.config import load_settings
from src.logging_config import setup_logging
from src.pipeline.predict import predict_from_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict car prices from CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file (optional)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to .joblib (default: from config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    root = Path(__file__).resolve().parent.parent
    settings = load_settings(root / "config.yaml", project_root=root)

    model_path = args.model or settings.resolve(settings.training.model_output_path)
    result = predict_from_file(
        input_path=args.input,
        output_path=args.output,
        model_path=model_path,
        config=settings.preprocessing,
    )

    print(f"Rows: {len(result)}")
    print(
        f"Predicted price — "
        f"min: {result['predicted_price'].min():.0f}, "
        f"mean: {result['predicted_price'].mean():.0f}, "
        f"max: {result['predicted_price'].max():.0f}"
    )


if __name__ == "__main__":
    main()
