import pandas as pd
import pytest
from src.config import PreprocessingConfig
from src.data.validation import validate_clean_data, validate_raw_schema


def test_validate_raw_schema_ok() -> None:
    df = pd.DataFrame(
        {
            "model": ("golf", "astra"),
            "gearbox": ("manual", "auto"),
            "registration_year": (2010, 2012),
            "power": (100, 120),
            "price": (5000, 6000),
        }
    )
    validate_raw_schema(df)  # не должно бросить


def test_validate_raw_schema_empty() -> None:
    with pytest.raises(ValueError, match="пуст"):
        validate_raw_schema(pd.DataFrame())


@pytest.mark.parametrize(
    "missing_col", ["model", "gearbox", "price", "power", "registration_year"]
)
def test_validate_raw_schema_missing_column(missing_col: str) -> None:
    df = pd.DataFrame(
        {
            "model": ("golf",),
            "gearbox": ("manual",),
            "registration_year": (2010,),
            "power": (100,),
            "price": (5000,),
        }
    ).drop(columns=[missing_col])

    with pytest.raises(ValueError, match="Отсутствуют обязательные колонки"):
        validate_raw_schema(df)


def test_validate_clean_data_ok() -> None:
    cfg = PreprocessingConfig()
    df = pd.DataFrame(
        {
            "price": (5000, 6000),
            "registration_year": (2010, 2012),
        }
    )
    validate_clean_data(df, cfg)


def test_validate_clean_data_empty() -> None:
    cfg = PreprocessingConfig()
    with pytest.raises(ValueError, match="не осталось строк"):
        validate_clean_data(pd.DataFrame(columns=["price", "registration_year"]), cfg)


def test_validate_clean_data_non_positive_price() -> None:
    cfg = PreprocessingConfig()
    df = pd.DataFrame({"price": (0, 5000), "registration_year": (2010, 2012)})
    with pytest.raises(ValueError, match="неположительные"):
        validate_clean_data(df, cfg)


def test_validate_clean_data_year_out_of_range() -> None:
    cfg = PreprocessingConfig(min_year=1920, max_year=2017)
    df = pd.DataFrame({"price": (5000,), "registration_year": (1900,)})
    with pytest.raises(ValueError, match="registration_year"):
        validate_clean_data(df, cfg)
