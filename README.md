# Прогнозирование цены автомобиля

## Описание проекта

Сервис по продаже автомобилей с пробегом «Не бит, не крашен» разрабатывает приложение, чтобы привлечь новых клиентов. В нём можно будет узнать рыночную стоимость своего автомобиля.

Нужно построить модель, которая умеет определять рыночную стоимость. В распоряжении есть данные о технических характеристиках, комплектации и ценах других автомобилей.

Критерии, которые важны заказчику:
- качество предсказания;
- время обучения модели;
- время предсказания модели.

## Данные

Датасет содержит информацию об автомобилях:
- **Целевая переменная:** `price` (цена автомобиля)
- **Признаки:** модель, тип топлива, тип кузова, коробка передач, год регистрации, мощность и др.

## Архитектура ML-пайплайна

### ETL процесс:

1. **Extract:** Загрузка данных из CSV
2. **Transform:** Обработка пропусков, удаление выбросов, кодирование категориальных признаков, масштабирование
3. **Load:** Сохранение предобработанных данных и модели

### Структура проекта

```
business_applications_of_machine_learning/
├── config.yaml                  # Все параметры: данные, модель, MLflow
├── pyproject.toml               # Зависимости и конфигурация инструментов
├── Dockerfile                   # Multi-stage сборка с uv
├── docker-compose.yml           # train и predict сервисы
├── .pre-commit-config.yaml      # ruff, mypy, trailing-whitespace
│
├── src/
│   ├── config.py                # Pydantic-модели конфигурации
│   ├── logging_config.py
│   ├── data/
│   │   ├── loader.py            # load_data, normalize_columns
│   │   └── validation.py        # validate_raw_schema, validate_clean_data
│   ├── features/
│   │   ├── cleaning.py          # standardize_categories, clean_training_data
│   │   ├── pipeline.py          # create_preprocessor
│   │   ├── schema.py            # get_features_target, get_columns_types
│   │   ├── split.py             # split_data
│   │   └── transformers.py      # PowerImputer (без утечки)
│   ├── models/
│   │   ├── metrics.py           # rmse, evaluate_predictions
│   │   ├── persistence.py       # save_model, load_model
│   │   ├── registry.py          # MODEL_REGISTRY (XGB/LGBM/CatBoost/LinReg)
│   │   └── trainer.py           # train_single_model
│   ├── optimization/
│   │   └── optimize.py          # make_objective, run_study
│   ├── pipeline/
│   │   ├── prepare.py           # prepare_training_dataset
│   │   ├── train.py             # train_pipeline (оркестратор)
│   │   └── predict.py           # predict_from_file (inference)
│   └── tracking/
│       └── mlflow_utils.py      # init_mlflow, log_system_metrics
│
├── scripts/
│   ├── train.py                 # Entry point: обучение
│   └── predict.py               # Entry point: предсказание
│
└── tests/                       # pytest + mypy --strict
```

## AutoML реализация

Использована библиотека **Optuna** (TPE Sampler) для байесовской оптимизации гиперпараметров. Каждая модель обучается с кросс-валидацией (`cv_folds=3`), метрика оптимизации — **CV RMSE**.

| Модель | Параметры для оптимизации |
|--------|--------------------------|
| XGBoost | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda |
| CatBoost | iterations, depth, learning_rate, l2_leaf_reg, border_count |
| LightGBM | n_estimators, max_depth, learning_rate, num_leaves, subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_samples |

## Как выбирается финальная модель

Выбор модели — **двухступенчатый**:

1. **Фильтр по качеству.** Модели, чей `CV RMSE` хуже лучшего более чем на `selection_tolerance` (по умолчанию 3%), отбрасываются.
2. **Выбор по времени.** Среди оставшихся побеждает самая быстрая по `train_time`.

Логика реализована в `src/models/selection.py`:

```
threshold = best_cv_rmse * (1 + selection_tolerance)
candidates = {model for model in results if model.cv_rmse <= threshold}
selected = min(candidates, key=train_time)
```

Порог `selection_tolerance` вынесен в `config.yaml` — стейкхолдер может его менять без правок в коде. При `tolerance=0` поведение как раньше: побеждает строго лучший по CV.

**Почему не по Test RMSE?** Test-выборка — это финальный отчёт, а не критерий выбора. Если выбирать по test, получим переобучение под конкретный отложенный набор (test set leakage). Поэтому `CV RMSE` — для выбора, `Test RMSE` — только для отчёта.|

## Результаты

**Данные:** 354 369 строк → после очистки 258 953 строки. Test split — 20%.
**Конфигурация:** `n_trials=10`, `cv_folds=3`, `selection_tolerance=0.03`, `random_state=42`.

| Модель | RMSE (CV) | RMSE (Test) | R² (Test) | Train time |
|--------|-----------|-------------|-----------|------------|
| LinearRegression | 2970.076 | 2975.812 | 0.591 | 1.9 s |
| XGBoost | 1688.322 | 1660.273 | 0.873 | 62.5 s |
| CatBoost | 1690.255 | 1656.561 | 0.873 | 253.4 s |
| **LightGBM** | **1704.501** | 1672.937 | 0.871 | **50.3 s** |

**Лучшая модель по CV:** XGBoost (1688.322).
**Выбрана селектором:** **LightGBM** — CV RMSE на 0.96% хуже, но в 1.24 раза быстрее. CatBoost и XGBoost попали в tolerance, но LightGBM оказался самым быстрым среди них.

> Качество в пределах 3% — это «небольшая разница» для бизнеса. При регулярном переобучении (раз в неделю) 50 s против 253 s дают существенную экономию compute и ускоряют цикл обновления модели.

## Итоговое заключение

Разработанный пайплайн (LightGBM + PowerImputer + OneHotEncoder) готов к внедрению в сервис «Не бит, не крашен». Модель оценивает рыночную стоимость автомобиля с ошибкой около 1670 евро (R² = 0.87) и обучается за ~50 секунд, что позволяет переобучать её еженедельно на свежих данных без заметных затрат.

Модель предсказывает цену за доли секунды и может обрабатывать тысячи запросов одновременно.

Рекомендуется: запустить модель в тестовом режиме на 10% трафика, собрать обратную связь и через месяц дообучить на новых данных. При положительных результатах — масштабировать на весь сервис.

---

## Требования

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) — менеджер пакетов и окружений
- Docker + Docker Compose (для запуска в контейнере)

## Установка

```bash
git clone https://github.com/zv-marsovich/business_applications_of_machine_learning.git
cd business_applications_of_machine_learning
uv sync
```

> Зависимости управляются через `uv` (см. `pyproject.toml` и `uv.lock`).
> `requirements.txt` не используется — для legacy-инструментов сгенерируйте
> его через `uv export --format requirements-txt`.
>
Все параметры (пути, гиперпараметры, MLflow-эксперимент) хранятся в `config.yaml` и валидируются через Pydantic при старте. Дефолтные значения — в `src/config.py`.

## Обучение

```bash
uv run python scripts/train.py
```

Что происходит:

1. Загрузка и валидация данных из `data/autos.csv`
2. Очистка (выбросы, дубликаты), унификация категорий
3. Train/test split (стратифицированно по `config.yaml`)
4. Optuna подбирает гиперпараметры для XGBoost, LightGBM, CatBoost
5. Метрики и артефакты логируются в MLflow
6. Лучший пайплайн сохраняется в `models/best_model_pipeline.joblib`

## Предсказание

```bash
uv run python scripts/predict.py \
    --input data/autos.csv \
    --output preds.csv
```

Если в input есть колонка `price` — в результат попадёт `actual_price` для batch-валидации. Если нет — только `predicted_price`. Строки **не удаляются**: сколько пришло — столько и вернётся.

## Мониторинг

```bash
uv run mlflow ui
```

- **MLflow:** логирование метрик, параметров и артефактов
- **Системные метрики:** CPU, RAM, диск (psutil)
- **Время обучения:** замеряется для каждой модели

![Диаграммы метрик в MLflow](mlflow_diagrams/diagram1.png)
![Диаграммы метрик в MLflow](mlflow_diagrams/diagram2.png)
![Диаграммы метрик в MLflow](mlflow_diagrams/diagram3.png)

## Docker

Сборка:

```bash
docker build -t car-price-ml .
```

Обучение:

```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/mlruns:/app/mlruns \
    car-price-ml
```

Предсказание:

```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    car-price-ml scripts.predict --input /app/data/autos.csv --output /app/preds.csv
```

Или через `docker-compose`:

```bash
docker compose run --rm train
docker compose run --rm predict
```

## Разработка

```bash
# Линтер и форматтер
uv run ruff check src tests
uv run ruff format src tests

# Проверка типов
uv run python -m mypy src tests

# Тесты
uv run pytest tests/ -v

# Все проверки разом
uv run pre-commit run --all-files
```

CI (GitHub Actions) прогоняет эти же команды на каждый push в `main` и на PR. Конфигурация — в `.github/workflows/ci.yml`.
