# Churn service

Небольшой FastAPI-сервис для предсказания оттока клиентов. Модель —
`LogisticRegression` или `RandomForestClassifier`, обучение запускается через API.

## Данные

`data/churn_dataset.csv` должен содержать колонки:

```text
monthly_fee, usage_hours, support_requests, account_age_months,
failed_payments, region, device_type, payment_method, autopay_enabled, churn
```

`churn` — целевая переменная: `0` или `1`.

## Локальный запуск

```bash
uv sync
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Документация: <http://localhost:8000/docs>

## Обучение

```bash
curl -X POST http://localhost:8000/model/train \
  -H 'Content-Type: application/json' \
  -d '{"model_type":"logreg","hyperparameters":{"max_iter":100}}'
```

Доступны модели `logreg` и `random_forest`.

## Предсказание

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "monthly_fee":49.9,
    "usage_hours":120.0,
    "support_requests":2,
    "account_age_months":18,
    "failed_payments":0,
    "region":"europe",
    "device_type":"mobile",
    "payment_method":"card",
    "autopay_enabled":1
  }'
```

## Podman

```bash
podman build --format docker -t churn-service .
podman run --rm --name churn-service -p 8000:8000 churn-service
```

Проверка состояния:

```bash
curl http://localhost:8000/health
podman logs -f churn-service
```
