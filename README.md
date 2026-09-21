# Прогноз спроса на грязной истории

Прогноз на 2026-01 … 2026-03 по ключу `sku × location`. Python 3.10+, расчёт на `pandas` и `numpy`.

```bash
pip install -r requirements.txt
python -m demand_forecasting.pipeline
python -m pytest -q
```

Результат — `outputs/forecast.csv` (sku, location, period, forecast, method).

## Структура

- `demand_forecasting/data.py` — чтение и очистка истории
- `demand_forecasting/models.py` — сегментация рядов и методы прогноза
- `demand_forecasting/metrics.py` — WAPE, bias, MASE, RMSSE
- `demand_forecasting/pipeline.py` — backtest, выбор метода, расчёт прогноза
- `demand_forecasting/config.py` — пороги
- `notebooks/` — EDA, очистка по шагам, сравнение методов
- `tests/` — тесты очистки, методов и метрик
- `outputs/` — прогноз, журнал дефектов `data_issues.csv`, метрики backtest
- `REPORT.md` — отчёт

Новый метод добавляется функцией с `@register("name")` в `models.py` и именем в `CANDIDATES` в `pipeline.py`.
