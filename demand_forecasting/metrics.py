import numpy as np


def wape(y, yhat):
    """Взвешенная абсолютная ошибка: сумма |прогноз - факт| / сумма факта. Определена при нулях в факте."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    total = np.abs(y).sum()
    return float(np.abs(yhat - y).sum() / total) if total > 0 else np.nan


def bias(y, yhat):
    """Смещение: сумма (прогноз - факт) / сумма факта. Плюс означает завышение, минус — занижение."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    total = np.abs(y).sum()
    return float((yhat - y).sum() / total) if total > 0 else np.nan


def naive_scale(train, lag):
    """Знаменатель MASE: MAE наивного прогноза с лагом lag на обучающей части."""
    train = np.asarray(train, float)
    if len(train) <= lag:
        lag = 1
    scale = np.abs(train[lag:] - train[:-lag]).mean()
    if not np.isfinite(scale) or scale == 0:
        scale = max(np.abs(train).mean(), 1.0)
    return float(scale)


def naive_scale_sq(train, lag=1):
    """Знаменатель RMSSE: средний квадрат ошибки наивного прогноза с лагом lag на обучающей части."""
    train = np.asarray(train, float)
    scale = ((train[lag:] - train[:-lag]) ** 2).mean() if len(train) > lag else 0.0
    return float(scale) if np.isfinite(scale) and scale > 0 else 1.0


def mase(y, yhat, scale):
    """Средняя абсолютная ошибка, делённая на scale. Меньше 1 — лучше наивного прогноза."""
    return float(np.abs(np.asarray(yhat, float) - np.asarray(y, float)).mean() / scale)


def rmsse(y, yhat, scale_sq):
    """Корень из среднего квадрата ошибки, делённого на scale_sq. Используется для редкого спроса."""
    e = np.asarray(yhat, float) - np.asarray(y, float)
    return float(np.sqrt((e ** 2).mean() / scale_sq))
