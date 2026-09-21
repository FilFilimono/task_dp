import numpy as np
import pandas as pd

from . import config as C
from .data import SERIES

REGISTRY = {}


def register(name):
    """Декоратор: добавляет метод прогноза в REGISTRY под именем name."""
    def deco(f):
        REGISTRY[name] = f
        return f
    return deco


def _ses_level(y, alpha):
    """SES с параметром alpha. Возвращает (последний уровень, SSE одношаговых ошибок)."""
    level = float(np.mean(y[:min(3, len(y))]))
    sse = 0.0
    for v in y:
        sse += (v - level) ** 2
        level = alpha * v + (1 - alpha) * level
    return level, sse


def _best_ses(y):
    """Подбирает alpha по сетке 0.05..0.5 по минимуму SSE и возвращает сглаженный уровень."""
    fits = [_ses_level(y, a) for a in np.arange(0.05, 0.51, 0.05)]
    return min(fits, key=lambda f: f[1])[0]


def _seasonal_indices(y):
    """Мультипликативные сезонные индексы: медиана отношения ряда к центрированному MA-12."""
    n, m = len(y), C.SEASON
    kernel = np.r_[0.5, np.ones(m - 1), 0.5] / m
    level = np.full(n, np.nan)
    level[m // 2: n - m // 2] = np.convolve(y, kernel, mode="valid")
    ratio = y / np.where(level > 0, level, np.nan)
    pos = np.arange(n) % m
    idx = np.ones(m)
    for k in range(m):
        r = ratio[pos == k]
        if np.isfinite(r).any():
            idx[k] = np.nanmedian(r)
    return idx / idx.mean()


@register("seasonal_naive")
def seasonal_naive(y, h):
    """Baseline: значение того же месяца год назад. При истории короче года — последнее значение."""
    if len(y) < C.SEASON:
        return np.repeat(y[-1], h)
    return np.array([y[len(y) - C.SEASON + i % C.SEASON] for i in range(h)])


@register("seasonal_naive_growth")
def seasonal_naive_growth(y, h):
    """Seasonal naive, умноженный на рост среднего уровня год к году (в пределах 0.5..2)."""
    if len(y) < 2 * C.SEASON:
        return seasonal_naive(y, h)
    prev = y[-2 * C.SEASON:-C.SEASON].mean()
    growth = np.clip(y[-C.SEASON:].mean() / prev, 0.5, 2.0) if prev > 0 else 1.0
    return seasonal_naive(y, h) * growth


@register("ses")
def ses(y, h):
    """Плоский прогноз на уровне экспоненциального сглаживания."""
    return np.repeat(_best_ses(y), h)


@register("moving_average_6")
def moving_average_6(y, h):
    """Плоский прогноз: среднее за последние 6 месяцев."""
    return np.repeat(y[-6:].mean(), h)


@register("seasonal_ses")
def seasonal_ses(y, h):
    """Снимает сезонность индексами, сглаживает уровень SES и возвращает сезонность обратно."""
    if len(y) < 2 * C.SEASON:
        return ses(y, h)
    idx = _seasonal_indices(y)
    pos = np.arange(len(y)) % C.SEASON
    level = _best_ses(y / idx[pos])
    future_pos = (len(y) + np.arange(h)) % C.SEASON
    return level * idx[future_pos]


@register("mean_12")
def mean_12(y, h):
    """Плоский прогноз: среднее за последние 12 месяцев, включая нули."""
    return np.repeat(y[-C.SEASON:].mean(), h)


@register("croston_sba")
def croston_sba(y, h, alpha=0.1):
    """Кростон с поправкой SBA: сглаживает размер продажи и интервал, убирает завышение."""
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return np.zeros(h)
    z = y[nz[0]]
    p = float(nz[0] + 1)
    q = 1
    for v in y[nz[0] + 1:]:
        if v > 0:
            z = alpha * v + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1
    return np.repeat((1 - alpha / 2) * z / p, h)


@register("tsb")
def tsb(y, h, alpha=0.1, beta=0.1):
    """Метод Teunter-Syntetos-Babai: сглаживает вероятность продажи (каждый месяц) и её размер."""
    nz = y[y > 0]
    if len(nz) == 0:
        return np.zeros(h)
    z = nz[0]
    prob = float((y > 0).mean())
    for v in y:
        prob = beta * (v > 0) + (1 - beta) * prob
        if v > 0:
            z = alpha * v + (1 - alpha) * z
    return np.repeat(prob * z, h)


def forecast(name, y, h):
    """Считает прогноз методом name на h шагов, отрицательные значения обрезает."""
    out = REGISTRY[name](np.asarray(y, dtype=float), h)
    return np.maximum(np.asarray(out, dtype=float), 0.0)


def adi(y):
    """Средний интервал между продажами: число периодов / число периодов с ненулевым спросом."""
    nonzero = int((y > 0).sum())
    return np.inf if nonzero == 0 else len(y) / nonzero


def seasonal_acf(y):
    """Автокорреляция на лаге 12 после снятия линейного тренда."""
    t = np.arange(len(y))
    resid = y - np.polyval(np.polyfit(t, y, 1), t)
    denom = (resid ** 2).sum()
    if denom == 0:
        return 0.0
    return float((resid[C.SEASON:] * resid[:-C.SEASON]).sum() / denom)


def classify(y):
    """Относит ряд к сегменту: short, intermittent, seasonal или regular. Пороги заданы в config."""
    if len(y) < C.SHORT_HISTORY:
        return "short"
    if adi(y) >= C.ADI_THRESHOLD:
        return "intermittent"
    if seasonal_acf(y) >= C.SEASONAL_ACF_THRESHOLD:
        return "seasonal"
    return "regular"


def segment(clean):
    """Считает по каждому ряду длину, ADI, ACF12 и средний спрос и присваивает сегмент."""
    rows = []
    for (sku, loc), g in clean.groupby(SERIES, sort=True):
        y = g.sort_values("period")["qty"].to_numpy()
        rows.append({
            "sku": sku,
            "location": loc,
            "n": len(y),
            "adi": round(adi(y), 2),
            "acf12": round(seasonal_acf(y), 2) if len(y) >= C.SHORT_HISTORY else np.nan,
            "mean_qty": round(float(y.mean()), 1),
            "segment": classify(y),
        })
    return pd.DataFrame(rows)
