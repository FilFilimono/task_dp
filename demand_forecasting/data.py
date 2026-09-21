from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

KEY = ["sku", "location", "period"]
SERIES = ["sku", "location"]


def load_raw(data_dir):
    """Читает исходные CSV и приводит period к месячному pd.Period."""
    d = Path(data_dir)
    raw = {
        "sales": pd.read_csv(d / "sales_history.csv"),
        "stock": pd.read_csv(d / "stock_history.csv"),
        "promo": pd.read_csv(d / "promo_calendar.csv"),
        "items": pd.read_csv(d / "item_master.csv"),
    }
    for name in ("sales", "stock", "promo"):
        raw[name]["period"] = pd.PeriodIndex(raw[name]["period"], freq="M")
    return raw


def _issue(row, issue, action, old=None, new=None):
    """Собирает одну запись журнала дефектов: где нашли, что именно и что сделали."""
    return {
        "sku": row["sku"],
        "location": row.get("location", ""),
        "period": str(row["period"]),
        "issue": issue,
        "action": action,
        "old": old,
        "new": new,
    }


def _mask(df, idx, flag):
    """Заменяет qty на NaN в строках idx и пишет причину в flag."""
    df = df.copy()
    df.loc[idx, "qty"] = np.nan
    df.loc[idx, "flag"] = df.loc[idx, "flag"].where(df.loc[idx, "flag"] != "", flag)
    return df


def drop_duplicates(df):
    """Удаляет полные дубли; дубли ключа с разным qty маскирует в NaN, а не суммирует."""
    issues = []
    exact = df.duplicated(KEY + ["qty"], keep="first")
    issues += [_issue(r, "exact_duplicate", "dropped", r["qty"]) for _, r in df[exact].iterrows()]
    df = df[~exact].copy()

    conflict = df.duplicated(KEY, keep=False)
    if conflict.any():
        firsts = df[conflict].drop_duplicates(KEY, keep="first")
        issues += [_issue(r, "conflicting_duplicate", "masked") for _, r in firsts.iterrows()]
        df = df[~df.duplicated(KEY, keep="first")].copy()
        bad = df.set_index(KEY).index.isin(firsts.set_index(KEY).index)
        df.loc[bad, "qty"] = np.nan

    df["flag"] = np.where(df["qty"].isna(), "conflicting_duplicate", "")
    return df.reset_index(drop=True), issues


def mask_negative(df):
    """Маскирует отрицательные продажи в NaN: ноль дал бы ложный провал."""
    neg = df["qty"] < 0
    issues = [_issue(r, "negative_qty", "masked", r["qty"]) for _, r in df[neg].iterrows()]
    return _mask(df, neg, "negative"), issues


def reindex_calendar(df, end):
    """Полный месячный календарь от первой продажи до end: пропуски — NaN, а не ноль.

    Ряд без продаж за последние INACTIVE_MONTHS помечается active=False.
    """
    issues, parts = [], []
    for (sku, loc), g in df.groupby(SERIES, sort=True):
        last_obs = g["period"].max()
        full = pd.period_range(g["period"].min(), end, freq="M")
        g = g.set_index("period").reindex(full).rename_axis("period").reset_index()
        g["sku"], g["location"] = sku, loc
        g["active"] = (end - last_obs).n < C.INACTIVE_MONTHS

        gap = g["flag"].isna()
        g.loc[gap, "flag"] = "missing_row"
        if g["active"].iloc[0]:
            issues += [_issue(r, "missing_row", "masked") for _, r in g[gap].iterrows()]
        else:
            issues.append(_issue({"sku": sku, "location": loc, "period": last_obs},
                                 "inactive_series", "excluded"))
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    return out[KEY + ["qty", "flag", "active"]], issues


def detect_unit_shift(y):
    """Ищет смену единицы учёта: скачок медианы в UNIT_SHIFT_MIN_RATIO раз без пересечения уровней.

    Возвращает (индекс сдвига, коэффициент как степень 10) или None.
    """
    y = np.asarray(y, dtype=float)
    best = None
    for t in range(C.UNIT_SHIFT_MIN_RUN, len(y) - C.UNIT_SHIFT_MIN_RUN + 1):
        before, after = y[:t], y[t:]
        before, after = before[before > 0], after[after > 0]
        if len(before) < C.UNIT_SHIFT_MIN_RUN or len(after) < C.UNIT_SHIFT_MIN_RUN:
            continue
        ratio = np.median(after) / np.median(before)
        separated = after.min() > before.max() or after.max() < before.min()
        strength = abs(np.log10(ratio))
        if separated and strength >= np.log10(C.UNIT_SHIFT_MIN_RATIO):
            if best is None or strength > best[2]:
                best = (t, 10.0 ** round(np.log10(ratio)), strength)
    return None if best is None else (best[0], best[1])


def fix_unit_shift(df):
    """Приводит историю до сдвига к текущей единице учёта, умножая её на найденный коэффициент."""
    issues, parts = [], []
    for (sku, loc), g in df.groupby(SERIES, sort=True):
        g = g.sort_values("period").copy()
        found = detect_unit_shift(g["qty"].to_numpy())
        if found:
            t, factor = found
            g.iloc[:t, g.columns.get_loc("qty")] *= factor
            issues.append(_issue({"sku": sku, "location": loc, "period": g["period"].iloc[t]},
                                 "unit_shift", f"history_before_x{factor:g}"))
        parts.append(g)
    return pd.concat(parts, ignore_index=True), issues


def mask_stockouts(df, stock):
    """Маскирует месяцы с out-of-stock: при дефиците продажи не равны спросу."""
    m = df.merge(stock[KEY + ["days_out_of_stock"]], on=KEY, how="left")
    oos = (m["days_out_of_stock"].fillna(0) >= C.OOS_MIN_DAYS) & m["qty"].notna()
    issues = [_issue(r, "stockout", "masked", r["qty"]) for _, r in m[oos].iterrows()]
    return _mask(df, oos.to_numpy(), "stockout"), issues


def mask_promo(df, promo):
    """Маскирует промо-месяцы по sku на всех локациях: прогнозируем регулярный спрос."""
    keys = set(zip(promo["sku"], promo["period"]))
    in_promo = np.array([(s, p) in keys for s, p in zip(df["sku"], df["period"])])
    in_promo &= df["qty"].notna().to_numpy()
    issues = [_issue(r, "promo", "masked", r["qty"]) for _, r in df[in_promo].iterrows()]
    return _mask(df, in_promo, "promo"), issues


def impute_series(y):
    """Заполняет NaN: уровень за 12 месяцев × сезонный индекс, при короткой истории — линейно."""
    y = np.asarray(y, dtype=float).copy()
    nan = np.isnan(y)
    if not nan.any():
        return y

    s = pd.Series(y)
    linear = s.interpolate(limit_direction="both").to_numpy()
    if (~nan).sum() < C.MIN_OBS_SEASONAL_IMPUTE:
        return linear

    level = s.rolling(C.SEASON, center=True, min_periods=C.SEASON // 2).mean()
    level = level.interpolate(limit_direction="both").to_numpy()
    ratio = y / np.where(level > 0, level, np.nan)
    pos = np.arange(len(y)) % C.SEASON
    idx = np.ones(C.SEASON)
    for k in range(C.SEASON):
        r = ratio[pos == k]
        if np.isfinite(r).any():
            idx[k] = np.nanmedian(r)
    idx = idx / idx.mean()

    y[nan] = (level * idx[pos])[nan]
    still_nan = np.isnan(y)
    y[still_nan] = linear[still_nan]
    return np.maximum(y, 0.0)


def impute(df):
    """Заполняет пропуски по рядам; observed=False у заполненных точек, в оценку они не идут."""
    issues, parts = [], []
    for _, g in df.groupby(SERIES, sort=True):
        g = g.sort_values("period").copy()
        was_nan = g["qty"].isna().to_numpy()
        g["qty"] = impute_series(g["qty"].to_numpy())
        g["observed"] = ~was_nan
        issues += [_issue(r, "imputed", f"filled({r['flag']})", None, round(r["qty"], 1))
                   for _, r in g[was_nan].iterrows()]
        parts.append(g)
    return pd.concat(parts, ignore_index=True), issues


def clean(raw):
    """Полная очистка истории. Возвращает (очищенная история, журнал дефектов)."""
    sales = raw["sales"][KEY + ["qty"]].copy()
    sales["qty"] = sales["qty"].astype(float)
    end = sales["period"].max()
    log = []

    df, issues = drop_duplicates(sales)
    log += issues
    df, issues = mask_negative(df)
    log += issues
    df, issues = reindex_calendar(df, end)
    log += issues
    df = df[df["active"]].drop(columns="active")
    df, issues = fix_unit_shift(df)
    log += issues
    df, issues = mask_stockouts(df, raw["stock"])
    log += issues
    df, issues = mask_promo(df, raw["promo"])
    log += issues
    df, issues = impute(df)
    log += issues
    return df, pd.DataFrame(log)
