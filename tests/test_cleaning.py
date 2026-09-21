import numpy as np
import pandas as pd
import pytest

from demand_forecasting import data as cl


def P(s):
    return pd.Period(s, freq="M")


def make(rows):
    df = pd.DataFrame(rows, columns=["sku", "location", "period", "qty"])
    df["period"] = pd.PeriodIndex(df["period"], freq="M")
    df["qty"] = df["qty"].astype(float)
    return df


def series(values, start="2023-01", sku="A", loc="MSK"):
    periods = pd.period_range(start, periods=len(values), freq="M")
    return make([(sku, loc, str(p), v) for p, v in zip(periods, values)])


def flagged(df):
    df = df.copy()
    df["flag"] = ""
    return df


def test_exact_duplicate_dropped_not_summed():
    df, issues = cl.drop_duplicates(make([("A", "MSK", "2023-01", 10), ("A", "MSK", "2023-01", 10)]))
    assert len(df) == 1 and df["qty"].iloc[0] == 10
    assert issues[0]["issue"] == "exact_duplicate"


def test_conflicting_duplicate_is_masked_not_guessed():
    df, issues = cl.drop_duplicates(make([("A", "MSK", "2023-01", 10), ("A", "MSK", "2023-01", 25)]))
    assert len(df) == 1 and np.isnan(df["qty"].iloc[0])
    assert df["flag"].iloc[0] == "conflicting_duplicate"


def test_negative_becomes_nan_not_zero():
    df, issues = cl.mask_negative(flagged(make([("A", "MSK", "2023-01", -8), ("A", "MSK", "2023-02", 5)])))
    assert np.isnan(df["qty"].iloc[0]) and df["qty"].iloc[1] == 5
    assert len(issues) == 1


def test_missing_month_is_nan_not_zero():
    df = flagged(make([("A", "MSK", "2023-01", 5), ("A", "MSK", "2023-03", 7)]))
    out, issues = cl.reindex_calendar(df, P("2023-03"))
    assert list(out["period"].astype(str)) == ["2023-01", "2023-02", "2023-03"]
    assert np.isnan(out["qty"].iloc[1]) and out["flag"].iloc[1] == "missing_row"


def test_leading_months_before_launch_not_created():
    out, _ = cl.reindex_calendar(flagged(make([("A", "MSK", "2023-06", 5)])), P("2023-08"))
    assert out["period"].min() == P("2023-06")


def test_discontinued_series_marked_inactive():
    out, issues = cl.reindex_calendar(flagged(make([("A", "MSK", "2023-01", 5)])), P("2023-12"))
    assert not out["active"].any()
    assert issues[0]["issue"] == "inactive_series"


def test_unit_shift_detected_and_rescaled():
    y = [200, 250, 220, 240, 210, 230] + [230000, 250000, 210000, 240000]
    df, issues = cl.fix_unit_shift(flagged(series(y)))
    assert issues[0]["issue"] == "unit_shift"
    assert df["qty"].iloc[0] == 200 * 1000 and df["qty"].iloc[-1] == 240000


def test_promo_spike_is_not_unit_shift():
    y = [200, 250, 220, 900, 210, 230, 240, 250]
    assert cl.detect_unit_shift(np.array(y, float)) is None


def test_intermittent_series_is_not_unit_shift():
    y = [0, 0, 5, 0, 0, 0, 40, 0, 0, 6, 0, 0]
    assert cl.detect_unit_shift(np.array(y, float)) is None


def test_stockout_zero_is_masked_but_true_zero_stays():
    df = flagged(series([10, 0, 0, 12]))
    stock = df[["sku", "location", "period"]].copy()
    stock["days_out_of_stock"] = [0, 30, 0, 0]
    out, issues = cl.mask_stockouts(df, stock)
    assert np.isnan(out["qty"].iloc[1]) and out["flag"].iloc[1] == "stockout"
    assert out["qty"].iloc[2] == 0
    assert len(issues) == 1


def test_promo_masked_on_all_locations():
    df = flagged(pd.concat([series([10, 50, 11]), series([5, 25, 6], loc="EKB")], ignore_index=True))
    promo = pd.DataFrame({"sku": ["A"], "period": pd.PeriodIndex(["2023-02"], freq="M")})
    out, issues = cl.mask_promo(df, promo)
    assert out["qty"].isna().sum() == 2 and len(issues) == 2


def test_impute_no_nan_unchanged():
    y = np.arange(30, dtype=float)
    assert np.array_equal(cl.impute_series(y), y)


def test_impute_short_series_is_linear():
    assert cl.impute_series(np.array([10, np.nan, 30.0]))[1] == pytest.approx(20)


def test_impute_preserves_seasonal_peak():
    season = np.array([1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1], float) * 100
    y = np.tile(season, 3)
    y[17] = np.nan
    filled = cl.impute_series(y)[17]
    assert filled > 200


def test_impute_never_negative_and_no_nan_left():
    y = np.tile(np.array([5, 1, 0, 2, 8, 3, 1, 0, 4, 6, 2, 1], float), 3)
    y[[0, 20, 35]] = np.nan
    out = cl.impute_series(y)
    assert np.isfinite(out).all() and (out >= 0).all()


def test_clean_pipeline_end_to_end():
    vals = list(100 + 10 * np.sin(np.arange(36)))
    sales = series(vals)
    sales = pd.concat([sales, sales.iloc[[3]]], ignore_index=True)
    sales.loc[5, "qty"] = -4
    sales = sales.drop(index=10).reset_index(drop=True)
    sales.loc[sales["period"] == P("2024-06"), "qty"] = 0
    sales.loc[sales["period"] == P("2024-09"), "qty"] = 400
    stock = sales[["sku", "location", "period"]].drop_duplicates().copy()
    stock["days_out_of_stock"] = np.where(stock["period"] == P("2024-06"), 30, 0)
    promo = pd.DataFrame({"sku": ["A"], "period": pd.PeriodIndex(["2024-09"], freq="M")})

    out, log = cl.clean({"sales": sales, "stock": stock, "promo": promo})
    assert len(out) == 36 and out["qty"].notna().all()
    assert (~out["observed"]).sum() == 4
    assert out["qty"].between(80, 120).all()
    assert {"exact_duplicate", "negative_qty", "missing_row", "stockout", "promo"} <= set(log["issue"])
