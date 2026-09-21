import numpy as np
import pytest

from demand_forecasting import metrics as M
from demand_forecasting.models import REGISTRY, classify, forecast
from demand_forecasting.pipeline import backtest_series


def test_seasonal_naive_repeats_last_year():
    y = np.arange(24, dtype=float)
    assert list(forecast("seasonal_naive", y, 3)) == [12, 13, 14]


def test_seasonal_ses_recovers_pure_seasonality():
    season = np.array([1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1], float) * 10
    assert forecast("seasonal_ses", np.tile(season, 3), 3) == pytest.approx(season[:3], rel=0.02)


def test_all_methods_return_h_nonnegative_finite():
    y = np.abs(np.random.default_rng(0).normal(50, 20, 36))
    for name in REGISTRY:
        out = forecast(name, y, 3)
        assert out.shape == (3,) and np.isfinite(out).all() and (out >= 0).all()


def test_croston_constant_demand_every_second_month():
    y = np.tile([0.0, 10.0], 18)
    assert forecast("croston_sba", y, 1)[0] == pytest.approx(0.95 * 10 / 2)


def test_bias_sign_and_wape():
    assert M.bias([100, 100], [110, 110]) == pytest.approx(0.10)
    assert M.wape([100, 0], [90, 10]) == pytest.approx(0.20)


def test_classify():
    rng = np.random.default_rng(1)
    assert classify(np.ones(10)) == "short"
    assert classify(np.tile([0, 0, 0, 5.0], 9)) == "intermittent"
    assert classify(np.tile([1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1.0], 3) * 10 + rng.normal(0, 1, 36)) == "seasonal"
    assert classify(100 + rng.normal(0, 5, 36)) == "regular"


def test_backtest_has_no_lookahead_and_skips_imputed():
    y = np.arange(30, dtype=float)
    observed = np.ones(30, bool)
    observed[27] = False
    bt = backtest_series(y, observed, ["seasonal_naive"], h=3, min_train=24)
    assert 27 not in set(bt["origin"] + bt["step"] - 1)
    assert (bt["yhat"] == bt["y"] - 12).all()
