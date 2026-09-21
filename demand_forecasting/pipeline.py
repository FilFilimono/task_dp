import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .data import SERIES, clean, load_raw
from .models import forecast, segment

CANDIDATES = {
    "seasonal": ["seasonal_naive", "seasonal_naive_growth", "seasonal_ses", "ses"],
    "regular": ["seasonal_naive", "ses", "moving_average_6", "seasonal_ses"],
    "intermittent": ["seasonal_naive", "mean_12", "croston_sba", "tsb"],
    "short": [],
}
FIXED = {"short": "ses"}
SELECT_BY = {"seasonal": "mase", "regular": "mase", "intermittent": "rmsse"}
SCALE_LAG = {"seasonal": C.SEASON, "regular": C.SEASON, "intermittent": 1}


def backtest_series(y, observed, methods, h=C.HORIZON, min_train=C.MIN_TRAIN):
    """Rolling-origin backtest одного ряда; точки с observed=False не оцениваются."""
    rows = []
    for t in range(min_train, len(y) - h + 1):
        train, test, obs = y[:t], y[t:t + h], observed[t:t + h]
        for m in methods:
            yhat = forecast(m, train, h)
            for step in range(h):
                if obs[step]:
                    rows.append({"method": m, "origin": t, "step": step + 1,
                                 "y": test[step], "yhat": yhat[step]})
    return pd.DataFrame(rows)


def backtest(history, segments, candidates):
    """Backtest по всем рядам с кандидатами своего сегмента; короткие ряды пропускаются."""
    seg = segments.set_index(SERIES)["segment"].to_dict()
    parts = []
    for (sku, loc), g in history.groupby(SERIES, sort=True):
        methods = candidates.get(seg[(sku, loc)], [])
        g = g.sort_values("period")
        if not methods or len(g) < C.MIN_TRAIN + C.HORIZON:
            continue
        bt = backtest_series(g["qty"].to_numpy(), g["observed"].to_numpy(), methods)
        bt["sku"], bt["location"], bt["segment"] = sku, loc, seg[(sku, loc)]
        parts.append(bt)
    return pd.concat(parts, ignore_index=True)


def score(bt, history):
    """Метрики по каждой паре (ряд, метод): MASE, RMSSE, WAPE, bias."""
    series = {k: g.sort_values("period")["qty"].to_numpy() for k, g in history.groupby(SERIES)}
    rows = []
    for (seg, sku, loc, m), g in bt.groupby(["segment", "sku", "location", "method"]):
        y_full = series[(sku, loc)]
        lag = SCALE_LAG[seg]
        fold_mase, fold_rmsse = [], []
        for origin, f in g.groupby("origin"):
            train = y_full[:origin]
            fold_mase.append(M.mase(f["y"], f["yhat"], M.naive_scale(train, lag)))
            fold_rmsse.append(M.rmsse(f["y"], f["yhat"], M.naive_scale_sq(train, lag)))
        err = g["yhat"] - g["y"]
        rows.append({
            "segment": seg, "sku": sku, "location": loc, "method": m,
            "mase": float(np.mean(fold_mase)),
            "rmsse": float(np.mean(fold_rmsse)),
            "wape": M.wape(g["y"], g["yhat"]),
            "bias": M.bias(g["y"], g["yhat"]),
            "abs_err": float(err.abs().sum()),
            "err": float(err.sum()),
            "vol": float(g["y"].sum()),
        })
    return pd.DataFrame(rows)


def summarize(scores):
    """Сводит метрики до (сегмент, метод): MASE/RMSSE — среднее по рядам, WAPE/bias — по объёму."""
    out = scores.groupby(["segment", "method"]).agg(
        n_series=("sku", "size"),
        mase=("mase", "mean"),
        rmsse=("rmsse", "mean"),
        abs_err=("abs_err", "sum"),
        err=("err", "sum"),
        vol=("vol", "sum"),
    ).reset_index()
    out["wape"] = out["abs_err"] / out["vol"]
    out["bias"] = out["err"] / out["vol"]
    return out.drop(columns=["abs_err", "err", "vol"])


def select(summary):
    """Выбирает метод сегмента: лучше baseline на MIN_IMPROVEMENT и не хуже по |bias|, иначе baseline."""
    chosen = dict(FIXED)
    for seg, g in summary.groupby("segment"):
        g = g.set_index("method")
        metric = SELECT_BY[seg]
        base = g.loc[C.BASELINE]
        bias_cap = max(C.BIAS_TOLERANCE, abs(base["bias"]))
        better = g[metric] <= base[metric] * (1 - C.MIN_IMPROVEMENT)
        ok = g[better & (g["bias"].abs() <= bias_cap)]
        chosen[seg] = ok[metric].idxmin() if len(ok) else C.BASELINE
    return chosen


def main(data_dir="data", out_dir="outputs"):
    """Полный расчёт: очистка, сегментация, backtest, выбор методов, прогноз. Пишет в out_dir."""
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    history, issues = clean(load_raw(data_dir))
    issues.to_csv(out / "data_issues.csv", index=False)
    history.assign(period=history["period"].astype(str)).to_csv(out / "clean_history.csv", index=False)

    segments = segment(history)
    bt = backtest(history, segments, CANDIDATES)
    scores = score(bt, history)
    summary = summarize(scores)
    chosen = select(summary)

    summary["selected"] = [chosen[s] == m for s, m in zip(summary["segment"], summary["method"])]
    summary.round(3).to_csv(out / "segment_summary.csv", index=False)
    scores.round(3).to_csv(out / "series_scores.csv", index=False)
    segments["method"] = segments["segment"].map(chosen)
    segments.to_csv(out / "segments.csv", index=False)

    last = history["period"].max()
    future = [str(last + i) for i in range(1, C.HORIZON + 1)]
    method_of = segments.set_index(SERIES)["method"].to_dict()
    rows = []
    for (sku, loc), g in history.groupby(SERIES, sort=True):
        y = g.sort_values("period")["qty"].to_numpy()
        method = method_of[(sku, loc)]
        for period, value in zip(future, forecast(method, y, C.HORIZON)):
            rows.append({"sku": sku, "location": loc, "period": period,
                         "forecast": round(float(value), 2), "method": method})
    pd.DataFrame(rows).to_csv(out / "forecast.csv", index=False)

    print(f"issues: {len(issues)}, series: {len(segments)}, forecast rows: {len(rows)}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    main(args.data, args.out)
