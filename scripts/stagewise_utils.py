#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared utilities for the stagewise composite-factor pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
LABEL_COL = "future_return_1q"
TRAIN_Q = range(1, 22)
VALID_Q = range(22, 26)
TEST_Q = range(26, 31)
MACRO_FACTORS = ["bm", "de", "dp", "ep", "infl", "itgr", "m2gr", "mtr", "ntis", "svar", "tms"]
RATING_LABELS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "B"]

ID_COL_CANDIDATES = ["股票编号", "证券代码", "stock_id", "code"]
NAME_COL_CANDIDATES = ["证券名称", "name", "stock_name"]
META_FIELDS = {
    "股票编号",
    "证券代码",
    "证券名称",
    "stock_id",
    "code",
    "name",
    "stock_name",
    "quarter_idx",
    "trade_date",
    "report_date",
    "quarter_period",
    "date",
    "sheet_name",
    "row_id_raw",
    "label_source",
    "label_validation_status",
    "is_tail_quarter",
    "_preprocess_split",
    "split",
}
RETURN_FIELDS = {
    "月平均收益率",
    LABEL_COL,
    "区间涨跌幅(1月)",
    "区间涨跌幅(3月)",
    "区间涨跌幅(6月)",
}
RETURN_TOKENS = ("涨跌幅", "月平均收益率", "future", "return", "ret", "label", "下一期", "未来")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)


def configure_matplotlib_chinese() -> None:
    """Configure matplotlib to render Chinese labels on Windows-first setups."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def fail(message: str) -> SystemExit:
    return SystemExit(f"[ERROR] {message}")


def split_name(quarter_idx: int) -> str:
    if quarter_idx in TRAIN_Q:
        return "train"
    if quarter_idx in VALID_Q:
        return "valid"
    if quarter_idx in TEST_Q:
        return "test"
    return "unknown"


def get_stock_id_col(df: pd.DataFrame) -> str:
    for col in ID_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise fail("无法识别股票 ID/代码字段。")


def get_stock_name_col(df: pd.DataFrame) -> str | None:
    for col in NAME_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def is_return_like(col: str) -> bool:
    lower = col.lower()
    if col in RETURN_FIELDS:
        return True
    return any(token in col for token in RETURN_TOKENS) or any(token in lower for token in ("future", "return", "ret"))


def safe_feature_name(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name[:120]


def rank_ic(y_true: Iterable[float], y_score: Iterable[float]) -> float:
    y = pd.Series(y_true, dtype="float64")
    s = pd.Series(y_score, dtype="float64")
    mask = y.notna() & s.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(y.loc[mask].rank(method="average").corr(s.loc[mask].rank(method="average")))


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    y = y[mask]
    p = p[mask]
    err = y - p
    mse = float(np.mean(err**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / sst) if sst > 1e-12 else np.nan
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": float(np.mean(np.abs(err))), "r2": r2}


def max_drawdown(return_pct: pd.Series) -> float:
    if return_pct.empty:
        return np.nan
    equity = (1.0 + return_pct.fillna(0.0) / 100.0).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def minmax_high_good(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    lo = s.min(skipna=True)
    hi = s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0)


def performance_metrics(period_returns: pd.DataFrame, strategy_cols: list[str], periods_per_year: int = 4) -> pd.DataFrame:
    rows = []
    benchmark = period_returns.get("benchmark_return")
    for strategy in strategy_cols:
        if strategy not in period_returns.columns:
            continue
        r = pd.to_numeric(period_returns[strategy], errors="coerce").dropna()
        n = len(r)
        if n == 0:
            continue
        cumulative = float((1.0 + r / 100.0).prod() - 1.0)
        annual_return = float((1.0 + cumulative) ** (periods_per_year / n) - 1.0)
        annual_vol = float((r / 100.0).std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else np.nan
        sharpe = annual_return / annual_vol if annual_vol and annual_vol > 0 else np.nan
        beta = np.nan
        alpha = np.nan
        if benchmark is not None:
            b = pd.to_numeric(period_returns.loc[r.index, "benchmark_return"], errors="coerce") / 100.0
            sr = r / 100.0
            mask = sr.notna() & b.notna()
            if int(mask.sum()) >= 2 and float(b.loc[mask].var()) > 1e-12:
                beta = float(np.cov(sr.loc[mask], b.loc[mask], ddof=1)[0, 1] / b.loc[mask].var())
                alpha = float((sr.loc[mask].mean() - beta * b.loc[mask].mean()) * periods_per_year)
        rows.append(
            {
                "strategy": strategy,
                "period_count": n,
                "holding_period_avg_return": float(r.mean()),
                "cumulative_return": cumulative,
                "annualized_return": annual_return,
                "annualized_volatility": annual_vol,
                "sharpe_ratio": sharpe,
                "alpha": alpha,
                "beta": beta,
                "max_drawdown": max_drawdown(r),
                "win_rate": float((r > 0).mean()),
            }
        )
    return pd.DataFrame(rows)
