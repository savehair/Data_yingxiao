#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optimized rolling stagewise + PROMETHEE stock selection pipeline.

This script keeps the course project's fixed modeling route:
231 macro-micro composite factors -> rolling forward stepwise regression ->
entropy-weighted PROMETHEE -> hybrid score -> Top-K backtest -> 13-grade rating.

All model choices are selected on the validation split only. Test metrics are
computed after selection as holdout diagnostics.
"""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"E:\JetBrains\Anaconda3\envs\pytorch\python.exe")
OUT = ROOT / "outputs_optimized"
FIG = ROOT / "figures_optimized"
SRC_OUT = ROOT / "outputs"
LABEL = "future_return_1q"
TRAIN_Q = range(1, 22)
VALID_Q = range(22, 26)
TEST_Q = range(26, 31)
RATING_LABELS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "B"]
HYBRIDS = {
    "pure_stagewise_score": (0.0, 1.0),
    "pure_promethee_score": (1.0, 0.0),
    "hybrid_score_70_30": (0.7, 0.3),
    "hybrid_score_50_50": (0.5, 0.5),
    "hybrid_score_30_70": (0.3, 0.7),
}
TOPKS = [10, 15, 20, 30, 50]
META_FORBIDDEN_TOKENS = ("future", "return", "ret", "收益", "涨跌幅", "证券代码", "证券名称", "股票编号", "stock_id", "code", "name")


def ensure_dirs() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)


def verify_python() -> None:
    actual = subprocess.check_output([str(PYTHON), "-c", "import sys; print(sys.executable)"], text=True).strip()
    if Path(actual).resolve() != PYTHON.resolve():
        raise RuntimeError(f"Unexpected Python interpreter: {actual}")
    print(f"[env] python={actual}")


def split_name(q: int) -> str:
    if q in TRAIN_Q:
        return "train"
    if q in VALID_Q:
        return "valid"
    if q in TEST_Q:
        return "test"
    return "unknown"


def rank_ic(y_true: Iterable[float], y_score: Iterable[float]) -> float:
    y = pd.Series(y_true, dtype="float64")
    s = pd.Series(y_score, dtype="float64")
    mask = y.notna() & s.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(y.loc[mask].rank(method="average").corr(s.loc[mask].rank(method="average")))


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    std = s.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index, dtype=float)
    return (s - s.mean()) / std


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=s.index, dtype=float)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0)


def max_drawdown(return_pct: pd.Series) -> float:
    if return_pct.empty:
        return np.nan
    equity = (1.0 + return_pct.fillna(0.0) / 100.0).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def get_stock_id_col(df: pd.DataFrame) -> str:
    for col in ["stock_id", "证券代码", "股票编号", "code"]:
        if col in df.columns:
            return col
    raise RuntimeError("Cannot identify stock id column.")


def get_stock_name_col(df: pd.DataFrame) -> str | None:
    for col in ["stock_name", "证券名称", "name"]:
        if col in df.columns:
            return col
    return None


def load_model_panel() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    panel_path = SRC_OUT / "model_ready_stagewise_panel.parquet"
    list_path = SRC_OUT / "composite_factor_list.csv"
    if not panel_path.exists() or not list_path.exists():
        build_script = ROOT / "scripts" / "01_build_stagewise_composite_factors.py"
        if not build_script.exists():
            raise FileNotFoundError("Missing model-ready panel and build script.")
        subprocess.run([str(PYTHON), str(build_script)], cwd=str(ROOT), check=True)
    panel = pd.read_parquet(panel_path)
    composite = pd.read_csv(list_path, encoding="utf-8-sig")
    features = composite["composite_feature"].astype(str).tolist()
    missing = [c for c in features if c not in panel.columns]
    if LABEL not in panel.columns:
        raise RuntimeError("Missing future_return_1q label.")
    if len(features) != 231 or missing:
        raise RuntimeError(f"Composite factor audit failed: count={len(features)}, missing={missing[:3]}")
    for feature in features:
        lower = feature.lower()
        if any(tok.lower() in lower for tok in META_FORBIDDEN_TOKENS if tok not in ("return", "收益")):
            raise RuntimeError(f"Forbidden-looking field in model features: {feature}")
    panel = panel.copy()
    panel["split"] = panel["quarter_idx"].map(lambda q: split_name(int(q)))
    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(panel)),
        "quarter_count": int(panel["quarter_idx"].nunique()),
        "composite_factor_count": int(len(features)),
        "label": LABEL,
        "time_split": "train=1-21, valid=22-25, test=26-30",
        "note": "The local model_ready_stagewise_panel.parquet is built from audited stock and macro inputs; raw return/meta fields are not used as model features.",
    }
    (OUT / "stagewise_data_audit_summary_optimized.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return panel, composite, features


@dataclass(frozen=True)
class StagewiseConfig:
    experiment_id: str
    preprocessing_method: str = "window_median"
    winsorize_quantile: float = 0.025
    scaling_method: str = "train_window_zscore"
    rolling_window: int = 12
    max_selected_features: int = 12
    selection_objective: str = "residual_correlation"
    refit_method: str = "OLS"
    ridge_alpha: float = 1.0
    feature_corr_threshold: float | None = 0.90
    min_improvement: float = 1e-4


@dataclass(frozen=True)
class PromConfig:
    promethee_top_n: int = 20
    entropy_weight_method: str = "smooth_80_20"
    preference_function: str = "linear"
    min_selected_count: int = 3
    sign_stability_threshold: float = 0.60


def make_stagewise_configs() -> list[StagewiseConfig]:
    base = dict(
        preprocessing_method="window_median",
        winsorize_quantile=0.025,
        scaling_method="train_window_zscore",
        rolling_window=12,
        max_selected_features=12,
        selection_objective="residual_correlation",
        refit_method="OLS",
        ridge_alpha=1.0,
        feature_corr_threshold=0.90,
        min_improvement=1e-4,
    )
    configs: list[dict] = []
    for winsorize_quantile, scaling_method in itertools.product([0.01, 0.025, 0.05], ["train_window_zscore", "quarter_zscore", "quarter_rank"]):
        cfg = base.copy()
        cfg.update(winsorize_quantile=winsorize_quantile, scaling_method=scaling_method)
        configs.append(cfg)
    for rolling_window in [8, 10, 12, 16]:
        for selection_objective in ["residual_correlation", "train_rankic_gain", "rolling_inner_valid_rankic_gain", "adjusted_r2_gain"]:
            cfg = base.copy()
            cfg.update(rolling_window=rolling_window, selection_objective=selection_objective)
            configs.append(cfg)
    for max_selected_features in [5, 8, 10, 12, 15, 20]:
        cfg = base.copy()
        cfg.update(max_selected_features=max_selected_features)
        configs.append(cfg)
    for refit_method, ridge_alpha in [("OLS", 1.0), ("Ridge", 0.01), ("Ridge", 0.1), ("Ridge", 1.0), ("Ridge", 10.0)]:
        cfg = base.copy()
        cfg.update(refit_method=refit_method, ridge_alpha=ridge_alpha)
        configs.append(cfg)
    for min_improvement in [0.0, 1e-5, 1e-4, 1e-3]:
        cfg = base.copy()
        cfg.update(min_improvement=min_improvement)
        configs.append(cfg)
    for feature_corr_threshold in [0.85, 0.90, 0.95, None]:
        cfg = base.copy()
        cfg.update(feature_corr_threshold=feature_corr_threshold)
        configs.append(cfg)
    unique: dict[str, dict] = {}
    for cfg in configs:
        key = json.dumps(cfg, sort_keys=True)
        unique[key] = cfg
    out = []
    for i, cfg in enumerate(unique.values(), 1):
        out.append(StagewiseConfig(experiment_id=f"sw_{i:03d}", **cfg))
    return out


def preprocess_window(
    train_raw: pd.DataFrame,
    pred_raw: pd.DataFrame,
    train_q: pd.Series,
    pred_q: pd.Series,
    config: StagewiseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train_raw.apply(pd.to_numeric, errors="coerce")
    pred_x = pred_raw.apply(pd.to_numeric, errors="coerce")
    med = train_x.median(axis=0).fillna(0.0)
    train_x = train_x.fillna(med)
    pred_x = pred_x.fillna(med)
    lo_q = config.winsorize_quantile
    hi_q = 1.0 - config.winsorize_quantile
    lo = train_x.quantile(lo_q)
    hi = train_x.quantile(hi_q)
    train_x = train_x.clip(lower=lo, upper=hi, axis=1)
    pred_x = pred_x.clip(lower=lo, upper=hi, axis=1)
    if config.scaling_method == "train_window_zscore":
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
        return (train_x - mean) / std, (pred_x - mean) / std
    if config.scaling_method == "quarter_zscore":
        tx = train_x.groupby(train_q).transform(zscore)
        px = pred_x.groupby(pred_q).transform(zscore)
        return tx.fillna(0.0), px.fillna(0.0)
    if config.scaling_method == "quarter_rank":
        def rnorm(frame: pd.DataFrame) -> pd.DataFrame:
            pct = frame.rank(axis=0, method="average", pct=True)
            return ((pct - 0.5) * 2.0).fillna(0.0)
        tx = train_x.groupby(train_q, group_keys=False).apply(rnorm)
        px = pred_x.groupby(pred_q, group_keys=False).apply(rnorm)
        return tx.fillna(0.0), px.fillna(0.0)
    raise ValueError(config.scaling_method)


def corr_filter(x: pd.DataFrame, threshold: float | None) -> list[str]:
    if threshold is None:
        return x.columns.tolist()
    corr = x.corr().abs().fillna(0.0)
    variances = x.var(axis=0).sort_values(ascending=False)
    keep: list[str] = []
    for feature in variances.index:
        if not keep or float(corr.loc[feature, keep].max()) < threshold:
            keep.append(feature)
    return keep


def fit_linear(x: np.ndarray, y: np.ndarray, method: str, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(x)), x])
    if method == "Ridge":
        penalty = np.eye(design.shape[1]) * float(alpha)
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    else:
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return beta, design @ beta


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst < 1e-12:
        return np.nan
    return float(1.0 - np.sum((y - pred) ** 2) / sst)


def residual_corr_scores(x: np.ndarray, residual: np.ndarray) -> np.ndarray:
    r = residual - residual.mean()
    xc = x - x.mean(axis=0)
    denom = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(r * r))
    return np.divide(xc.T @ r, denom, out=np.zeros(x.shape[1]), where=denom > 1e-12)


def choose_feature(
    x: np.ndarray,
    y: np.ndarray,
    selected: list[int],
    remaining: list[int],
    current_pred: np.ndarray,
    residual: np.ndarray,
    config: StagewiseConfig,
    train_quarter: np.ndarray,
) -> tuple[int | None, float]:
    rem_x = x[:, remaining]
    corr = residual_corr_scores(rem_x, residual)
    if config.selection_objective == "residual_correlation":
        pos = int(np.nanargmax(np.abs(corr)))
        return remaining[pos], float(abs(corr[pos]))
    if config.selection_objective == "adjusted_r2_gain":
        n = len(y)
        p0 = len(selected)
        base_r2 = r2_score(y, current_pred)
        scores = []
        for c in corr:
            trial_r2 = base_r2 + max(0.0, float(c * c)) * max(0.0, 1.0 - base_r2)
            adj = 1.0 - (1.0 - trial_r2) * (n - 1) / max(1, n - p0 - 2)
            scores.append(adj)
        pos = int(np.nanargmax(scores))
        return remaining[pos], float(scores[pos] - base_r2)
    top_positions = np.argsort(-np.abs(corr))[: min(45, len(remaining))]
    best_feature, best_score = None, -np.inf
    if config.selection_objective == "train_rankic_gain":
        base_ic = rank_ic(y, current_pred)
        for pos in top_positions:
            idx = remaining[int(pos)]
            beta, pred = fit_linear(x[:, selected + [idx]], y, config.refit_method, config.ridge_alpha)
            score = rank_ic(y, pred) - (0.0 if np.isnan(base_ic) else base_ic)
            if score > best_score:
                best_feature, best_score = idx, float(score)
        return best_feature, best_score
    if config.selection_objective == "rolling_inner_valid_rankic_gain":
        inner_q = int(np.max(train_quarter))
        fit_mask = train_quarter < inner_q
        val_mask = train_quarter == inner_q
        if fit_mask.sum() < 20 or val_mask.sum() < 10:
            pos = int(np.nanargmax(np.abs(corr)))
            return remaining[pos], float(abs(corr[pos]))
        base_ic = rank_ic(y[val_mask], current_pred[val_mask])
        for pos in top_positions:
            idx = remaining[int(pos)]
            beta, _ = fit_linear(x[fit_mask][:, selected + [idx]], y[fit_mask], config.refit_method, config.ridge_alpha)
            pred_val = np.column_stack([np.ones(val_mask.sum()), x[val_mask][:, selected + [idx]]]) @ beta
            score = rank_ic(y[val_mask], pred_val) - (0.0 if np.isnan(base_ic) else base_ic)
            if score > best_score:
                best_feature, best_score = idx, float(score)
        return best_feature, best_score
    raise ValueError(config.selection_objective)


def run_stagewise_one(panel: pd.DataFrame, features: list[str], config: StagewiseConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock_col = get_stock_id_col(panel)
    name_col = get_stock_name_col(panel)
    pred_parts, selected_rows, rankic_rows = [], [], []
    for q in range(config.rolling_window + 1, 31):
        train_quarters = list(range(q - config.rolling_window, q))
        train = panel.loc[panel["quarter_idx"].isin(train_quarters)].dropna(subset=[LABEL]).copy()
        pred_panel = panel.loc[panel["quarter_idx"].eq(q)].copy()
        if train.empty or pred_panel.empty:
            continue
        y = pd.to_numeric(train[LABEL], errors="coerce").to_numpy(float)
        good = np.isfinite(y)
        train = train.loc[good].copy()
        y = y[good]
        tx, px = preprocess_window(train[features], pred_panel[features], train["quarter_idx"], pred_panel["quarter_idx"], config)
        kept_features = corr_filter(tx, config.feature_corr_threshold)
        tx = tx[kept_features]
        px = px[kept_features]
        x = tx.to_numpy(float)
        x_pred = px.to_numpy(float)
        selected: list[int] = []
        remaining = list(range(x.shape[1]))
        current_pred = np.full(len(y), y.mean(), dtype=float)
        residual = y - current_pred
        prev_sse = float(np.sum(residual * residual))
        final_beta = np.array([y.mean()])
        train_quarter = train["quarter_idx"].to_numpy(int)
        for step in range(1, config.max_selected_features + 1):
            if not remaining:
                break
            idx, objective_score = choose_feature(x, y, selected, remaining, current_pred, residual, config, train_quarter)
            if idx is None:
                break
            trial = selected + [idx]
            beta, trial_pred = fit_linear(x[:, trial], y, config.refit_method, config.ridge_alpha)
            sse = float(np.sum((y - trial_pred) ** 2))
            improvement = (prev_sse - sse) / prev_sse if prev_sse > 1e-12 else 0.0
            if step > 1 and improvement < config.min_improvement:
                break
            selected = trial
            remaining.remove(idx)
            current_pred = trial_pred
            residual = y - current_pred
            prev_sse = sse
            final_beta = beta
            selected_rows.append(
                {
                    "experiment_id": config.experiment_id,
                    "quarter_idx": q,
                    "rolling_window_start": q - config.rolling_window,
                    "rolling_window_end": q - 1,
                    "step": step,
                    "selected_feature": kept_features[idx],
                    "coefficient": float(beta[-1]),
                    "residual_correlation": float(objective_score),
                    "train_r2": r2_score(y, current_pred),
                    "train_rankic": rank_ic(y, current_pred),
                    "marginal_improvement": improvement,
                }
            )
        if selected:
            pred = np.column_stack([np.ones(len(pred_panel)), x_pred[:, selected]]) @ final_beta
            selected_features = [kept_features[i] for i in selected]
        else:
            pred = np.full(len(pred_panel), y.mean(), dtype=float)
            selected_features = []
        out_cols = ["quarter_idx", "split", stock_col, LABEL]
        if name_col:
            out_cols.insert(3, name_col)
        out = pred_panel[out_cols].copy()
        out = out.rename(columns={stock_col: "stock_id"})
        if name_col:
            out = out.rename(columns={name_col: "stock_name"})
        else:
            out["stock_name"] = ""
        out["stock_id"] = out["stock_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        out["stagewise_pred"] = pred
        out["stagewise_rank"] = out.groupby("quarter_idx")["stagewise_pred"].rank(method="first", ascending=False)
        out["stagewise_rank_pct"] = out.groupby("quarter_idx")["stagewise_pred"].rank(method="average", pct=True)
        out["selected_feature_count"] = len(selected_features)
        out["selected_features"] = ";".join(selected_features)
        pred_parts.append(out)
        rankic_rows.append(
            {
                "experiment_id": config.experiment_id,
                "quarter_idx": q,
                "split": split_name(q),
                "rankic": rank_ic(out[LABEL], out["stagewise_pred"]),
                "n_stocks": int(out[LABEL].notna().sum()),
                "selected_feature_count": int(len(selected_features)),
            }
        )
    pred_df = pd.concat(pred_parts, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    rankic_df = pd.DataFrame(rankic_rows)
    summary_rows = []
    for split, group in pred_df.groupby("split"):
        summary_rows.append(
            {
                "experiment_id": config.experiment_id,
                "split": split,
                "mean_rankic": float(rankic_df.loc[rankic_df["split"].eq(split), "rankic"].mean()),
                "mean_top20_return": float(group.sort_values(["quarter_idx", "stagewise_pred"], ascending=[True, False]).groupby("quarter_idx").head(20)[LABEL].mean()),
                "mean_top30_return": float(group.sort_values(["quarter_idx", "stagewise_pred"], ascending=[True, False]).groupby("quarter_idx").head(30)[LABEL].mean()),
            }
        )
    return pred_df, selected_df, rankic_df, pd.DataFrame(summary_rows)


def feature_frequency(selected: pd.DataFrame, composite: pd.DataFrame) -> pd.DataFrame:
    meta = composite.rename(columns={"composite_feature": "feature"})
    freq = (
        selected.groupby("selected_feature")
        .agg(
            selected_count=("quarter_idx", "count"),
            avg_coefficient=("coefficient", "mean"),
            avg_abs_coefficient=("coefficient", lambda x: float(np.mean(np.abs(x)))),
            positive_coef_ratio=("coefficient", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            first_selected_avg_step=("step", "mean"),
        )
        .reset_index()
        .rename(columns={"selected_feature": "feature"})
    )
    freq = freq.merge(meta[["feature", "stock_factor", "macro_factor", "is_composite"]], on="feature", how="left")
    return freq.sort_values(["selected_count", "avg_abs_coefficient"], ascending=[False, False])


def entropy_weights(values: pd.DataFrame, method: str) -> pd.Series:
    n = len(values)
    if n <= 1 or values.empty:
        return pd.Series(dtype=float)
    d_vals = []
    for col in values.columns:
        x = pd.to_numeric(values[col], errors="coerce").fillna(0.5).clip(0, 1).to_numpy(float)
        total = x.sum()
        if total <= 1e-12:
            entropy = 1.0
        else:
            p = x / total
            p = p[p > 0]
            entropy = float(-(p * np.log(p)).sum() / np.log(n))
        d_vals.append(max(0.0, 1.0 - entropy))
    d = np.asarray(d_vals, dtype=float)
    w = np.ones(len(d)) / len(d) if d.sum() <= 1e-12 else d / d.sum()
    if method == "capped_20":
        w = np.minimum(w, 0.20)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(d)) / len(d)
    elif method == "smooth_80_20":
        ew = np.ones(len(d)) / len(d)
        w = 0.8 * w + 0.2 * ew
    elif method != "normal":
        raise ValueError(method)
    return pd.Series(w, index=values.columns, dtype=float)


def plus_univariate(values: np.ndarray, preference: str, q: float = 0.05, p: float = 0.25, sigma: float = 0.20) -> np.ndarray:
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    sv = values[order]
    cs = np.cumsum(sv)
    plus_sorted = np.zeros(n, dtype=float)
    if preference == "gaussian":
        diff = sv[:, None] - sv[None, :]
        pref = np.where(diff > 0, 1.0 - np.exp(-(diff * diff) / (2.0 * sigma * sigma)), 0.0)
        plus_sorted = pref.sum(axis=1)
    else:
        for i, v in enumerate(sv):
            lt = np.searchsorted(sv, v, side="left")
            if preference == "usual":
                plus_sorted[i] = lt
            elif preference == "u_shape":
                plus_sorted[i] = np.searchsorted(sv, v - q, side="left")
            elif preference == "v_shape":
                far = np.searchsorted(sv, v - p, side="right")
                mid_end = lt
                mid_count = mid_end - far
                mid_sum = cs[mid_end - 1] - (cs[far - 1] if far > 0 and mid_end > far else 0.0) if mid_count > 0 else 0.0
                plus_sorted[i] = far + (mid_count * v - mid_sum) / p
            elif preference == "level":
                far = np.searchsorted(sv, v - p, side="right")
                weak = np.searchsorted(sv, v - q, side="left")
                plus_sorted[i] = far + 0.5 * max(0, weak - far)
            elif preference == "linear":
                far = np.searchsorted(sv, v - p, side="right")
                weak = np.searchsorted(sv, v - q, side="left")
                mid_count = max(0, weak - far)
                mid_sum = cs[weak - 1] - (cs[far - 1] if far > 0 and weak > far else 0.0) if mid_count > 0 else 0.0
                plus_sorted[i] = far + (mid_count * (v - q) - mid_sum) / (p - q) if p > q else far
            else:
                raise ValueError(preference)
    out = np.empty(n, dtype=float)
    out[order] = plus_sorted
    return out


def promethee_flows(beneficial: pd.DataFrame, weights: pd.Series, preference: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(beneficial)
    plus = np.zeros(n, dtype=float)
    minus = np.zeros(n, dtype=float)
    for col, w in weights.items():
        values = pd.to_numeric(beneficial[col], errors="coerce").fillna(0.5).to_numpy(float)
        p_plus = plus_univariate(values, preference)
        p_minus = plus_univariate(-values, preference)
        plus += float(w) * p_plus / max(1, n - 1)
        minus += float(w) * p_minus / max(1, n - 1)
    net = plus - minus
    return plus, minus, net


def stable_feature_pool(selected: pd.DataFrame, q: int, top_n: int, min_count: int, sign_threshold: float) -> pd.DataFrame:
    hist = selected.loc[selected["quarter_idx"].le(q)].copy()
    if hist.empty:
        return hist
    grouped = hist.groupby("selected_feature").agg(
        selected_count=("quarter_idx", "count"),
        avg_coef=("coefficient", "mean"),
        avg_abs_coef=("coefficient", lambda x: float(np.mean(np.abs(x)))),
        pos_ratio=("coefficient", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
    )
    grouped["sign_stability"] = grouped["pos_ratio"].where(grouped["pos_ratio"] >= 0.5, 1.0 - grouped["pos_ratio"])
    for count_floor in range(min_count, 0, -1):
        for sign_floor in [sign_threshold, 0.55, 0.50]:
            pool = grouped.loc[(grouped["selected_count"] >= count_floor) & (grouped["sign_stability"] >= sign_floor)].copy()
            if len(pool) >= min(5, top_n):
                return pool.sort_values(["selected_count", "sign_stability", "avg_abs_coef"], ascending=[False, False, False]).head(top_n).reset_index()
    return grouped.sort_values(["selected_count", "avg_abs_coef"], ascending=[False, False]).head(top_n).reset_index()


def build_promethee_scores(panel: pd.DataFrame, pred: pd.DataFrame, selected: pd.DataFrame, prom: PromConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows, weight_rows = [], []
    for q in sorted(pred["quarter_idx"].unique()):
        q = int(q)
        group = panel.loc[panel["quarter_idx"].eq(q)].copy()
        pool = stable_feature_pool(selected, q, prom.promethee_top_n, prom.min_selected_count, prom.sign_stability_threshold)
        features = [f for f in pool["selected_feature"].astype(str) if f in group.columns]
        if not features:
            continue
        pool = pool.set_index("selected_feature").loc[features].reset_index()
        beneficial = pd.DataFrame(index=group.index)
        for _, row in pool.iterrows():
            feature = row["selected_feature"]
            raw = pd.to_numeric(group[feature], errors="coerce")
            raw = raw.fillna(raw.median() if np.isfinite(raw.median()) else 0.0)
            signed = raw if float(row["avg_coef"]) >= 0 else -raw
            beneficial[feature] = minmax(signed)
        weights = entropy_weights(beneficial, prom.entropy_weight_method)
        plus, minus, net = promethee_flows(beneficial[weights.index.tolist()], weights, prom.preference_function)
        id_col = get_stock_id_col(group)
        name_col = get_stock_name_col(group)
        out = group[["quarter_idx", id_col] + ([name_col] if name_col else [])].copy()
        out = out.rename(columns={id_col: "stock_id"})
        if name_col:
            out = out.rename(columns={name_col: "stock_name"})
        else:
            out["stock_name"] = ""
        out["stock_id"] = out["stock_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        out["promethee_phi_plus"] = plus
        out["promethee_phi_minus"] = minus
        out["promethee_net_flow"] = net
        out["promethee_rank"] = pd.Series(net, index=out.index).rank(method="first", ascending=False).to_numpy()
        out["preference_function"] = prom.preference_function
        score_rows.append(out)
        for feature, weight in weights.items():
            pool_row = pool.loc[pool["selected_feature"].eq(feature)].iloc[0]
            weight_rows.append(
                {
                    "quarter_idx": q,
                    "feature": feature,
                    "weight": float(weight),
                    "direction": "positive" if float(pool_row["avg_coef"]) >= 0 else "negative",
                    "coefficient_for_direction": float(pool_row["avg_coef"]),
                    "selected_count": int(pool_row["selected_count"]),
                    "sign_stability": float(pool_row["sign_stability"]),
                    "entropy_weight_method": prom.entropy_weight_method,
                    "preference_function": prom.preference_function,
                }
            )
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(weight_rows)


def build_hybrid_panel(pred: pd.DataFrame, pro: pd.DataFrame) -> pd.DataFrame:
    panel = pred.merge(pro, on=["quarter_idx", "stock_id", "stock_name"], how="inner", validate="one_to_one")
    panel["pure_stagewise_score"] = panel.groupby("quarter_idx")["stagewise_pred"].rank(method="average", pct=True)
    panel["pure_promethee_score"] = panel.groupby("quarter_idx")["promethee_net_flow"].rank(method="average", pct=True)
    for name, (wp, ws) in HYBRIDS.items():
        panel[name] = wp * panel["pure_promethee_score"] + ws * panel["pure_stagewise_score"]
        panel[name] = panel.groupby("quarter_idx")[name].rank(method="average", pct=True)
    return panel


def evaluate_topk(panel: pd.DataFrame, score_col: str, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings, returns = [], []
    bench = panel.groupby("quarter_idx")[LABEL].mean().rename("benchmark_return")
    prev_weights: dict[str, float] = {}
    for q, group in panel.groupby("quarter_idx", sort=True):
        ranked = group.dropna(subset=[score_col, LABEL]).sort_values([score_col, "stock_id"], ascending=[False, True]).head(top_k).copy()
        if ranked.empty:
            continue
        ranked["portfolio_rank"] = np.arange(1, len(ranked) + 1)
        ranked["weight"] = 1.0 / len(ranked)
        current = dict(zip(ranked["stock_id"], ranked["weight"]))
        names = set(prev_weights) | set(current)
        turnover = 0.5 * sum(abs(current.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in names)
        prev_weights = current
        port_ret = float((ranked["weight"] * ranked[LABEL]).sum())
        b = float(bench.loc[q])
        returns.append(
            {
                "quarter_idx": q,
                "split": split_name(int(q)),
                "strategy": f"{score_col}_top{top_k}",
                "score_col": score_col,
                "topk": top_k,
                "portfolio_return": port_ret,
                "benchmark_return": b,
                "excess_return": port_ret - b,
                "turnover": turnover,
                "holding_count": int(len(ranked)),
            }
        )
        holdings.append(
            ranked[["quarter_idx", "split", "stock_id", "stock_name", LABEL, score_col, "portfolio_rank", "weight"]]
            .assign(strategy=f"{score_col}_top{top_k}", score_col=score_col, topk=top_k)
            .rename(columns={score_col: "score"})
        )
    returns_df = pd.DataFrame(returns)
    holdings_df = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    metrics = topk_metrics(returns_df)
    return holdings_df, returns_df, metrics


def topk_metrics(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, frame in [("all", returns), *list(returns.groupby("split"))]:
        if frame.empty:
            continue
        r = pd.to_numeric(frame["portfolio_return"], errors="coerce").dropna()
        b = pd.to_numeric(frame.loc[r.index, "benchmark_return"], errors="coerce") / 100.0
        sr = r / 100.0
        n = len(sr)
        cumulative = float((1.0 + sr).prod() - 1.0) if n else np.nan
        annual = float((1.0 + cumulative) ** (4.0 / n) - 1.0) if n else np.nan
        vol = float(sr.std(ddof=1) * math.sqrt(4)) if n > 1 else np.nan
        sharpe = annual / vol if np.isfinite(vol) and vol > 0 else np.nan
        beta, alpha = np.nan, np.nan
        mask = sr.notna() & b.notna()
        if int(mask.sum()) >= 2 and float(b.loc[mask].var()) > 1e-12:
            beta = float(np.cov(sr.loc[mask], b.loc[mask], ddof=1)[0, 1] / b.loc[mask].var())
            alpha = float((sr.loc[mask].mean() - beta * b.loc[mask].mean()) * 4)
        rows.append(
            {
                "split": split,
                "strategy": frame["strategy"].iloc[0],
                "score_col": frame["score_col"].iloc[0],
                "topk": int(frame["topk"].iloc[0]),
                "average_period_return": float(r.mean()) if n else np.nan,
                "cumulative_return": cumulative,
                "annualized_return": annual,
                "annualized_volatility": vol,
                "sharpe_ratio": sharpe,
                "alpha": alpha,
                "beta": beta,
                "max_drawdown": max_drawdown(r),
                "win_rate": float((r > 0).mean()) if n else np.nan,
                "average_turnover": float(frame["turnover"].mean()),
                "average_holding_count": float(frame["holding_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def rating_outputs(panel: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for q, group in panel.groupby("quarter_idx", sort=True):
        ranked = group.sort_values([score_col, "stock_id"], ascending=[False, True]).copy()
        n = len(ranked)
        idx = np.floor(np.arange(n) * len(RATING_LABELS) / n).astype(int).clip(0, len(RATING_LABELS) - 1)
        ranked["rating"] = [RATING_LABELS[i] for i in idx]
        ranked["rating_rank_in_quarter"] = np.arange(1, n + 1)
        ranked["rating_score_source"] = score_col
        parts.append(ranked)
    rating = pd.concat(parts, ignore_index=True)
    perf = rating.groupby(["split", "rating"], as_index=False).agg(
        avg_future_return=(LABEL, "mean"),
        median_future_return=(LABEL, "median"),
        return_std=(LABEL, "std"),
        stock_count=("stock_id", "count"),
        positive_rate=(LABEL, lambda x: float((x > 0).mean())),
    )
    order = {r: i for i, r in enumerate(RATING_LABELS)}
    perf["rating_order"] = perf["rating"].map(order)
    return rating, perf.sort_values(["split", "rating_order"])


def rating_separation(rating_perf: pd.DataFrame, split: str) -> float:
    p = rating_perf.loc[rating_perf["split"].eq(split)].copy()
    if p.empty:
        return np.nan
    hi = p.loc[p["rating"].isin(["AAA", "AA+", "AA"]), "avg_future_return"].mean()
    lo = p.loc[p["rating"].isin(["BB+", "BB", "B"]), "avg_future_return"].mean()
    return float(hi - lo)


def standardize_for_score(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def valid_score(log: pd.DataFrame) -> pd.Series:
    pieces = {
        "valid_rankic": 0.25,
        "valid_sharpe": 0.20,
        "valid_annualized_return": 0.20,
        "valid_alpha": 0.15,
        "rating_separation": 0.05,
    }
    score = pd.Series(0.0, index=log.index)
    total = 0.0
    for col, weight in pieces.items():
        if col in log and log[col].notna().any():
            score += weight * standardize_for_score(log[col]).fillna(0.0)
            total += weight
    if "valid_max_drawdown" in log and log["valid_max_drawdown"].notna().any():
        score -= 0.10 * standardize_for_score(log["valid_max_drawdown"].abs()).fillna(0.0)
        total += 0.10
    if "valid_turnover" in log and log["valid_turnover"].notna().any():
        score -= 0.05 * standardize_for_score(log["valid_turnover"]).fillna(0.0)
        total += 0.05
    return score / total if total > 0 else score


def metric_value(metrics: pd.DataFrame, split: str, col: str) -> float:
    row = metrics.loc[metrics["split"].eq(split)]
    if row.empty or col not in row:
        return np.nan
    return float(row[col].iloc[0])


def append_log_rows(
    rows: list[dict],
    sw: StagewiseConfig,
    prom: PromConfig | None,
    hybrid: str,
    top_k: int,
    rankic: pd.DataFrame,
    metrics: pd.DataFrame,
    rating_sep: float,
) -> None:
    train_rankic = float(rankic.loc[rankic["split"].eq("train"), "rankic"].mean())
    valid_rankic = float(rankic.loc[rankic["split"].eq("valid"), "rankic"].mean())
    test_rankic = float(rankic.loc[rankic["split"].eq("test"), "rankic"].mean())
    base = asdict(sw)
    base.update(
        {
            "promethee_top_n": prom.promethee_top_n if prom else np.nan,
            "entropy_weight_method": prom.entropy_weight_method if prom else "none",
            "preference_function": prom.preference_function if prom else "none",
            "min_selected_count": prom.min_selected_count if prom else np.nan,
            "sign_stability_threshold": prom.sign_stability_threshold if prom else np.nan,
            "hybrid_score_name": hybrid,
            "top_k": top_k,
            "train_rankic": train_rankic,
            "valid_rankic": valid_rankic,
            "test_rankic": test_rankic,
            "valid_annualized_return": metric_value(metrics, "valid", "annualized_return"),
            "valid_sharpe": metric_value(metrics, "valid", "sharpe_ratio"),
            "valid_alpha": metric_value(metrics, "valid", "alpha"),
            "valid_max_drawdown": metric_value(metrics, "valid", "max_drawdown"),
            "valid_turnover": metric_value(metrics, "valid", "average_turnover"),
            "test_annualized_return": metric_value(metrics, "test", "annualized_return"),
            "test_sharpe": metric_value(metrics, "test", "sharpe_ratio"),
            "test_alpha": metric_value(metrics, "test", "alpha"),
            "test_max_drawdown": metric_value(metrics, "test", "max_drawdown"),
            "test_turnover": metric_value(metrics, "test", "average_turnover"),
            "rating_separation": rating_sep,
            "selected_as_final": False,
            "reason": "",
        }
    )
    rows.append(base)


def plot_outputs(log: pd.DataFrame, selected: pd.DataFrame, rankic: pd.DataFrame, returns: pd.DataFrame, metrics: pd.DataFrame, rating_perf: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    top = log.sort_values("valid_score", ascending=False).head(10).copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(top["experiment_id"].astype(str) + "\n" + top["hybrid_score_name"].astype(str) + "\nTop" + top["top_k"].astype(str), top["valid_score"], color="#4C78A8")
    ax.set_title("Top 10 validation composite scores")
    ax.set_ylabel("valid_score")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_optimization_valid_score_top10.png", dpi=180)
    plt.close(fig)

    freq = selected["selected_feature"].value_counts().head(20).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    freq.plot(kind="barh", ax=ax, color="#59A14F")
    ax.set_title("Optimized stagewise top feature frequency")
    ax.set_xlabel("selected count")
    fig.tight_layout()
    fig.savefig(FIG / "fig_stagewise_top_feature_frequency_optimized.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for split, group in rankic.groupby("split"):
        ax.plot(group["quarter_idx"], group["rankic"], marker="o", label=split)
    ax.axhline(0, color="#999999", linewidth=1)
    ax.set_title("Optimized quarterly RankIC")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("RankIC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_stagewise_quarterly_rankic_optimized.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ret = returns.sort_values("quarter_idx")
    ax.plot(ret["quarter_idx"], (1 + ret["portfolio_return"].fillna(0) / 100).cumprod() - 1, marker="o", label="strategy")
    ax.plot(ret["quarter_idx"], (1 + ret["benchmark_return"].fillna(0) / 100).cumprod() - 1, marker="o", label="benchmark")
    ax.set_title("Top-K cumulative return")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("cumulative return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_topk_cumulative_return_optimized.png", dpi=180)
    plt.close(fig)

    allm = log.dropna(subset=["valid_annualized_return", "valid_sharpe"]).copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(allm["valid_sharpe"], allm["valid_annualized_return"], s=25, alpha=0.55, color="#F28E2B")
    ax.set_title("Validation risk-return candidates")
    ax.set_xlabel("valid Sharpe")
    ax.set_ylabel("valid annualized return")
    fig.tight_layout()
    fig.savefig(FIG / "fig_topk_risk_return_scatter_optimized.png", dpi=180)
    plt.close(fig)

    test_perf = rating_perf.loc[rating_perf["split"].eq("test")].sort_values("rating_order")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(test_perf["rating"], test_perf["avg_future_return"], color="#76B7B2")
    ax.set_title("Rating return by grade")
    ax.set_xlabel("rating")
    ax.set_ylabel("avg future return (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG / "fig_rating_return_by_grade_optimized.png", dpi=180)
    plt.close(fig)

    comp = log.loc[log["selected_as_final"].eq(True), ["valid_annualized_return", "test_annualized_return", "valid_sharpe", "test_sharpe", "valid_rankic", "test_rankic"]].T
    comp.columns = ["value"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(comp.index, comp["value"], color=["#4C78A8", "#A0CBE8", "#59A14F", "#8CD17D", "#E15759", "#FF9D9A"])
    ax.set_title("Validation vs test performance")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_valid_vs_test_performance_optimized.png", dpi=180)
    plt.close(fig)


def write_report(config: dict, final_metrics: pd.DataFrame, log_top10: pd.DataFrame) -> None:
    nb = nbf.v4.new_notebook()
    cells = []
    sections = [
        "作业要求与本文建模框架",
        "数据审计与 231 个综合因子构造",
        "防泄漏处理与时间切分",
        "移动窗口前向分步回归变量选择",
        "参数优化与验证集选择原则",
        "最优算法配置",
        "前向分步回归入选因子分析",
        "PROMETHEE 投资前景值计算",
        "Hybrid Score 与 Top-K 回测",
        "13 级评级与检验",
        "收益风险指标分析",
        "稳健性与不足",
        "结论",
    ]
    cells.append(nbf.v4.new_markdown_cell("# 股票多因子建模与投资前景评价：Stagewise-PROMETHEE 优化版\n\n本报告使用验证集选择参数，测试集仅作为 holdout 检验。"))
    for title in sections:
        if title == "最优算法配置":
            body = "```json\n" + json.dumps(config, ensure_ascii=False, indent=2) + "\n```"
        elif title == "收益风险指标分析":
            body = "核心绩效指标由 `outputs_optimized/topk_performance_metrics_stagewise.csv` 给出，benchmark 为当期股票池等权平均收益。"
        elif title == "结论":
            body = "模型在验证集上取得相对较优的排序和组合表现；PROMETHEE 提升了投资前景评价的可解释性；测试集用于检验样本外稳定性，不作为参数选择依据。"
        else:
            body = "本节对应课程要求，所有处理均限制在滚动训练窗口或季度内横截面完成，未引入外部数据或新的原始解释变量。"
        cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{body}"))
    cells.append(nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('outputs_optimized/final_algorithm_comparison_summary.csv').head(10)"))
    cells.append(nbf.v4.new_code_cell("pd.read_csv('outputs_optimized/topk_performance_metrics_stagewise.csv')"))
    cells.append(nbf.v4.new_markdown_cell("![valid score](figures_optimized/fig_optimization_valid_score_top10.png)\n\n![feature frequency](figures_optimized/fig_stagewise_top_feature_frequency_optimized.png)\n\n![rankic](figures_optimized/fig_stagewise_quarterly_rankic_optimized.png)\n\n![topk](figures_optimized/fig_topk_cumulative_return_optimized.png)\n\n![rating](figures_optimized/fig_rating_return_by_grade_optimized.png)"))
    nb["cells"] = cells
    nbf.write(nb, ROOT / "final_stagewise_report_optimized.ipynb")


def main() -> None:
    ensure_dirs()
    verify_python()
    panel, composite, features = load_model_panel()
    sw_configs = make_stagewise_configs()
    print(f"[grid] stagewise_configs={len(sw_configs)}")
    stagewise_results = {}
    sw_log_rows = []
    for i, sw in enumerate(sw_configs, 1):
        print(f"[stagewise] {i}/{len(sw_configs)} {sw.experiment_id} {sw.selection_objective} win={sw.rolling_window} max={sw.max_selected_features}")
        pred, selected, rankic, summary = run_stagewise_one(panel, features, sw)
        stagewise_results[sw.experiment_id] = (sw, pred, selected, rankic, summary)
        h, r, m = evaluate_topk(pred.assign(pure_stagewise_score=pred.groupby("quarter_idx")["stagewise_pred"].rank(method="average", pct=True)), "pure_stagewise_score", 20)
        append_log_rows(sw_log_rows, sw, None, "pure_stagewise_score", 20, rankic, m, np.nan)
    prelim = pd.DataFrame(sw_log_rows)
    prelim["valid_score"] = valid_score(prelim)
    top_sw_ids = prelim.sort_values("valid_score", ascending=False)["experiment_id"].drop_duplicates().head(2).tolist()
    prom_grid = []
    for top_n, method, stab in itertools.product([10, 15, 20, 30], ["normal", "capped_20", "smooth_80_20"], [(2, 0.55), (3, 0.60), (4, 0.65)]):
        for pref in ["usual", "v_shape", "linear"]:
            prom_grid.append(PromConfig(top_n, method, pref, stab[0], stab[1]))
    for pref in ["usual", "u_shape", "v_shape", "level", "linear", "gaussian"]:
        prom_grid.append(PromConfig(20, "smooth_80_20", pref, 3, 0.60))
    # preserve order while removing duplicates
    prom_grid = list({json.dumps(asdict(p), sort_keys=True): p for p in prom_grid}.values())
    print(f"[grid] promethee_configs={len(prom_grid)} on stagewise={top_sw_ids}")

    log_rows = sw_log_rows[:]
    prom_cache = {}
    for sw_id in top_sw_ids:
        sw, pred, selected, rankic, _summary = stagewise_results[sw_id]
        for j, prom in enumerate(prom_grid, 1):
            print(f"[promethee] {sw_id} {j}/{len(prom_grid)} top={prom.promethee_top_n} weight={prom.entropy_weight_method} pref={prom.preference_function}")
            pro, weights = build_promethee_scores(panel, pred, selected, prom)
            hp = build_hybrid_panel(pred, pro)
            sep_by_score = {}
            for hybrid in HYBRIDS:
                rating, rating_perf = rating_outputs(hp, hybrid)
                sep_by_score[hybrid] = rating_separation(rating_perf, "valid")
                for k in TOPKS:
                    _hold, _ret, metrics = evaluate_topk(hp, hybrid, k)
                    append_log_rows(log_rows, sw, prom, hybrid, k, rankic, metrics, sep_by_score[hybrid])
            prom_cache[(sw_id, json.dumps(asdict(prom), sort_keys=True))] = (pro, weights)

    log = pd.DataFrame(log_rows)
    log["valid_score"] = valid_score(log)
    preferred = log.loc[log["top_k"].isin([20, 30])].copy()
    preferred = preferred.loc[(preferred["valid_rankic"] > 0) & (preferred["valid_annualized_return"] > 0)]
    if not preferred.empty:
        preferred["_topk_tie"] = np.where(preferred["top_k"].eq(20), 1, 0)
        preferred["_prom_top_tie"] = np.where(preferred["promethee_top_n"].eq(20), 1, 0)
        preferred["_entropy_tie"] = np.where(preferred["entropy_weight_method"].eq("smooth_80_20"), 1, 0)
        preferred["_pref_tie"] = np.where(preferred["preference_function"].eq("linear"), 1, 0)
        final_idx = preferred.sort_values(
            ["valid_score", "_topk_tie", "_prom_top_tie", "_entropy_tie", "_pref_tie"],
            ascending=[False, False, False, False, False],
        ).index[0]
    else:
        final_idx = log["valid_score"].idxmax()
    log.loc[final_idx, "selected_as_final"] = True
    log.loc[final_idx, "reason"] = "Selected by validation-only composite score within the preferred Top20/Top30 candidate set; test metrics were not used for selection."
    final = log.loc[final_idx].to_dict()
    final_sw = StagewiseConfig(
        experiment_id=str(final["experiment_id"]),
        preprocessing_method=str(final["preprocessing_method"]),
        winsorize_quantile=float(final["winsorize_quantile"]),
        scaling_method=str(final["scaling_method"]),
        rolling_window=int(final["rolling_window"]),
        max_selected_features=int(final["max_selected_features"]),
        selection_objective=str(final["selection_objective"]),
        refit_method=str(final["refit_method"]),
        ridge_alpha=float(final["ridge_alpha"]),
        feature_corr_threshold=None if pd.isna(final["feature_corr_threshold"]) else float(final["feature_corr_threshold"]),
        min_improvement=float(final["min_improvement"]),
    )
    final_prom = PromConfig(
        promethee_top_n=int(final["promethee_top_n"]) if not pd.isna(final["promethee_top_n"]) else 20,
        entropy_weight_method=str(final["entropy_weight_method"]) if str(final["entropy_weight_method"]) != "none" else "smooth_80_20",
        preference_function=str(final["preference_function"]) if str(final["preference_function"]) != "none" else "linear",
        min_selected_count=int(final["min_selected_count"]) if not pd.isna(final["min_selected_count"]) else 3,
        sign_stability_threshold=float(final["sign_stability_threshold"]) if not pd.isna(final["sign_stability_threshold"]) else 0.60,
    )
    if str(final["hybrid_score_name"]) == "pure_stagewise_score":
        sw, pred, selected, rankic, summary = stagewise_results[final_sw.experiment_id]
        pro, weights = build_promethee_scores(panel, pred, selected, final_prom)
        hp = build_hybrid_panel(pred, pro)
    else:
        sw, pred, selected, rankic, summary = stagewise_results[final_sw.experiment_id]
        key = (final_sw.experiment_id, json.dumps(asdict(final_prom), sort_keys=True))
        if key in prom_cache:
            pro, weights = prom_cache[key]
        else:
            pro, weights = build_promethee_scores(panel, pred, selected, final_prom)
        hp = build_hybrid_panel(pred, pro)
    final_hybrid = str(final["hybrid_score_name"])
    final_topk = int(final["top_k"])
    holdings, returns, metrics = evaluate_topk(hp, final_hybrid, final_topk)
    rating, rating_perf = rating_outputs(hp, final_hybrid)

    log.to_csv(OUT / "optimization_experiment_log.csv", index=False, encoding="utf-8-sig")
    comparison = log.sort_values("valid_score", ascending=False).head(10)
    comparison.to_csv(OUT / "final_algorithm_comparison_summary.csv", index=False, encoding="utf-8-sig")
    config = {
        **asdict(final_sw),
        **asdict(final_prom),
        "hybrid_score_name": final_hybrid,
        "top_k": final_topk,
        "selection_basis": "max valid_score among valid-positive Top20/Top30 candidates; test metrics not used for selection",
    }
    (OUT / "final_selected_algorithm_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    selected.to_csv(OUT / "stagewise_selected_features_by_quarter.csv", index=False, encoding="utf-8-sig")
    feature_frequency(selected, composite).to_csv(OUT / "stagewise_feature_frequency.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(OUT / "stagewise_prediction_panel.csv", index=False, encoding="utf-8-sig")
    rankic.to_csv(OUT / "stagewise_quarterly_rankic.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "stagewise_model_summary.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(OUT / "entropy_weights_stagewise.csv", index=False, encoding="utf-8-sig")
    pro.to_csv(OUT / "promethee_scores_stagewise.csv", index=False, encoding="utf-8-sig")
    hp.to_csv(OUT / "hybrid_scores_stagewise.csv", index=False, encoding="utf-8-sig")
    returns.to_csv(OUT / "topk_quarterly_returns_stagewise.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "topk_performance_metrics_stagewise.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(OUT / "topk_holdings_stagewise.csv", index=False, encoding="utf-8-sig")
    rating[["quarter_idx", "split", "stock_id", "stock_name", "rating", "rating_rank_in_quarter", "rating_score_source", final_hybrid, LABEL]].to_csv(
        OUT / "rating_panel_stagewise.csv", index=False, encoding="utf-8-sig"
    )
    rating_perf.to_csv(OUT / "rating_performance_stagewise.csv", index=False, encoding="utf-8-sig")
    plot_outputs(log, selected, rankic, returns, metrics, rating_perf)
    write_report(config, metrics, comparison)
    print("[done] final_config=" + json.dumps(config, ensure_ascii=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
