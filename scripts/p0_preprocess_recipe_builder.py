#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a baseline preprocessing recipe for missing values, outliers, and scaling.

The fitted statistics are learned only from the training window. The default
baseline follows the confirmed project assumptions:
- missing_value_policy: median_by_period
- outlier_policy: winsorize_by_period_1pct_99pct
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler


RETURN_LIKE_TOKENS = ("平均收益率", "区间涨跌幅", "下一期收益", "未来收益", "future_return", "return", "ret")
POSITIVE_SCALE_TOKENS = ("总市值", "流通市值")


def fail(message: str) -> SystemExit:
    return SystemExit(f"[ERROR] {message}")


def resolve_path(path_text: str, base_dir: Path, must_exist: bool = True) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise fail(f"路径不存在: {path}")
    return path


def read_project_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    config: dict[str, Any] = {}
    current_key = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if raw.startswith("  ") and stripped.endswith(":"):
            current_key = stripped[:-1]
            config.setdefault(current_key, {})
        elif current_key and raw.startswith("    ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            value = value.strip().strip('"').strip("'")
            if value.lower() == "true":
                parsed: Any = True
            elif value.lower() == "false":
                parsed = False
            else:
                parsed = value
            config[current_key][key] = parsed
    return config


def train_mask_by_time(panel: pd.DataFrame, quarter_col: str, train_ratio: float) -> pd.Series:
    if quarter_col not in panel.columns:
        raise fail(f"缺少 {quarter_col}，无法按时间窗口拟合预处理参数。")
    quarters = sorted(pd.to_numeric(panel[quarter_col], errors="coerce").dropna().unique().tolist())
    if len(quarters) < 3:
        raise fail(f"{quarter_col} 数量过少，无法划分训练窗口: {quarters}")
    train_count = max(1, int(len(quarters) * train_ratio))
    train_quarters = set(quarters[:train_count])
    return panel[quarter_col].isin(train_quarters)


def role_map(roles: pd.DataFrame) -> dict[str, str]:
    required = {"original_column", "role"}
    missing = required - set(roles.columns)
    if missing:
        raise fail(f"feature_role_map 缺少字段: {missing}")
    return dict(zip(roles["original_column"].astype(str), roles["role"].astype(str)))


def is_return_like(col: str, role: str) -> bool:
    return role == "return_like" or any(token.lower() in col.lower() for token in RETURN_LIKE_TOKENS)


def numeric_stats(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    valid = numeric.dropna()
    if valid.empty:
        return {
            "missing_rate": float(series.isna().mean()),
            "q01": None,
            "q05": None,
            "q50": None,
            "q95": None,
            "q99": None,
            "mean": None,
            "std": None,
            "skew": None,
            "kurtosis": None,
            "zero_rate": None,
            "negative_rate": None,
        }
    return {
        "missing_rate": float(numeric.isna().mean()),
        "q01": float(valid.quantile(0.01)),
        "q05": float(valid.quantile(0.05)),
        "q50": float(valid.quantile(0.50)),
        "q95": float(valid.quantile(0.95)),
        "q99": float(valid.quantile(0.99)),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
        "skew": float(valid.skew()) if len(valid) > 2 else 0.0,
        "kurtosis": float(valid.kurtosis()) if len(valid) > 3 else 0.0,
        "zero_rate": float((valid == 0).mean()),
        "negative_rate": float((valid < 0).mean()),
    }


def choose_strategies(
    col: str,
    role: str,
    stats: dict[str, Any],
    config: dict[str, Any],
    allow_return_like: bool,
    disabled_features: set[str],
) -> dict[str, Any]:
    if role in {"id", "meta"}:
        return {
            "feature_action": "preserve_unmodified",
            "missing_strategy": "none",
            "outlier_strategy": "none",
            "scaling_strategy": "none",
            "transform_strategy": "none",
            "candidate_missing_strategies": "none",
            "candidate_scaling_strategies": "none",
            "reason": "id/meta 字段只用于索引、展示或审计，不参与数值治理。",
            "risk_flags": "",
        }
    if role == "label" or col in disabled_features:
        return {
            "feature_action": "exclude_from_features",
            "missing_strategy": "not_applicable_excluded_by_label_or_leakage_guard",
            "outlier_strategy": "not_applicable_excluded_by_label_or_leakage_guard",
            "scaling_strategy": "not_applicable_excluded_by_label_or_leakage_guard",
            "transform_strategy": "not_applicable_excluded_by_label_or_leakage_guard",
            "candidate_missing_strategies": "none",
            "candidate_scaling_strategies": "none",
            "reason": "标签或防泄漏配置禁用字段，不参与特征变换。",
            "risk_flags": "leakage_guard_or_label",
        }
    if is_return_like(col, role) and not allow_return_like:
        return {
            "feature_action": "exclude_from_features",
            "missing_strategy": "not_applicable_return_like_excluded_by_policy",
            "outlier_strategy": "not_applicable_return_like_excluded_by_policy",
            "scaling_strategy": "not_applicable_return_like_excluded_by_policy",
            "transform_strategy": "not_applicable_return_like_excluded_by_policy",
            "candidate_missing_strategies": "simple_median;add_indicator",
            "candidate_scaling_strategies": "robust;quantile",
            "reason": "return-like 字段默认不进入特征，避免与未来收益标签泄漏或混淆。",
            "risk_flags": "return_like_disabled",
        }

    skew = abs(stats.get("skew") or 0.0)
    kurtosis = stats.get("kurtosis") or 0.0
    missing = stats.get("missing_rate") or 0.0
    neg_rate = stats.get("negative_rate") or 0.0
    zero_rate = stats.get("zero_rate") or 0.0
    risk_flags = []
    if missing > 0.40:
        risk_flags.append("missing_too_high")
    if skew > 5:
        risk_flags.append("extreme_skew")
    elif skew > 2:
        risk_flags.append("high_skew")
    if kurtosis > 20:
        risk_flags.append("heavy_tail")
    if neg_rate > 0 and any(token in col for token in POSITIVE_SCALE_TOKENS):
        risk_flags.append("negative_value_in_positive_scale_field")
    if neg_rate > 0 and "市盈率PE" in col:
        risk_flags.append("negative_pe")
    if neg_rate > 0 and "市现率PCF" in col:
        risk_flags.append("negative_pcf")

    missing_policy = config.get("missing_value_policy", {}).get("value", "median_by_period")
    outlier_policy = config.get("outlier_policy", {}).get("value", "winsorize_by_period_1pct_99pct")
    if missing_policy == "median_by_period":
        missing_strategy = "SimpleImputer(median), fit on train rows within each quarter when available"
    elif missing_policy in {"mean", "median", "most_frequent", "constant"}:
        missing_strategy = f"SimpleImputer({missing_policy})"
    else:
        missing_strategy = str(missing_policy)

    transform_strategy = "none"
    scaling_strategy = "StandardScaler"
    if skew > 2 or kurtosis > 10:
        scaling_strategy = "RobustScaler"
        transform_strategy = "signed_log1p" if neg_rate > 0 else "log1p"
    if skew > 5 or kurtosis > 20:
        scaling_strategy = "QuantileTransformer(candidate) / RobustScaler(baseline)"
        transform_strategy = "signed_log1p"
    if zero_rate > 0.25 and neg_rate == 0 and skew > 2:
        transform_strategy = "log1p"

    return {
        "feature_action": "transform_as_factor",
        "missing_strategy": missing_strategy,
        "outlier_strategy": outlier_policy,
        "scaling_strategy": scaling_strategy,
        "transform_strategy": transform_strategy,
        "candidate_missing_strategies": "drop;SimpleImputer(mean/median/most_frequent/constant);KNNImputer;IterativeImputer;MissingIndicator;forward_fill_by_security",
        "candidate_scaling_strategies": "StandardScaler;MinMaxScaler;RobustScaler;QuantileTransformer;PowerTransformer;log1p;signed_log1p",
        "reason": "factor 字段作为候选解释变量；缺失和缩尾参数只在训练窗口估计。长尾/偏态字段优先建议 Robust/Quantile/Power。",
        "risk_flags": ";".join(risk_flags),
    }


def fit_quarter_medians(panel: pd.DataFrame, train_mask: pd.Series, quarter_col: str, cols: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    train = panel.loc[train_mask]
    for col in cols:
        per_q = train.groupby(quarter_col)[col].median(numeric_only=True).dropna()
        global_median = pd.to_numeric(train[col], errors="coerce").median()
        stats[col] = {
            "global": float(global_median) if pd.notna(global_median) else 0.0,
            "by_quarter": {str(k): float(v) for k, v in per_q.items()},
        }
    return stats


def fit_winsor_bounds(panel: pd.DataFrame, train_mask: pd.Series, quarter_col: str, cols: list[str]) -> dict[str, dict[str, Any]]:
    bounds: dict[str, dict[str, Any]] = {}
    train = panel.loc[train_mask]
    for col in cols:
        numeric = pd.to_numeric(train[col], errors="coerce")
        global_low = numeric.quantile(0.01)
        global_high = numeric.quantile(0.99)
        by_q: dict[str, dict[str, float]] = {}
        for quarter, group in train.groupby(quarter_col):
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                continue
            by_q[str(quarter)] = {
                "low": float(values.quantile(0.01)),
                "high": float(values.quantile(0.99)),
            }
        bounds[col] = {
            "global": {
                "low": float(global_low) if pd.notna(global_low) else None,
                "high": float(global_high) if pd.notna(global_high) else None,
            },
            "by_quarter": by_q,
        }
    return bounds


def impute_by_quarter(df: pd.DataFrame, quarter_col: str, cols: list[str], medians: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        info = medians[col]
        global_value = info["global"]
        by_q = info["by_quarter"]
        for quarter, idx in out.groupby(quarter_col).groups.items():
            fill_value = by_q.get(str(quarter), global_value)
            out.loc[idx, col] = pd.to_numeric(out.loc[idx, col], errors="coerce").fillna(fill_value)
    return out


def winsorize_by_quarter(df: pd.DataFrame, quarter_col: str, cols: list[str], bounds: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    log_rows = []
    for col in cols:
        info = bounds[col]
        for quarter, idx in out.groupby(quarter_col).groups.items():
            q_bounds = info["by_quarter"].get(str(quarter), info["global"])
            low = q_bounds.get("low")
            high = q_bounds.get("high")
            if low is None or high is None:
                continue
            values = pd.to_numeric(out.loc[idx, col], errors="coerce")
            below = int((values < low).sum())
            above = int((values > high).sum())
            out.loc[idx, col] = values.clip(lower=low, upper=high)
            if below or above:
                log_rows.append({"column": col, "quarter_idx": quarter, "below_low_count": below, "above_high_count": above, "low": low, "high": high})
    return out, pd.DataFrame(log_rows)


def apply_transform(df: pd.DataFrame, cols: list[str], strategy_by_col: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        strategy = strategy_by_col.get(col, "none")
        values = pd.to_numeric(out[col], errors="coerce")
        if strategy == "log1p":
            min_value = values.min(skipna=True)
            if pd.notna(min_value) and min_value < -1:
                out[col] = np.sign(values) * np.log1p(np.abs(values))
            else:
                out[col] = np.log1p(values.clip(lower=-0.999999))
        elif strategy == "signed_log1p":
            out[col] = np.sign(values) * np.log1p(np.abs(values))
        else:
            out[col] = values
    return out


def fit_and_apply_scalers(
    df: pd.DataFrame,
    train_mask: pd.Series,
    cols: list[str],
    strategy_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    params: dict[str, Any] = {}
    for col in cols:
        strategy = strategy_table.loc[strategy_table["column"] == col, "scaling_strategy"].iloc[0]
        numeric = pd.to_numeric(out[[col]][col], errors="coerce").to_frame()
        train_values = numeric.loc[train_mask]
        if train_values[col].nunique(dropna=True) <= 1:
            out[col] = 0.0
            params[col] = {"strategy": "constant_to_zero", "reason": "训练窗口内常数列"}
            continue
        scaler_name = "standard"
        scaler: Any = StandardScaler()
        if "RobustScaler" in strategy:
            scaler_name = "robust"
            scaler = RobustScaler()
        elif "MinMaxScaler" in strategy:
            scaler_name = "minmax"
            scaler = MinMaxScaler()
        elif "PowerTransformer" in strategy:
            scaler_name = "power_yeo_johnson"
            scaler = PowerTransformer(method="yeo-johnson", standardize=True)
        elif "QuantileTransformer" in strategy:
            scaler_name = "quantile_normal"
            n_quantiles = max(10, min(1000, len(train_values)))
            scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=2026)
        scaler.fit(train_values)
        out[col] = scaler.transform(numeric).reshape(-1)
        params[col] = {
            "strategy": scaler_name,
            "fit_rows": int(train_values.shape[0]),
            "fit_window": "train_only",
        }
    return out, params


def summarize_before_after(before: pd.DataFrame, after: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        b = numeric_stats(before[col])
        a = numeric_stats(after[col])
        rows.append(
            {
                "column": col,
                "missing_rate_before": b["missing_rate"],
                "missing_rate_after": a["missing_rate"],
                "q01_before": b["q01"],
                "q01_after": a["q01"],
                "q50_before": b["q50"],
                "q50_after": a["q50"],
                "q99_before": b["q99"],
                "q99_after": a["q99"],
                "skew_before": b["skew"],
                "skew_after": a["skew"],
                "kurtosis_before": b["kurtosis"],
                "kurtosis_after": a["kurtosis"],
                "negative_rate_before": b["negative_rate"],
                "negative_rate_after": a["negative_rate"],
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    base_dir = Path.cwd()
    panel_path = resolve_path(args.model_ready_panel_path, base_dir)
    role_path = resolve_path(args.feature_role_map, base_dir)
    config_path = resolve_path(args.config_yaml, base_dir, must_exist=False)
    output_dir = resolve_path(args.output_dir, base_dir, must_exist=False)
    log_dir = output_dir / "preprocess_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_path)
    roles_df = pd.read_csv(role_path, encoding="utf-8-sig")
    roles = role_map(roles_df)
    config = read_project_config(config_path)
    guard_path = output_dir / "leakage_guard_config.json"
    disabled_features: set[str] = set()
    guard: dict[str, Any] = {}
    if guard_path.exists():
        guard = json.loads(guard_path.read_text(encoding="utf-8"))
        disabled_features = set(guard.get("disabled_feature_names", []))

    if args.quarter_col not in panel.columns:
        raise fail(f"面板缺少 {args.quarter_col}")
    train_mask = train_mask_by_time(panel, args.quarter_col, args.train_ratio)
    panel["_preprocess_split"] = np.where(train_mask, "train", "non_train")

    strategy_rows = []
    factor_cols = []
    for col in panel.columns:
        role = roles.get(col, "meta")
        stats = numeric_stats(panel[col]) if pd.api.types.is_numeric_dtype(panel[col]) else {"missing_rate": float(panel[col].isna().mean())}
        strategy = choose_strategies(col, role, stats, config, args.allow_return_like_features, disabled_features)
        row = {
            "column": col,
            "role": role,
            "dtype": str(panel[col].dtype),
            "feature_action": strategy["feature_action"],
            "missing_rate": stats.get("missing_rate"),
            "skew": stats.get("skew"),
            "kurtosis": stats.get("kurtosis"),
            "negative_rate": stats.get("negative_rate"),
            "zero_rate": stats.get("zero_rate"),
            **strategy,
        }
        strategy_rows.append(row)
        if strategy["feature_action"] == "transform_as_factor" and pd.api.types.is_numeric_dtype(panel[col]):
            factor_cols.append(col)

    strategy_table = pd.DataFrame(strategy_rows)
    if not factor_cols:
        raise fail("没有可治理的 factor 数值字段。")

    before = panel.copy()
    medians = fit_quarter_medians(panel, train_mask, args.quarter_col, factor_cols)
    winsor_bounds = fit_winsor_bounds(panel, train_mask, args.quarter_col, factor_cols)
    processed = impute_by_quarter(panel, args.quarter_col, factor_cols, medians)
    processed, winsor_log = winsorize_by_quarter(processed, args.quarter_col, factor_cols, winsor_bounds)
    transform_by_col = dict(zip(strategy_table["column"], strategy_table["transform_strategy"]))
    processed = apply_transform(processed, factor_cols, transform_by_col)
    processed, scaler_params = fit_and_apply_scalers(processed, train_mask, factor_cols, strategy_table)

    summary = summarize_before_after(before, processed, factor_cols)
    high_risk = strategy_table.loc[strategy_table["risk_flags"].fillna("") != "", ["column", "role", "risk_flags", "reason"]].copy()

    strategy_path = output_dir / "preprocess_strategy_table.csv"
    preprocessed_path = output_dir / "preprocessed_panel_baseline.parquet"
    config_out_path = output_dir / "preprocess_config.json"
    summary_path = log_dir / "missing_outlier_scaling_summary.csv"
    winsor_path = log_dir / "winsorization_log.csv"
    high_risk_path = log_dir / "high_risk_fields.csv"

    strategy_table.to_csv(strategy_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    winsor_log.to_csv(winsor_path, index=False, encoding="utf-8-sig")
    high_risk.to_csv(high_risk_path, index=False, encoding="utf-8-sig")
    processed.to_parquet(preprocessed_path, index=False)

    preprocess_config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_panel": str(panel_path),
        "feature_role_map": str(role_path),
        "project_config": str(config_path),
        "leakage_guard_config": str(guard_path) if guard_path.exists() else None,
        "fit_policy": "all imputation, winsorization, and scaling parameters are fitted on train window only",
        "quarter_col": args.quarter_col,
        "train_ratio": args.train_ratio,
        "train_quarters": sorted(pd.to_numeric(panel.loc[train_mask, args.quarter_col], errors="coerce").dropna().unique().tolist()),
        "factor_columns": factor_cols,
        "disabled_features": sorted(disabled_features),
        "missing_parameters": medians,
        "winsor_bounds": winsor_bounds,
        "scaler_parameters": scaler_params,
        "supported_missing_strategies": ["drop", "SimpleImputer(mean/median/most_frequent/constant)", "KNNImputer", "IterativeImputer", "MissingIndicator/add_indicator", "forward_fill_by_security"],
        "supported_outlier_strategies": ["winsorize_by_quarter", "trim", "keep_raw_plus_outlier_flag"],
        "supported_scaling_transforms": ["StandardScaler/Z-score", "MinMaxScaler", "RobustScaler", "QuantileTransformer", "PowerTransformer", "log1p", "signed_log1p"],
    }
    config_out_path.write_text(json.dumps(preprocess_config, ensure_ascii=False, indent=2), encoding="utf-8")

    factor_strategy_count = int((strategy_table.loc[strategy_table["role"] == "factor", "missing_strategy"] != "").sum())
    print(f"[OK] factor_columns_transformed={len(factor_cols)}")
    print(f"[OK] factor_strategy_rows={factor_strategy_count}")
    print(f"[OK] high_risk_fields={len(high_risk)}")
    print(f"[OK] 已生成: {strategy_path}")
    print(f"[OK] 已生成: {preprocessed_path}")
    print(f"[OK] 已生成: {config_out_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建缺失、异常值和标准化治理策略，并输出预处理后的基线面板。")
    parser.add_argument("--model-ready-panel-path", default="outputs/model_ready_panel.parquet", help="P0-F 输出的模型就绪面板")
    parser.add_argument("--feature-role-map", default="outputs/feature_role_map.csv", help="字段角色表")
    parser.add_argument("--config-yaml", default="configs/project_assumptions.yaml", help="项目假设配置")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--quarter-col", default="quarter_idx", help="季度列")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="用于拟合治理参数的训练窗口比例")
    parser.add_argument("--allow-return-like-features", action="store_true", help="允许 return-like 字段参与特征治理")
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("[ERROR] 用户中断")
    except BrokenPipeError:
        sys.stderr.close()
