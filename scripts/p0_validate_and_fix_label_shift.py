#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate whether the stock panel label is already shifted to next-quarter return,
then create a canonical future_return_1q label and leakage guard config.
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


REPORT_COLUMNS = [
    "quarter_idx",
    "next_quarter_idx",
    "left_rows",
    "right_rows",
    "joined_rows",
    "valid_compare_rows",
    "matched_rows",
    "match_rate",
    "max_abs_error",
    "mean_abs_error",
    "missing_label_rows",
    "missing_next_return_rows",
    "duplicate_left_ids",
    "duplicate_right_ids",
    "assertion",
    "note",
]

ANOMALY_COLUMNS = [
    "quarter_idx",
    "next_quarter_idx",
    "security_id",
    "label_value_at_t",
    "next_return_value_at_t_plus_1",
    "abs_error",
    "error_type",
]


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


def require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise fail(f"{path} 缺少必要字段: {missing}")


def to_numeric_checked(series: pd.Series, col: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    failed = int(series.notna().sum() - numeric.notna().sum())
    if failed:
        print(f"[WARN] 字段 {col} 有 {failed} 个非空值无法转成数值，比较时按缺失处理。")
    return numeric


def duplicate_count(frame: pd.DataFrame, id_col: str) -> int:
    return int(frame[id_col].duplicated(keep=False).sum())


def compare_adjacent_quarters(
    panel: pd.DataFrame,
    id_col: str,
    quarter_col: str,
    label_col: str,
    ret_3m_col: str,
    match_threshold: float,
    abs_tolerance: float,
    rel_tolerance: float,
    max_anomalies: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    quarters = sorted(pd.to_numeric(panel[quarter_col], errors="coerce").dropna().unique().tolist())
    if len(quarters) < 2:
        raise fail(f"{quarter_col} 少于 2 个季度，无法验证 t -> t+1 标签前移。")

    rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []
    total_valid = 0
    total_matched = 0
    global_max_abs_error = 0.0

    for quarter in quarters:
        next_quarter = quarter + 1
        left = panel.loc[panel[quarter_col] == quarter, [id_col, label_col]].copy()
        if next_quarter not in quarters:
            rows.append(
                {
                    "quarter_idx": quarter,
                    "next_quarter_idx": next_quarter,
                    "left_rows": len(left),
                    "right_rows": 0,
                    "joined_rows": 0,
                    "valid_compare_rows": 0,
                    "matched_rows": 0,
                    "match_rate": np.nan,
                    "max_abs_error": np.nan,
                    "mean_abs_error": np.nan,
                    "missing_label_rows": int(left[label_col].isna().sum()),
                    "missing_next_return_rows": 0,
                    "duplicate_left_ids": duplicate_count(left, id_col),
                    "duplicate_right_ids": 0,
                    "assertion": "tail_without_t_plus_1",
                    "note": "尾季度没有 t+1，不能参与前移验证。",
                }
            )
            continue

        right = panel.loc[panel[quarter_col] == next_quarter, [id_col, ret_3m_col]].copy()
        left["_label_numeric"] = to_numeric_checked(left[label_col], label_col)
        right["_next_return_numeric"] = to_numeric_checked(right[ret_3m_col], ret_3m_col)
        joined = left.merge(right, on=id_col, how="left", suffixes=("_t", "_t_plus_1"))
        valid_mask = joined["_label_numeric"].notna() & joined["_next_return_numeric"].notna()
        valid = joined.loc[valid_mask].copy()
        if valid.empty:
            match_rate = np.nan
            max_abs_error = np.nan
            mean_abs_error = np.nan
            matched_rows = 0
            assertion = "no_valid_pairs"
        else:
            valid["abs_error"] = (valid["_label_numeric"] - valid["_next_return_numeric"]).abs()
            valid["matched"] = np.isclose(
                valid["_label_numeric"],
                valid["_next_return_numeric"],
                atol=abs_tolerance,
                rtol=rel_tolerance,
                equal_nan=False,
            )
            matched_rows = int(valid["matched"].sum())
            match_rate = matched_rows / len(valid)
            max_abs_error = float(valid["abs_error"].max())
            mean_abs_error = float(valid["abs_error"].mean())
            total_valid += int(len(valid))
            total_matched += matched_rows
            global_max_abs_error = max(global_max_abs_error, max_abs_error)
            assertion = "pass" if match_rate >= match_threshold else "fail"

            anomalies = valid.loc[~valid["matched"], [id_col, "_label_numeric", "_next_return_numeric", "abs_error"]]
            remaining = max(0, max_anomalies - len(anomaly_rows))
            if remaining:
                for item in anomalies.head(remaining).to_dict(orient="records"):
                    anomaly_rows.append(
                        {
                            "quarter_idx": quarter,
                            "next_quarter_idx": next_quarter,
                            "security_id": item[id_col],
                            "label_value_at_t": item["_label_numeric"],
                            "next_return_value_at_t_plus_1": item["_next_return_numeric"],
                            "abs_error": item["abs_error"],
                            "error_type": "value_mismatch",
                        }
                    )

        missing_next = int(joined["_next_return_numeric"].isna().sum()) if "_next_return_numeric" in joined else len(joined)
        rows.append(
            {
                "quarter_idx": quarter,
                "next_quarter_idx": next_quarter,
                "left_rows": int(len(left)),
                "right_rows": int(len(right)),
                "joined_rows": int(len(joined)),
                "valid_compare_rows": int(valid_mask.sum()),
                "matched_rows": matched_rows,
                "match_rate": match_rate,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "missing_label_rows": int(joined["_label_numeric"].isna().sum()) if "_label_numeric" in joined else int(left[label_col].isna().sum()),
                "missing_next_return_rows": missing_next,
                "duplicate_left_ids": duplicate_count(left, id_col),
                "duplicate_right_ids": duplicate_count(right, id_col),
                "assertion": assertion,
                "note": "",
            }
        )

    overall_rate = total_matched / total_valid if total_valid else np.nan
    summary = {
        "overall_match_rate": float(overall_rate) if pd.notna(overall_rate) else None,
        "total_valid_compare_rows": int(total_valid),
        "total_matched_rows": int(total_matched),
        "global_max_abs_error": float(global_max_abs_error) if total_valid else None,
        "comparable_quarter_count": int(sum(pd.notna(row["match_rate"]) for row in rows)),
        "quarter_pass_count": int(sum(row["assertion"] == "pass" for row in rows)),
        "tail_quarter": int(max(quarters)) if all(float(q).is_integer() for q in quarters) else max(quarters),
        "decision_threshold": match_threshold,
        "label_already_shifted": bool(pd.notna(overall_rate) and overall_rate >= match_threshold),
    }
    quarterly = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    anomalies = pd.DataFrame(anomaly_rows, columns=ANOMALY_COLUMNS)
    return quarterly, anomalies, summary


def rebuild_future_label(
    panel: pd.DataFrame,
    id_col: str,
    quarter_col: str,
    ret_3m_col: str,
) -> pd.Series:
    future_returns = panel[[id_col, quarter_col, ret_3m_col]].copy()
    future_returns[quarter_col] = pd.to_numeric(future_returns[quarter_col], errors="coerce") - 1
    future_returns = future_returns.rename(columns={ret_3m_col: "future_return_1q"})
    merged = panel[[id_col, quarter_col]].merge(future_returns, on=[id_col, quarter_col], how="left")
    return pd.to_numeric(merged["future_return_1q"], errors="coerce")


def build_model_ready_panel(
    panel: pd.DataFrame,
    summary: dict[str, Any],
    id_col: str,
    quarter_col: str,
    label_col: str,
    ret_3m_col: str,
) -> pd.DataFrame:
    out = panel.copy()
    out[quarter_col] = pd.to_numeric(out[quarter_col], errors="coerce")
    max_quarter = out[quarter_col].max()
    if summary["label_already_shifted"]:
        out["future_return_1q"] = pd.to_numeric(out[label_col], errors="coerce")
        out["label_source"] = f"{label_col}: validated as shifted t+1 return"
        out["label_validation_status"] = np.where(
            out[quarter_col] == max_quarter,
            "unverifiable_tail_quarter",
            "validated_existing_shifted_label",
        )
    else:
        out["future_return_1q"] = rebuild_future_label(out, id_col, quarter_col, ret_3m_col)
        out["label_source"] = f"rebuilt from {ret_3m_col} at t+1"
        out["label_validation_status"] = np.where(
            out["future_return_1q"].notna(),
            "rebuilt_from_next_quarter_return",
            "missing_t_plus_1_return_or_security",
        )
    out["is_tail_quarter"] = out[quarter_col] == max_quarter
    return out


def leakage_guard_config(
    summary: dict[str, Any],
    label_col: str,
    ret_3m_col: str,
    match_threshold: float,
    abs_tolerance: float,
    rel_tolerance: float,
) -> dict[str, Any]:
    disabled = ["future_return_1q", label_col]
    high_risk = []
    if summary["label_already_shifted"]:
        high_risk.append(ret_3m_col)
        disabled.append(ret_3m_col)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "label_already_shifted" if summary["label_already_shifted"] else "label_rebuilt_from_t_plus_1_return",
        "formal_label_col": "future_return_1q",
        "source_label_col": label_col,
        "ret_3m_col": ret_3m_col,
        "overall_match_rate": summary["overall_match_rate"],
        "match_threshold": match_threshold,
        "abs_tolerance": abs_tolerance,
        "rel_tolerance": rel_tolerance,
        "high_risk_leakage_fields": high_risk,
        "disabled_feature_names": sorted(set(disabled)),
        "rules": [
            "训练脚本必须使用 future_return_1q 作为正式监督标签。",
            "disabled_feature_names 中的字段不得进入模型特征矩阵。",
            "若 label_already_shifted=true，ret_3m_col 是验证标签前移的源字段，默认作为高风险泄漏字段剔除。",
            "尾季度没有 t+1 可验证收益，必须在训练/评估前按任务需要剔除或单独标记。",
        ],
    }


def write_report(
    path: Path,
    summary: dict[str, Any],
    quarterly_path: Path,
    anomalies_path: Path,
    model_ready_path: Path,
    guard_path: Path,
    label_col: str,
    ret_3m_col: str,
) -> None:
    decision = "已前移" if summary["label_already_shifted"] else "未确认前移，已重建标签"
    rebuild_note = (
        f"`future_return_1q = {label_col}`，因为整体匹配率超过阈值。"
        if summary["label_already_shifted"]
        else f"`future_return_1q` 由同一证券 `t+1` 季度的 `{ret_3m_col}` 重建。"
    )
    lines = [
        "# Label Shift Report",
        "",
        "## Overall",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- decision: `{decision}`",
        f"- overall_match_rate: `{summary['overall_match_rate']}`",
        f"- total_valid_compare_rows: `{summary['total_valid_compare_rows']}`",
        f"- total_matched_rows: `{summary['total_matched_rows']}`",
        f"- global_max_abs_error: `{summary['global_max_abs_error']}`",
        f"- match_threshold: `{summary['decision_threshold']}`",
        f"- comparable_quarter_count: `{summary['comparable_quarter_count']}`",
        f"- quarter_pass_count: `{summary['quarter_pass_count']}`",
        "",
        "## Label Construction",
        "",
        f"- formal_label_col: `future_return_1q`",
        f"- logic: {rebuild_note}",
        "- tail_quarter_handling: 尾季度没有 t+1，已用 `is_tail_quarter` 和 `label_validation_status` 标记。",
        "",
        "## Leakage Guard",
        "",
        f"- source_label_col: `{label_col}`",
        f"- checked_next_return_col: `{ret_3m_col}`",
        "- disabled/high-risk fields are written to `leakage_guard_config.json`.",
        "",
        "## Output Files",
        "",
        f"- model_ready_panel: `{model_ready_path}`",
        f"- quarterly_detail: `{quarterly_path}`",
        f"- anomaly_detail: `{anomalies_path}`",
        f"- leakage_guard_config: `{guard_path}`",
        "",
        "## Assertion",
        "",
        "- overall metric: `overall_match_rate`",
        "- quarterly metric: see `label_shift_quarterly_report.csv`",
        "- required downstream rule: `future_return_1q` is the only formal model label.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    base_dir = Path.cwd()
    stock_path = resolve_path(args.stock_panel_path, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir, must_exist=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(stock_path)
    require_columns(panel, [args.id_col, args.quarter_col, args.label_col, args.ret_3m_col], stock_path)
    panel[args.quarter_col] = pd.to_numeric(panel[args.quarter_col], errors="coerce")
    if panel[args.quarter_col].isna().any():
        raise fail(f"{args.quarter_col} 存在无法解析的季度值。")

    quarterly, anomalies, summary = compare_adjacent_quarters(
        panel=panel,
        id_col=args.id_col,
        quarter_col=args.quarter_col,
        label_col=args.label_col,
        ret_3m_col=args.ret_3m_col,
        match_threshold=args.match_threshold,
        abs_tolerance=args.abs_tolerance,
        rel_tolerance=args.rel_tolerance,
        max_anomalies=args.max_anomalies,
    )
    model_ready = build_model_ready_panel(panel, summary, args.id_col, args.quarter_col, args.label_col, args.ret_3m_col)
    guard = leakage_guard_config(summary, args.label_col, args.ret_3m_col, args.match_threshold, args.abs_tolerance, args.rel_tolerance)

    model_ready_path = output_dir / "model_ready_panel.parquet"
    quarterly_path = output_dir / "label_shift_quarterly_report.csv"
    anomalies_path = output_dir / "label_shift_anomalies.csv"
    summary_path = output_dir / "label_shift_summary.json"
    guard_path = output_dir / "leakage_guard_config.json"
    report_path = output_dir / "label_shift_report.md"

    model_ready.to_parquet(model_ready_path, index=False)
    quarterly.to_csv(quarterly_path, index=False, encoding="utf-8-sig")
    anomalies.to_csv(anomalies_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    guard_path.write_text(json.dumps(guard, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, summary, quarterly_path, anomalies_path, model_ready_path, guard_path, args.label_col, args.ret_3m_col)

    print(f"[OK] overall_match_rate={summary['overall_match_rate']}")
    print(f"[OK] decision={guard['decision']}")
    print(f"[OK] 已生成: {model_ready_path}")
    print(f"[OK] 已生成: {report_path}")
    print(f"[OK] 已生成: {guard_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证并修复下一期收益标签前移，输出防泄漏配置。")
    parser.add_argument("--stock-panel-path", default="outputs/stock_panel.parquet", help="个股长表 parquet")
    parser.add_argument("--id-col", default="证券代码", help="证券代码列")
    parser.add_argument("--quarter-col", default="quarter_idx", help="季度序号列")
    parser.add_argument("--label-col", default="月平均收益率", help="待验证标签列")
    parser.add_argument("--ret-3m-col", default="区间涨跌幅(3月)", help="t+1 季度用于匹配的 3 月收益列")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--match-threshold", type=float, default=0.99, help="判定已前移的整体匹配率阈值")
    parser.add_argument("--abs-tolerance", type=float, default=1e-8, help="数值匹配绝对容差")
    parser.add_argument("--rel-tolerance", type=float, default=1e-8, help="数值匹配相对容差")
    parser.add_argument("--max-anomalies", type=int, default=10000, help="异常样本明细最大输出行数")
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
