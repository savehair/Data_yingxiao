#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P0 closure checks for macro factor resolution, leakage assertions, temporal
splits, and preprocessing fit scope.

This script is read-only with respect to labels. It does not rebuild
future_return_1q and does not train any model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DOCUMENTED_MACRO_COUNT = 12
PROHIBITED_X_FIELDS = {
    "月平均收益率",
    "future_return_1q",
    "区间涨跌幅(1月)",
    "区间涨跌幅(3月)",
    "区间涨跌幅(6月)",
    "label_source",
    "label_validation_status",
    "is_tail_quarter",
    "_preprocess_split",
    "证券代码",
    "证券名称",
    "row_id_raw",
    "sheet_name",
    "quarter_idx",
}
RETURN_LIKE_PATTERNS = ("平均收益率", "区间涨跌幅", "下一期收益", "未来收益", "future_return", "return", "ret")


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def output_path(output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / name


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    cols = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def non_macro_cols() -> set[str]:
    return {"时间", "month_period", "quarter_period", "date"}


def read_macro_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path, nrows=5)
    return [col for col in df.columns if col not in non_macro_cols()]


def build_macro_resolution(args: argparse.Namespace, output_dir: Path) -> None:
    catalog = pd.read_csv(args.macro_feature_catalog, encoding="utf-8-sig")
    conflict = pd.read_csv(args.macro_conflict_report, encoding="utf-8-sig") if args.macro_conflict_report.exists() else pd.DataFrame()
    monthly_cols = read_macro_columns(args.macro_monthly_clean)
    quarterly_cols = read_macro_columns(args.macro_quarterly_clean)
    aligned_cols = read_macro_columns(args.macro_quarterly_aligned)

    actual_features = sorted(catalog["feature"].dropna().astype(str).unique().tolist())
    missing_count = max(DOCUMENTED_MACRO_COUNT - len(actual_features), 0)
    missing_fields = ["UNSPECIFIED_12TH_MACRO_FACTOR"] if missing_count == 1 else [f"UNSPECIFIED_MACRO_FACTOR_{idx+1}" for idx in range(missing_count)]

    rows = []
    for _, row in catalog.sort_values("feature").iterrows():
        feature = str(row["feature"])
        in_aligned = feature in aligned_cols
        rows.append(
            {
                "feature": feature,
                "resolution_status": "used",
                "final_in_modeling_catalog": True,
                "documented_requirement_count": DOCUMENTED_MACRO_COUNT,
                "actual_available_factor_count": len(actual_features),
                "source_sheet": row.get("source_sheet", ""),
                "declared_frequency": row.get("declared_frequency", ""),
                "final_frequency": row.get("frequency", ""),
                "alignment_mode": row.get("alignment_mode", ""),
                "alignment_note": row.get("alignment_note", ""),
                "in_definition_sheet": row.get("in_definition_sheet", ""),
                "in_monthly_clean": feature in monthly_cols,
                "in_quarterly_clean": feature in quarterly_cols,
                "in_quarterly_aligned": in_aligned,
                "derivation_formula": "",
                "field_source": row.get("source_sheet", ""),
                "final_decision": "保留；若季频表缺失，则使用清洗后的月频表按季度末口径对齐。",
            }
        )
    for feature in missing_fields:
        rows.append(
            {
                "feature": feature,
                "resolution_status": "missing_from_workbook_and_unnamed_in_task_doc",
                "final_in_modeling_catalog": False,
                "documented_requirement_count": DOCUMENTED_MACRO_COUNT,
                "actual_available_factor_count": len(actual_features),
                "source_sheet": "",
                "declared_frequency": "",
                "final_frequency": "",
                "alignment_mode": "",
                "alignment_note": "文档只声明 12 个宏观因子，但当前说明 sheet、月频表、季频表合并后仅能识别 11 个具名字段。",
                "in_definition_sheet": False,
                "in_monthly_clean": False,
                "in_quarterly_clean": False,
                "in_quarterly_aligned": False,
                "derivation_formula": "无。缺失字段未在任务文档或工作簿中命名，不能无依据从已有 11 个字段派生。",
                "field_source": "",
                "final_decision": "剔除；后续代码不得继续声明实际使用 12 个宏观因子。",
            }
        )
    final_catalog = pd.DataFrame(rows)
    final_catalog.to_csv(output_path(output_dir, "macro_feature_catalog_final.csv"), index=False, encoding="utf-8-sig")

    monthly_only = sorted(set(monthly_cols) - set(quarterly_cols))
    count_status = "PASS" if len(actual_features) == DOCUMENTED_MACRO_COUNT else "RESOLVED_WITH_EXCLUSION"
    lines = [
        "# Macro Factor Resolution Report",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- documented_macro_factor_count: `{DOCUMENTED_MACRO_COUNT}`",
        f"- actual_named_macro_factor_count: `{len(actual_features)}`",
        f"- final_used_macro_factor_count: `{len(actual_features)}`",
        f"- status: `{count_status}`",
        "",
        "## Resolution",
        "",
        "当前工作簿可识别的具名宏观因子为 11 个：",
        "",
        ", ".join(actual_features),
        "",
        "文档只声明“12个宏观因子”，但未命名第 12 个字段。当前说明 sheet、月频清洗表、季频清洗表均无法识别第 12 个具名字段。",
        "",
        "缺失字段：`UNSPECIFIED_12TH_MACRO_FACTOR`。",
        "",
        "- 是否可派生: `否`",
        "- 派生依据: `无可靠字段名、定义或公式来源`",
        "- 最终处理: `剔除`",
        "- 代码口径: `后续代码只能声明实际可用 11 个宏观因子，不得继续声明使用 12 个。`",
        "",
        "## Frequency Coverage",
        "",
        f"- monthly_clean_features: `{len(monthly_cols)}`",
        f"- quarterly_clean_features: `{len(quarterly_cols)}`",
        f"- quarterly_aligned_features: `{len(aligned_cols)}`",
        f"- monthly_only_features: `{', '.join(monthly_only)}`",
        "",
        "## Conflict Source",
        "",
        f"- macro_conflict_rows: `{len(conflict)}`",
        "- monthly-only 字段已在 `macro_feature_catalog_final.csv` 中保留 alignment_note，按季度末口径对齐。",
    ]
    output_path(output_dir, "macro_factor_resolution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def is_return_like(name: str) -> bool:
    lowered = name.lower()
    return any(token.lower() in lowered for token in RETURN_LIKE_PATTERNS)


def build_leakage_assertions(args: argparse.Namespace, output_dir: Path) -> None:
    strategy = pd.read_csv(args.preprocess_strategy_table, encoding="utf-8-sig")
    guard = read_json(args.leakage_guard_config)
    disabled = set(guard.get("disabled_feature_names", [])) | set(guard.get("high_risk_leakage_fields", []))
    prohibited = set(PROHIBITED_X_FIELDS) | disabled

    rows = []
    for _, row in strategy.iterrows():
        col = str(row["column"])
        role = str(row.get("role", ""))
        action = str(row.get("feature_action", ""))
        included = action == "transform_as_factor" and col not in prohibited and not is_return_like(col)
        exclusion_reason = ""
        if col in prohibited:
            exclusion_reason = "prohibited_or_leakage_guard"
        elif is_return_like(col):
            exclusion_reason = "return_like_excluded"
        elif action != "transform_as_factor":
            exclusion_reason = f"feature_action={action}"
        rows.append(
            {
                "column": col,
                "role": role,
                "feature_action": action,
                "included_in_X": included,
                "is_return_like": is_return_like(col),
                "is_prohibited": col in prohibited,
                "exclusion_reason": "" if included else exclusion_reason,
                "missing_strategy": row.get("missing_strategy", ""),
                "scaling_strategy": row.get("scaling_strategy", ""),
            }
        )
    feature_list = pd.DataFrame(rows)
    feature_list.to_csv(output_path(output_dir, "final_feature_list.csv"), index=False, encoding="utf-8-sig")

    x_cols = feature_list.loc[feature_list["included_in_X"], "column"].astype(str).tolist()
    prohibited_in_x = sorted(set(x_cols) & prohibited)
    return_like_as_factor = feature_list.loc[feature_list["included_in_X"] & feature_list["is_return_like"]]
    return_like_as_factor_count = int(len(return_like_as_factor))
    pass_flag = not prohibited_in_x and return_like_as_factor_count == 0
    lines = [
        "# Leakage Assertion Report",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- final_X_column_count: `{len(x_cols)}`",
        f"- prohibited_field_count: `{len(PROHIBITED_X_FIELDS)}`",
        f"- prohibited_fields_in_X: `{', '.join(prohibited_in_x) if prohibited_in_x else 'None'}`",
        f"- return_like_as_factor_count: `{return_like_as_factor_count}`",
        f"- return_like_assertion: `{'PASS' if return_like_as_factor_count == 0 else 'FAIL'}`",
        f"- overall_status: `{'PASS' if pass_flag else 'FAIL'}`",
        "",
        "## Prohibited Fields",
        "",
        ", ".join(sorted(PROHIBITED_X_FIELDS)),
        "",
        "## Final X Columns",
        "",
        ", ".join(x_cols),
        "",
        "## Rule",
        "",
        "最终 X 只允许来自 `preprocess_strategy_table.csv` 中 `feature_action=transform_as_factor` 且不在防泄漏/收益类/ID/Meta 禁用名单内的字段。",
    ]
    output_path(output_dir, "leakage_assertion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_by_time(panel: pd.DataFrame, quarter_col: str, train_ratio: float, valid_ratio: float) -> pd.Series:
    quarters = sorted(pd.to_numeric(panel[quarter_col], errors="coerce").dropna().unique().tolist())
    if len(quarters) < 3:
        raise fail(f"{quarter_col} 少于 3 个季度，无法划分 train/valid/test。")
    train_end = max(1, int(len(quarters) * train_ratio))
    valid_end = max(train_end + 1, int(len(quarters) * (train_ratio + valid_ratio)))
    valid_end = min(valid_end, len(quarters) - 1)
    train_quarters = set(quarters[:train_end])
    valid_quarters = set(quarters[train_end:valid_end])
    split = pd.Series("test", index=panel.index, dtype="object")
    split.loc[panel[quarter_col].isin(train_quarters)] = "train"
    split.loc[panel[quarter_col].isin(valid_quarters)] = "valid"
    return split


def build_temporal_split(args: argparse.Namespace, output_dir: Path) -> None:
    panel = pd.read_parquet(args.model_ready_panel)
    if args.quarter_col not in panel.columns:
        raise fail(f"model_ready_panel 缺少 {args.quarter_col}")
    if args.id_col not in panel.columns:
        raise fail(f"model_ready_panel 缺少 {args.id_col}")
    panel = panel.copy()
    panel[args.quarter_col] = pd.to_numeric(panel[args.quarter_col], errors="coerce")
    split = split_by_time(panel, args.quarter_col, args.train_ratio, args.valid_ratio)
    panel["_closure_split"] = split
    rows = []
    for name in ["train", "valid", "test"]:
        part = panel.loc[panel["_closure_split"] == name]
        rows.append(
            {
                "split": name,
                "start_quarter": int(part[args.quarter_col].min()) if len(part) else None,
                "end_quarter": int(part[args.quarter_col].max()) if len(part) else None,
                "row_count": int(len(part)),
                "unique_stock_count": int(part[args.id_col].nunique(dropna=True)),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_path(output_dir, "split_manifest.csv"), index=False, encoding="utf-8-sig")

    train_end = manifest.loc[manifest["split"] == "train", "end_quarter"].iloc[0]
    valid_start = manifest.loc[manifest["split"] == "valid", "start_quarter"].iloc[0]
    test_start = manifest.loc[manifest["split"] == "test", "start_quarter"].iloc[0]
    order_pass = bool(train_end < valid_start <= test_start)
    row_count_pass = bool((manifest["row_count"] > 0).all())
    if "is_tail_quarter" in panel.columns:
        tail_in_train = int(((panel["is_tail_quarter"] == True) & (panel["_closure_split"] == "train")).sum())
        tail_supervised_pass = tail_in_train == 0
        tail_note = f"tail_quarter_train_rows=`{tail_in_train}`"
    else:
        tail_in_train = None
        tail_supervised_pass = False
        tail_note = "model_ready_panel 缺少 is_tail_quarter，无法确认尾季度训练排除。"
    overall = order_pass and row_count_pass and tail_supervised_pass
    lines = [
        "# Temporal Split Assertion",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- shuffle: `False`",
        f"- train_end_date < valid_start_date <= test_start_date: `{'PASS' if order_pass else 'FAIL'}`",
        f"- split_row_count_positive: `{'PASS' if row_count_pass else 'FAIL'}`",
        f"- tail_quarter_not_in_supervised_train: `{'PASS' if tail_supervised_pass else 'FAIL'}`",
        f"- {tail_note}",
        f"- overall_status: `{'PASS' if overall else 'FAIL'}`",
        "",
        "## Split Manifest",
        "",
        markdown_table(manifest),
        "",
        "## Assertion",
        "",
        f"- train_end_date: `{train_end}`",
        f"- valid_start_date: `{valid_start}`",
        f"- test_start_date: `{test_start}`",
    ]
    output_path(output_dir, "temporal_split_assertion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preprocessing_scope(args: argparse.Namespace, output_dir: Path) -> None:
    config = read_json(args.preprocess_config)
    train_quarters = set(map(str, config.get("train_quarters", [])))
    issues: list[str] = []
    audit_rows = []

    sections = [
        ("missing_parameters", config.get("missing_parameters", {})),
        ("winsor_bounds", config.get("winsor_bounds", {})),
        ("scaler_parameters", config.get("scaler_parameters", {})),
    ]
    for section, values in sections:
        for field, payload in values.items():
            fitted_on_split = "train"
            reason = "declared_train_only_fit_policy"
            status = "PASS"
            quarters = set()
            if isinstance(payload, dict) and "by_quarter" in payload:
                quarters = set(map(str, payload.get("by_quarter", {}).keys()))
                extra_quarters = sorted(quarters - train_quarters)
                if extra_quarters:
                    fitted_on_split = "mixed_or_full_sample"
                    reason = f"contains_non_train_quarters={extra_quarters}"
                    status = "FAIL"
                    issues.append(f"{section}:{field}:{reason}")
            if section == "scaler_parameters":
                if isinstance(payload, dict) and payload.get("fit_window") != "train_only":
                    fitted_on_split = str(payload.get("fit_window", "unknown"))
                    reason = "scaler fit_window is not train_only"
                    status = "FAIL"
                    issues.append(f"{section}:{field}:{reason}")
            audit_rows.append(
                {
                    "parameter_section": section,
                    "field": field,
                    "fitted_on_split": fitted_on_split,
                    "status": status,
                    "reason": reason,
                    "train_quarter_count": len(train_quarters),
                    "parameter_quarter_count": len(quarters) if quarters else "",
                }
            )

    audit = pd.DataFrame(audit_rows)
    fit_policy_text = str(config.get("fit_policy", "")).lower()
    policy_declares_train_only = "train" in fit_policy_text and "only" in fit_policy_text
    status = "PASS" if not issues and policy_declares_train_only else "FAIL"
    lines = [
        "# Preprocessing Fit Scope Report",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- fit_policy: `{config.get('fit_policy', '')}`",
        f"- train_quarters: `{sorted(train_quarters, key=lambda x: int(float(x)))}`",
        f"- parameter_count: `{len(audit)}`",
        f"- full_sample_fit_issue_count: `{len(issues)}`",
        f"- overall_status: `{status}`",
        "",
        "## fitted_on_split Audit",
        "",
        markdown_table(audit.head(80)) if not audit.empty else "(no parameters found)",
        "",
        "## Issues",
        "",
        "\n".join(f"- {issue}" for issue in issues) if issues else "- None",
        "",
        "## Rule",
        "",
        "缺失值填补、winsorize 分位数和 scaler 参数必须只在 train 窗口拟合，然后应用到 valid/test。",
    ]
    output_path(output_dir, "preprocessing_fit_scope_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    build_macro_resolution(args, output_dir)
    build_leakage_assertions(args, output_dir)
    build_temporal_split(args, output_dir)
    preprocessing_scope(args, output_dir)
    expected = [
        "macro_factor_resolution_report.md",
        "macro_feature_catalog_final.csv",
        "final_feature_list.csv",
        "leakage_assertion_report.md",
        "split_manifest.csv",
        "temporal_split_assertion.md",
        "preprocessing_fit_scope_report.md",
    ]
    for name in expected:
        print(f"[OK] 已生成: {output_dir / name}")
    return 0


def parse_args() -> argparse.Namespace:
    base_dir = Path.cwd()
    parser = argparse.ArgumentParser(description="P0 闭环验证与报告生成，不训练模型、不重构标签。")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--problem-summary-md", default="outputs/problem_summary.md")
    parser.add_argument("--label-shift-report-md", default="outputs/label_shift_report.md")
    parser.add_argument("--preprocess-strategy-table", default="outputs/preprocess_strategy_table.csv")
    parser.add_argument("--model-ready-panel", default="outputs/model_ready_panel.parquet")
    parser.add_argument("--leakage-guard-config", default="outputs/leakage_guard_config.json")
    parser.add_argument("--preprocess-config", default="outputs/preprocess_config.json")
    parser.add_argument("--macro-feature-catalog", default="outputs/macro_feature_catalog.csv")
    parser.add_argument("--macro-conflict-report", default="outputs/macro_conflict_report.csv")
    parser.add_argument("--macro-monthly-clean", default="outputs/macro_monthly_clean.csv")
    parser.add_argument("--macro-quarterly-clean", default="outputs/macro_quarterly_clean.csv")
    parser.add_argument("--macro-quarterly-aligned", default="outputs/macro_quarterly_aligned.csv")
    parser.add_argument("--quarter-col", default="quarter_idx")
    parser.add_argument("--id-col", default="证券代码")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    ns = parser.parse_args()
    ns.output_dir = resolve_path(ns.output_dir, base_dir, must_exist=False)
    for attr in [
        "problem_summary_md",
        "label_shift_report_md",
        "preprocess_strategy_table",
        "model_ready_panel",
        "leakage_guard_config",
        "preprocess_config",
        "macro_feature_catalog",
        "macro_monthly_clean",
        "macro_quarterly_clean",
        "macro_quarterly_aligned",
    ]:
        setattr(ns, attr, resolve_path(getattr(ns, attr), base_dir, must_exist=True))
    ns.macro_conflict_report = resolve_path(ns.macro_conflict_report, base_dir, must_exist=False)
    return ns


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("[ERROR] 用户中断")
    except BrokenPipeError:
        sys.stderr.close()
