#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a long stock panel from quarterly Excel sheets and audit the schema.

Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


CORE_OUTPUTS = {
    "panel": "stock_panel.parquet",
    "schema_audit": "stock_schema_audit.csv",
    "sheet_summary": "stock_sheet_summary.csv",
    "column_mapping": "stock_column_mapping.csv",
    "feature_role_map": "feature_role_map.csv",
    "conversion_log": "stock_conversion_log.csv",
    "run_summary": "stock_panel_run_summary.json",
}


def fail(message: str) -> SystemExit:
    return SystemExit(f"[ERROR] {message}")


def resolve_workbook(path_text: str, base_dir: Path) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    basename = Path(path_text).name
    matches = sorted(path for path in base_dir.rglob(basename) if path.is_file())
    if len(matches) == 1:
        print(f"[WARN] 默认路径不存在，已按文件名找到工作簿: {matches[0]}")
        return matches[0].resolve()
    if len(matches) > 1:
        detail = "\n".join(str(path) for path in matches)
        raise fail(f"工作簿路径不存在且发现多个同名文件，请用 --stock-workbook 指定一个:\n{detail}")
    raise fail(f"工作簿不存在: {candidate}")


def parse_quarter_idx(sheet_name: str) -> int | None:
    match = re.match(r"^(\d+)季$", sheet_name)
    if not match:
        return None
    return int(match.group(1))


def normalize_columns(columns: list[Any]) -> tuple[list[str], list[dict[str, str]]]:
    normalized: list[str] = []
    rename_rows: list[dict[str, str]] = []
    seen: dict[str, int] = {}
    for idx, raw_col in enumerate(columns):
        original = "" if raw_col is None else str(raw_col).strip()
        new_col = original
        if idx == 0 and (not original or original.lower().startswith("unnamed:")):
            new_col = "row_id_raw"
        if not new_col:
            new_col = f"empty_col_{idx + 1}"

        if new_col in seen:
            seen[new_col] += 1
            deduped = f"{new_col}__dup{seen[new_col]}"
            rename_rows.append({"original_column": original, "normalized_column": deduped, "reason": "duplicate_column_name"})
            new_col = deduped
        else:
            seen[new_col] = 0

        if new_col != original:
            rename_rows.append({"original_column": original, "normalized_column": new_col, "reason": "empty_or_unnamed_first_column"})
        normalized.append(new_col)
    return normalized, rename_rows


def stringify_id_series(series: pd.Series) -> pd.Series:
    def convert(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    return series.map(convert).astype("string")


def convert_types(
    df: pd.DataFrame,
    sheet_name: str,
    id_col: str,
    name_col: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    converted = df.copy()
    conversion_rows: list[dict[str, Any]] = []
    numeric_ratio_by_col: dict[str, float] = {}

    for col in converted.columns:
        if col in {id_col, name_col}:
            converted[col] = stringify_id_series(converted[col])
            numeric_ratio_by_col[col] = 0.0
            continue

        original = converted[col]
        non_missing_mask = original.notna() & (original.astype(str).str.strip() != "")
        non_missing_count = int(non_missing_mask.sum())
        numeric = pd.to_numeric(original, errors="coerce")
        failed_mask = non_missing_mask & numeric.isna()
        failed_count = int(failed_mask.sum())
        success_count = non_missing_count - failed_count
        numeric_ratio = success_count / non_missing_count if non_missing_count else 0.0
        numeric_ratio_by_col[col] = numeric_ratio

        if failed_count:
            examples = original[failed_mask].astype(str).head(5).tolist()
            conversion_rows.append(
                {
                    "sheet_name": sheet_name,
                    "column": col,
                    "attempted_type": "numeric",
                    "non_missing_count": non_missing_count,
                    "failed_count": failed_count,
                    "success_count": success_count,
                    "numeric_success_ratio": round(numeric_ratio, 6),
                    "examples": " | ".join(examples),
                    "action": "kept_original_object" if success_count == 0 else "converted_with_failed_values_as_na",
                }
            )

        if success_count > 0:
            converted[col] = numeric
        else:
            converted[col] = original.astype("string")

    return converted, conversion_rows, numeric_ratio_by_col


def missing_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    empty_string = series.astype(str).str.strip() == ""
    return float((series.isna() | empty_string).mean())


def infer_role(col: str, id_col: str, name_col: str, label_col: str) -> str:
    if col in {id_col, "security_code", "row_id_raw"}:
        return "id"
    if col in {name_col, "sheet_name", "quarter_idx"}:
        return "meta"
    if col == label_col:
        return "label"
    lowered = col.lower()
    return_like_tokens = ["平均收益率", "下一期收益", "未来收益", "return"]
    if any(token in col for token in return_like_tokens) or "return" in lowered or lowered in {"ret", "next_ret"}:
        return "return_like"
    return "factor"


def make_safe_column_mapping(columns: list[str], id_col: str, name_col: str, label_col: str) -> list[dict[str, str]]:
    mapping: list[dict[str, str]] = []
    used: set[str] = set()
    factor_idx = 1

    fixed = {
        id_col: "security_code",
        name_col: "security_name",
        label_col: "label_return",
        "quarter_idx": "quarter_idx",
        "sheet_name": "sheet_name",
        "row_id_raw": "row_id_raw",
    }
    for col in columns:
        role = infer_role(col, id_col, name_col, label_col)
        safe_col = fixed.get(col)
        if not safe_col:
            prefix = "return_like" if role == "return_like" else "factor"
            safe_col = f"{prefix}_{factor_idx:03d}"
            factor_idx += 1
        original_safe_col = safe_col
        suffix = 2
        while safe_col in used:
            safe_col = f"{original_safe_col}_{suffix}"
            suffix += 1
        used.add(safe_col)
        mapping.append({"original_column": col, "safe_column": safe_col, "role": role})
    return mapping


def compare_schema(sheet_columns: dict[str, list[str]]) -> list[dict[str, Any]]:
    if not sheet_columns:
        return []
    baseline_sheet = sorted(sheet_columns)[0]
    baseline_cols = sheet_columns[baseline_sheet]
    baseline_set = set(baseline_cols)
    rows: list[dict[str, Any]] = []
    for sheet, cols in sorted(sheet_columns.items(), key=lambda item: parse_quarter_idx(item[0]) or 0):
        current_set = set(cols)
        missing = [col for col in baseline_cols if col not in current_set]
        extra = [col for col in cols if col not in baseline_set]
        order_drift = not missing and not extra and cols != baseline_cols
        rows.append(
            {
                "record_type": "schema_drift",
                "sheet_name": sheet,
                "baseline_sheet": baseline_sheet,
                "has_column_drift": bool(missing or extra or order_drift),
                "missing_columns_vs_baseline": "；".join(missing),
                "extra_columns_vs_baseline": "；".join(extra),
                "order_drift": order_drift,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_panel(args: argparse.Namespace) -> int:
    base_dir = Path.cwd()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = resolve_workbook(args.stock_workbook, base_dir)
    try:
        excel = pd.ExcelFile(workbook_path, engine="openpyxl")
    except ImportError as exc:
        raise fail("读取 xlsx 需要安装 pandas 和 openpyxl。") from exc
    except Exception as exc:
        raise fail(f"无法读取工作簿: {workbook_path}；原因: {exc}") from exc

    pattern = re.compile(args.sheet_pattern)
    sheet_names = [sheet for sheet in excel.sheet_names if pattern.match(sheet)]
    sheet_names = sorted(sheet_names, key=lambda sheet: parse_quarter_idx(sheet) or 10**9)
    if len(sheet_names) < args.min_quarter_sheets:
        raise fail(f"至少应识别 {args.min_quarter_sheets} 个季度 sheet，实际识别 {len(sheet_names)} 个: {sheet_names}")

    panel_frames: list[pd.DataFrame] = []
    sheet_summary_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    conversion_rows: list[dict[str, Any]] = []
    sheet_columns: dict[str, list[str]] = {}
    rename_rows: list[dict[str, str]] = []

    for sheet_name in sheet_names:
        quarter_idx = parse_quarter_idx(sheet_name)
        if quarter_idx is None:
            raise fail(f"无法从季度 sheet 提取 quarter_idx: {sheet_name}")

        raw_df = pd.read_excel(workbook_path, sheet_name=sheet_name, engine="openpyxl", dtype=object)
        normalized_columns, sheet_renames = normalize_columns(list(raw_df.columns))
        for row in sheet_renames:
            rename_rows.append({"sheet_name": sheet_name, **row})
        raw_df.columns = normalized_columns

        required_cols = [args.id_col, args.name_col, args.label_col]
        missing_required = [col for col in required_cols if col not in raw_df.columns]
        if missing_required:
            raise fail(f"{sheet_name} 缺少核心字段: {', '.join(missing_required)}")

        typed_df, sheet_conversion_rows, numeric_ratio_by_col = convert_types(raw_df, sheet_name, args.id_col, args.name_col)
        conversion_rows.extend(sheet_conversion_rows)
        typed_df.insert(0, "quarter_idx", quarter_idx)
        typed_df.insert(1, "sheet_name", sheet_name)

        duplicate_id_count = int(typed_df[args.id_col].duplicated(keep=False).sum())
        unique_security_count = int(typed_df[args.id_col].replace("", pd.NA).dropna().nunique())
        sheet_columns[sheet_name] = list(typed_df.columns)
        sheet_summary_rows.append(
            {
                "sheet_name": sheet_name,
                "quarter_idx": quarter_idx,
                "row_count": len(typed_df),
                "column_count": len(typed_df.columns),
                "unique_security_count": unique_security_count,
                "duplicate_security_code_count": duplicate_id_count,
                "has_duplicate_security_code": duplicate_id_count > 0,
            }
        )

        for col in typed_df.columns:
            audit_rows.append(
                {
                    "record_type": "column_profile",
                    "sheet_name": sheet_name,
                    "quarter_idx": quarter_idx,
                    "column": col,
                    "row_count": len(typed_df),
                    "missing_rate": round(missing_rate(typed_df[col]), 6),
                    "numeric_value_ratio": round(numeric_ratio_by_col.get(col, 1.0 if col == "quarter_idx" else 0.0), 6),
                    "duplicate_security_code_count": duplicate_id_count,
                    "has_duplicate_security_code": duplicate_id_count > 0,
                }
            )

        panel_frames.append(typed_df)

    schema_drift_rows = compare_schema(sheet_columns)
    audit_rows.extend(schema_drift_rows)
    for row in rename_rows:
        audit_rows.append({"record_type": "column_rename", **row})
    for row in conversion_rows:
        audit_rows.append({"record_type": "conversion_issue", **row})

    panel = pd.concat(panel_frames, ignore_index=True, sort=False)
    core_missing = [col for col in [args.id_col, args.name_col, "quarter_idx", args.label_col] if col not in panel.columns]
    if core_missing:
        raise fail(f"输出长表缺少核心字段: {', '.join(core_missing)}")

    all_columns = list(panel.columns)
    mapping_rows = make_safe_column_mapping(all_columns, args.id_col, args.name_col, args.label_col)
    role_rows = [
        {
            "original_column": row["original_column"],
            "safe_column": row["safe_column"],
            "role": row["role"],
            "downstream_usage": role_downstream_usage(row["role"]),
        }
        for row in mapping_rows
    ]

    panel_path = output_dir / CORE_OUTPUTS["panel"]
    try:
        panel.to_parquet(panel_path, index=False)
    except ImportError as exc:
        raise fail("写入 parquet 需要安装 pyarrow 或 fastparquet。建议: run.bat -m pip install pyarrow") from exc
    except Exception as exc:
        raise fail(f"写入 parquet 失败: {panel_path}；原因: {exc}") from exc

    write_csv(
        output_dir / CORE_OUTPUTS["sheet_summary"],
        sheet_summary_rows,
        ["sheet_name", "quarter_idx", "row_count", "column_count", "unique_security_count", "duplicate_security_code_count", "has_duplicate_security_code"],
    )
    write_csv(
        output_dir / CORE_OUTPUTS["schema_audit"],
        audit_rows,
        [
            "record_type",
            "sheet_name",
            "quarter_idx",
            "column",
            "row_count",
            "missing_rate",
            "numeric_value_ratio",
            "duplicate_security_code_count",
            "has_duplicate_security_code",
            "baseline_sheet",
            "has_column_drift",
            "missing_columns_vs_baseline",
            "extra_columns_vs_baseline",
            "order_drift",
            "original_column",
            "normalized_column",
            "reason",
            "attempted_type",
            "non_missing_count",
            "failed_count",
            "success_count",
            "numeric_success_ratio",
            "examples",
            "action",
        ],
    )
    write_csv(output_dir / CORE_OUTPUTS["column_mapping"], mapping_rows, ["original_column", "safe_column", "role"])
    write_csv(output_dir / CORE_OUTPUTS["feature_role_map"], role_rows, ["original_column", "safe_column", "role", "downstream_usage"])
    write_csv(
        output_dir / CORE_OUTPUTS["conversion_log"],
        conversion_rows,
        ["sheet_name", "column", "attempted_type", "non_missing_count", "failed_count", "success_count", "numeric_success_ratio", "examples", "action"],
    )

    drift_count = sum(1 for row in schema_drift_rows if row["has_column_drift"])
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workbook_path": str(workbook_path),
        "sheet_count": len(sheet_names),
        "sheet_names": sheet_names,
        "panel_rows": int(len(panel)),
        "panel_columns": int(len(panel.columns)),
        "schema_drift_sheet_count": drift_count,
        "conversion_issue_count": len(conversion_rows),
        "outputs": {key: str(output_dir / filename) for key, filename in CORE_OUTPUTS.items()},
    }
    (output_dir / CORE_OUTPUTS["run_summary"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] 识别季度 sheet: {len(sheet_names)}")
    print(f"[OK] 长表行数: {len(panel)}；列数: {len(panel.columns)}")
    print(f"[OK] 已生成: {panel_path}")
    print(f"[OK] 已生成: {output_dir / CORE_OUTPUTS['schema_audit']}")
    print(f"[OK] 已生成: {output_dir / CORE_OUTPUTS['sheet_summary']}")
    print(f"[OK] 已生成: {output_dir / CORE_OUTPUTS['column_mapping']}")
    print(f"[OK] 已生成: {output_dir / CORE_OUTPUTS['feature_role_map']}")
    if drift_count:
        print(f"[WARN] 发现 {drift_count} 个 sheet 存在列名漂移，差异已写入 stock_schema_audit.csv")
    if conversion_rows:
        print(f"[WARN] 发现 {len(conversion_rows)} 个列转换问题，明细已写入 stock_conversion_log.csv")
    return 0


def role_downstream_usage(role: str) -> str:
    return {
        "id": "样本唯一性、股票级 join 和重复检查",
        "meta": "分组、展示、审计和时间面板索引",
        "factor": "候选解释变量、变量选择和综合评价",
        "return_like": "收益分析、候选标签或回测评估",
        "label": "监督学习标签和下一期收益匹配检查",
    }.get(role, "待确认用途")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把季度分表个股数据拼接成长表面板，并生成 schema 审计报告。")
    parser.add_argument("--stock-workbook", default="原始数据/因子汇总+月平均收益率(各年份独立)24个因子.xlsx", help="个股因子工作簿路径")
    parser.add_argument("--sheet-pattern", default=r"^\d+季$", help="季度 sheet 名称正则")
    parser.add_argument("--id-col", default="证券代码", help="证券代码列")
    parser.add_argument("--name-col", default="证券名称", help="证券名称列")
    parser.add_argument("--label-col", default="月平均收益率", help="标签列")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--min-quarter-sheets", type=int, default=30, help="最少季度 sheet 数验收阈值")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return build_panel(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("[ERROR] 用户中断")
    except BrokenPipeError:
        sys.stderr.close()
