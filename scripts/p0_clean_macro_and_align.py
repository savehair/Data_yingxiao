#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean macro factor tables and build frequency-alignment audit outputs.

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


OUTPUTS = {
    "clean_monthly": "macro_monthly_clean.csv",
    "clean_quarterly": "macro_quarterly_clean.csv",
    "aligned_quarterly": "macro_quarterly_aligned.csv",
    "feature_catalog": "macro_feature_catalog.csv",
    "time_mapping_report": "macro_time_mapping_report.md",
    "conflict_report": "macro_conflict_report.csv",
    "time_parse_exceptions": "macro_time_parse_exceptions.csv",
    "run_summary": "macro_clean_run_summary.json",
}

VALID_ALIGNMENT_MODES = {"prefer_quarterly", "aggregate_monthly_to_quarter", "dual_track"}
LEVEL_TOKENS = ("dp", "de", "bm", "svar", "ep", "ntis", "tms", "infl", "mtr")
GROWTH_TOKENS = ("gr", "growth", "同比", "增速", "增长")


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
        raise fail(f"工作簿路径不存在且发现多个同名文件，请用 --macro-workbook 指定一个:\n{detail}")
    raise fail(f"工作簿不存在: {candidate}")


def read_sheet(workbook_path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(workbook_path, sheet_name=sheet_name, engine="openpyxl", dtype=object)
    except ValueError as exc:
        raise fail(f"工作簿缺少 sheet: {sheet_name}") from exc
    except Exception as exc:
        raise fail(f"读取 sheet 失败: {sheet_name}；原因: {exc}") from exc


def is_empty_time(value: Any) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def clean_macro_sheet(df: pd.DataFrame, sheet_name: str, time_col: str) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    if time_col not in df.columns:
        raise fail(f"{sheet_name} 缺少时间列: {time_col}")

    working = df.copy()
    explanation_mask = working[time_col].map(is_empty_time) & working.drop(columns=[time_col]).notna().any(axis=1)
    removed_explanation_rows = int(explanation_mask.sum())
    working = working.loc[~explanation_mask].copy()

    empty_time_mask = working[time_col].map(is_empty_time)
    empty_time_rows = working.loc[empty_time_mask].copy()
    working = working.loc[~empty_time_mask].copy()

    exceptions: list[dict[str, Any]] = []
    for idx, row in empty_time_rows.iterrows():
        exceptions.append(
            {
                "sheet_name": sheet_name,
                "row_index": int(idx),
                "raw_time": "",
                "issue": "empty_time_row_removed",
                "row_preview": "; ".join(str(v) for v in row.dropna().head(5).tolist()),
            }
        )

    parsed_periods: list[pd.Period | pd.NaT] = []
    parsed_dates: list[pd.Timestamp | pd.NaT] = []
    for idx, raw_value in working[time_col].items():
        try:
            normalized = normalize_yyyymm(raw_value)
            period = pd.Period(normalized, freq="M")
            parsed_periods.append(period)
            parsed_dates.append(period.to_timestamp())
        except Exception as exc:
            parsed_periods.append(pd.NaT)
            parsed_dates.append(pd.NaT)
            exceptions.append(
                {
                    "sheet_name": sheet_name,
                    "row_index": int(idx),
                    "raw_time": raw_value,
                    "issue": "time_parse_failed",
                    "row_preview": str(exc),
                }
            )

    working["month_period"] = [str(value) if not pd.isna(value) else "" for value in parsed_periods]
    working["quarter_period"] = [
        str(value.asfreq("Q")) if not pd.isna(value) else ""
        for value in parsed_periods
    ]
    working["date"] = parsed_dates
    working = working.loc[working["month_period"] != ""].copy()

    for col in [c for c in working.columns if c not in {time_col, "month_period", "quarter_period", "date"}]:
        numeric = pd.to_numeric(working[col], errors="coerce")
        failed_mask = working[col].notna() & (working[col].astype(str).str.strip() != "") & numeric.isna()
        if failed_mask.any():
            exceptions.append(
                {
                    "sheet_name": sheet_name,
                    "row_index": "",
                    "raw_time": "",
                    "issue": "numeric_parse_failed",
                    "row_preview": f"{col}: " + " | ".join(working.loc[failed_mask, col].astype(str).head(5).tolist()),
                }
            )
        working[col] = numeric

    return working, exceptions, removed_explanation_rows


def validate_quarter_observations(df: pd.DataFrame, sheet_name: str) -> list[dict[str, Any]]:
    exceptions: list[dict[str, Any]] = []
    valid_quarter_months = {"01", "04", "07", "10"}
    for idx, row in df.iterrows():
        month_period = str(row.get("month_period", ""))
        if len(month_period) >= 7 and month_period[5:7] not in valid_quarter_months:
            exceptions.append(
                {
                    "sheet_name": sheet_name,
                    "row_index": int(idx),
                    "raw_time": row.get("时间", ""),
                    "issue": "non_quarter_observation_in_quarter_sheet",
                    "row_preview": f"month_period={month_period}; expected month in 01/04/07/10",
                }
            )
    return exceptions


def normalize_yyyymm(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    if re.fullmatch(r"\d{6}\.0", text):
        text = text[:-2]
    if not re.fullmatch(r"\d{6}", text):
        raise ValueError(f"not YYYYMM: {value}")
    month = int(text[4:6])
    if month < 1 or month > 12:
        raise ValueError(f"invalid month in YYYYMM: {value}")
    return f"{text[:4]}-{text[4:6]}"


def classify_macro_variable(col: str, definition_row: dict[str, str] | None = None) -> str:
    lowered = col.lower()
    definition_text = ""
    if definition_row:
        definition_text = " ".join(str(v) for v in definition_row.values()).lower()
    if any(token in lowered for token in GROWTH_TOKENS) or any(token in definition_text for token in GROWTH_TOKENS):
        return "growth"
    return "level"


def aggregate_monthly_to_quarter(
    monthly_df: pd.DataFrame,
    time_col: str,
    level_agg: str,
    growth_agg: str,
    definitions: dict[str, dict[str, str]],
) -> pd.DataFrame:
    if level_agg not in {"quarter_end", "quarter_mean"}:
        raise fail("--level-agg 仅支持 quarter_end / quarter_mean")
    if growth_agg not in {"quarter_end", "quarter_mean"}:
        raise fail("--growth-agg 仅支持 quarter_end / quarter_mean")

    feature_cols = [col for col in monthly_df.columns if col not in {time_col, "month_period", "quarter_period", "date"}]
    sorted_monthly = monthly_df.sort_values("date").copy()
    grouped = sorted_monthly.groupby("quarter_period", sort=True)
    output = pd.DataFrame({"quarter_period": sorted(grouped.groups.keys())})

    for col in feature_cols:
        variable_type = classify_macro_variable(col, definitions.get(col))
        agg_mode = growth_agg if variable_type == "growth" else level_agg
        if agg_mode == "quarter_end":
            series = grouped[col].last()
        else:
            series = grouped[col].mean()
        output[col] = output["quarter_period"].map(series)

    output["date"] = output["quarter_period"].map(lambda q: pd.Period(q, freq="Q").end_time.normalize())
    return output


def read_definition_catalog(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    required_cols = ["首字母缩略词", "可变因素，变量", "定义", "更新频率"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return {}
    catalog: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        acronym = "" if pd.isna(row["首字母缩略词"]) else str(row["首字母缩略词"]).strip()
        if not acronym:
            continue
        catalog[acronym] = {
            "description": "" if pd.isna(row["可变因素，变量"]) else str(row["可变因素，变量"]).strip(),
            "definition": "" if pd.isna(row["定义"]) else str(row["定义"]).strip(),
            "declared_frequency": "" if pd.isna(row["更新频率"]) else str(row["更新频率"]).strip(),
        }
    return catalog


def build_feature_catalog(
    definitions: dict[str, dict[str, str]],
    monthly_cols: list[str],
    quarterly_cols: list[str],
    alignment_mode: str,
    level_agg: str,
    growth_agg: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_names = sorted(set(definitions) | set(monthly_cols) | set(quarterly_cols))
    catalog_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []

    for feature in feature_names:
        definition = definitions.get(feature, {})
        in_monthly = feature in monthly_cols
        in_quarterly = feature in quarterly_cols
        sources = []
        if feature in definitions:
            sources.append("宏观因子")
        if in_monthly:
            sources.append("宏观因子具体数据")
        if in_quarterly:
            sources.append("Sheet1")

        original_frequency = infer_original_frequency(definition.get("declared_frequency", ""), in_monthly, in_quarterly)
        alignment_note = make_alignment_note(feature, alignment_mode, in_monthly, in_quarterly, definitions, level_agg, growth_agg)
        catalog_rows.append(
            {
                "feature": feature,
                "source_sheet": "；".join(sources),
                "declared_frequency": definition.get("declared_frequency", ""),
                "original_frequency": original_frequency,
                "frequency": resolved_frequency(alignment_mode, in_monthly, in_quarterly),
                "alignment_mode": alignment_mode,
                "alignment_note": alignment_note,
                "description": definition.get("description", ""),
                "definition": definition.get("definition", ""),
                "in_definition_sheet": feature in definitions,
                "in_monthly_sheet": in_monthly,
                "in_quarterly_sheet": in_quarterly,
            }
        )

        issue = None
        recommendation = ""
        if feature in definitions and not (in_monthly or in_quarterly):
            issue = "defined_but_no_data_column"
            recommendation = "核对字段缩写是否改名，或从建模特征中剔除。"
        elif (in_monthly or in_quarterly) and feature not in definitions:
            issue = "data_column_missing_definition"
            recommendation = "补充说明 sheet 定义，确认经济含义和更新频率。"
        elif in_monthly and not in_quarterly:
            issue = "monthly_only"
            recommendation = "若需要季度建模，按聚合规则转季度，或保留为 dual_track。"
        elif in_quarterly and not in_monthly:
            issue = "quarterly_only"
            recommendation = "优先使用季频列；若需月频解释，补充月频来源或说明。"

        if issue:
            conflict_rows.append(
                {
                    "feature": feature,
                    "issue": issue,
                    "declared_frequency": definition.get("declared_frequency", ""),
                    "in_definition_sheet": feature in definitions,
                    "in_monthly_sheet": in_monthly,
                    "in_quarterly_sheet": in_quarterly,
                    "recommendation": recommendation,
                }
            )

    expected_factor_count = 12
    available_factor_count = len(set(monthly_cols) | set(quarterly_cols))
    if available_factor_count != expected_factor_count:
        conflict_rows.append(
            {
                "feature": "__factor_count__",
                "issue": "factor_count_mismatch",
                "declared_frequency": "",
                "in_definition_sheet": len(definitions),
                "in_monthly_sheet": len(monthly_cols),
                "in_quarterly_sheet": len(quarterly_cols),
                "recommendation": f"文档口径为 {expected_factor_count} 个宏观因子，当前数据列去重后为 {available_factor_count} 个；请确认缺失或衍生字段。",
            }
        )
    return catalog_rows, conflict_rows


def infer_original_frequency(declared: str, in_monthly: bool, in_quarterly: bool) -> str:
    if declared:
        return declared
    if in_monthly and in_quarterly:
        return "monthly_and_quarterly"
    if in_monthly:
        return "monthly"
    if in_quarterly:
        return "quarterly"
    return "unknown"


def resolved_frequency(alignment_mode: str, in_monthly: bool, in_quarterly: bool) -> str:
    if alignment_mode == "dual_track" and in_monthly and in_quarterly:
        return "monthly_and_quarterly"
    return "quarterly"


def make_alignment_note(
    feature: str,
    alignment_mode: str,
    in_monthly: bool,
    in_quarterly: bool,
    definitions: dict[str, dict[str, str]],
    level_agg: str,
    growth_agg: str,
) -> str:
    if alignment_mode == "prefer_quarterly":
        if in_quarterly:
            return "使用季频 sheet 的季度观测；与个股 quarter_idx 按季度顺序映射。"
        return f"季频 sheet 缺失，需从月频聚合；level={level_agg}, growth={growth_agg}。"
    if alignment_mode == "aggregate_monthly_to_quarter":
        variable_type = classify_macro_variable(feature, definitions.get(feature))
        agg_mode = growth_agg if variable_type == "growth" else level_agg
        return f"从月频表聚合到季度；变量类型={variable_type}；聚合规则={agg_mode}。"
    if alignment_mode == "dual_track":
        return "同时保留月频清洗表和季频/聚合季度表；建模阶段显式选择季度键或月度可得性键。"
    return "未知对齐方式。"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_dataframe_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_mapping_rows(aligned_quarterly: pd.DataFrame, output_dir: Path) -> list[dict[str, Any]]:
    quarter_periods = sorted(aligned_quarterly["quarter_period"].dropna().astype(str).unique())
    mapping_rows = [
        {"quarter_idx": idx + 1, "macro_quarter_period": quarter, "mapping_rule": "按宏观季度时间升序映射"}
        for idx, quarter in enumerate(quarter_periods)
    ]
    stock_summary_path = output_dir / "stock_sheet_summary.csv"
    if stock_summary_path.exists():
        stock_summary = pd.read_csv(stock_summary_path)
        if "quarter_idx" in stock_summary.columns and len(stock_summary):
            max_stock_idx = int(stock_summary["quarter_idx"].max())
            for row in mapping_rows:
                row["exists_in_stock_panel"] = row["quarter_idx"] <= max_stock_idx
    return mapping_rows


def write_time_mapping_report(
    path: Path,
    alignment_mode: str,
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    aligned_df: pd.DataFrame,
    mapping_rows: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Macro Time Mapping Report",
        "",
        "## 对齐方式",
        "",
        f"- alignment_mode: `{alignment_mode}`",
        "- 月频时间列按 `YYYYMM` 解析为 `month_period`，再派生 `quarter_period`。",
        "- 季频 sheet 的 `YYYYMM` 观测按所在季度解析，例如 `201601 -> 2016Q1`。",
        "- 个股 `quarter_idx` 默认按宏观季度升序映射：第 1 个季度为 `quarter_idx=1`，第 2 个季度为 `quarter_idx=2`。",
        "",
        "## 数据范围",
        "",
        f"- 月频清洗后行数: {len(monthly_df)}",
        f"- 季频清洗后行数: {len(quarterly_df)}",
        f"- 对齐后季度行数: {len(aligned_df)}",
        "",
        "## quarter_idx 映射样例",
        "",
        "| quarter_idx | macro_quarter_period | mapping_rule |",
        "|---:|---|---|",
    ]
    for row in mapping_rows[:12]:
        lines.append(f"| {row['quarter_idx']} | {row['macro_quarter_period']} | {row['mapping_rule']} |")
    lines.extend(["", "## 字段冲突摘要", ""])
    if conflicts:
        for conflict in conflicts:
            lines.append(f"- `{conflict['feature']}`: {conflict['issue']}；建议: {conflict['recommendation']}")
    else:
        lines.append("- 未发现字段冲突。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_aligned_quarterly(
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    time_col: str,
    alignment_mode: str,
    level_agg: str,
    growth_agg: str,
    definitions: dict[str, dict[str, str]],
) -> pd.DataFrame:
    aggregated = aggregate_monthly_to_quarter(monthly_df, time_col, level_agg, growth_agg, definitions)
    if alignment_mode == "aggregate_monthly_to_quarter":
        return aggregated
    if alignment_mode == "prefer_quarterly":
        quarterly_features = [col for col in quarterly_df.columns if col not in {time_col, "month_period", "quarter_period", "date"}]
        output = quarterly_df[["quarter_period", "date"] + quarterly_features].copy()
        monthly_only = [col for col in aggregated.columns if col not in set(output.columns) | {time_col}]
        if monthly_only:
            output = output.merge(aggregated[["quarter_period"] + monthly_only], on="quarter_period", how="left")
        return output
    if alignment_mode == "dual_track":
        quarterly_pref = build_aligned_quarterly(monthly_df, quarterly_df, time_col, "prefer_quarterly", level_agg, growth_agg, definitions)
        pref_cols = [col for col in quarterly_pref.columns if col not in {"quarter_period", "date"}]
        agg_cols = [col for col in aggregated.columns if col not in {"quarter_period", "date"}]
        renamed_pref = quarterly_pref.rename(columns={col: f"quarterly_{col}" for col in pref_cols})
        renamed_agg = aggregated.rename(columns={col: f"monthly_agg_{col}" for col in agg_cols})
        return renamed_pref.merge(renamed_agg, on="quarter_period", how="outer")
    raise fail(f"未知 alignment_mode: {alignment_mode}")


def run(args: argparse.Namespace) -> int:
    if args.alignment_mode not in VALID_ALIGNMENT_MODES:
        raise fail(f"--alignment-mode 仅支持: {', '.join(sorted(VALID_ALIGNMENT_MODES))}")

    base_dir = Path.cwd()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = resolve_workbook(args.macro_workbook, base_dir)
    definition_df = read_sheet(workbook_path, args.macro_definition_sheet)
    monthly_raw = read_sheet(workbook_path, args.macro_month_sheet)
    quarterly_raw = read_sheet(workbook_path, args.macro_quarter_sheet)

    definitions = read_definition_catalog(definition_df)
    monthly_clean, monthly_exceptions, monthly_removed = clean_macro_sheet(monthly_raw, args.macro_month_sheet, args.time_col)
    quarterly_clean, quarterly_exceptions, quarterly_removed = clean_macro_sheet(quarterly_raw, args.macro_quarter_sheet, args.time_col)
    quarterly_exceptions.extend(validate_quarter_observations(quarterly_clean, args.macro_quarter_sheet))
    time_exceptions = monthly_exceptions + quarterly_exceptions
    fatal_time_errors = [
        row
        for row in time_exceptions
        if row["issue"] in {"time_parse_failed", "non_quarter_observation_in_quarter_sheet"}
    ]
    if fatal_time_errors:
        write_csv(
            output_dir / OUTPUTS["time_parse_exceptions"],
            time_exceptions,
            ["sheet_name", "row_index", "raw_time", "issue", "row_preview"],
        )
        raise fail(f"存在无法解析的时间字段，详见: {output_dir / OUTPUTS['time_parse_exceptions']}")

    monthly_features = [col for col in monthly_clean.columns if col not in {args.time_col, "month_period", "quarter_period", "date"}]
    quarterly_features = [col for col in quarterly_clean.columns if col not in {args.time_col, "month_period", "quarter_period", "date"}]
    catalog_rows, conflict_rows = build_feature_catalog(
        definitions,
        monthly_features,
        quarterly_features,
        args.alignment_mode,
        args.level_agg,
        args.growth_agg,
    )
    missing_catalog_notes = [row for row in catalog_rows if not row.get("frequency") or not row.get("alignment_note")]
    if missing_catalog_notes:
        raise fail("macro_feature_catalog 中存在缺失 frequency 或 alignment_note 的字段。")

    aligned_quarterly = build_aligned_quarterly(
        monthly_clean,
        quarterly_clean,
        args.time_col,
        args.alignment_mode,
        args.level_agg,
        args.growth_agg,
        definitions,
    )

    write_dataframe_csv(output_dir / OUTPUTS["clean_monthly"], monthly_clean)
    write_dataframe_csv(output_dir / OUTPUTS["clean_quarterly"], quarterly_clean)
    write_dataframe_csv(output_dir / OUTPUTS["aligned_quarterly"], aligned_quarterly)
    write_csv(
        output_dir / OUTPUTS["feature_catalog"],
        catalog_rows,
        [
            "feature",
            "source_sheet",
            "declared_frequency",
            "original_frequency",
            "frequency",
            "alignment_mode",
            "alignment_note",
            "description",
            "definition",
            "in_definition_sheet",
            "in_monthly_sheet",
            "in_quarterly_sheet",
        ],
    )
    write_csv(
        output_dir / OUTPUTS["conflict_report"],
        conflict_rows,
        ["feature", "issue", "declared_frequency", "in_definition_sheet", "in_monthly_sheet", "in_quarterly_sheet", "recommendation"],
    )
    write_csv(
        output_dir / OUTPUTS["time_parse_exceptions"],
        time_exceptions,
        ["sheet_name", "row_index", "raw_time", "issue", "row_preview"],
    )
    mapping_rows = build_mapping_rows(aligned_quarterly, output_dir)
    write_time_mapping_report(
        output_dir / OUTPUTS["time_mapping_report"],
        args.alignment_mode,
        monthly_clean,
        quarterly_clean,
        aligned_quarterly,
        mapping_rows,
        conflict_rows,
    )
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workbook_path": str(workbook_path),
        "alignment_mode": args.alignment_mode,
        "monthly_rows": int(len(monthly_clean)),
        "quarterly_rows": int(len(quarterly_clean)),
        "aligned_quarterly_rows": int(len(aligned_quarterly)),
        "monthly_explanation_rows_removed": monthly_removed,
        "quarterly_explanation_rows_removed": quarterly_removed,
        "definition_factor_count": len(definitions),
        "monthly_factor_count": len(monthly_features),
        "quarterly_factor_count": len(quarterly_features),
        "available_factor_count": len(set(monthly_features) | set(quarterly_features)),
        "conflict_count": len(conflict_rows),
        "time_exception_count": len(time_exceptions),
        "outputs": {key: str(output_dir / filename) for key, filename in OUTPUTS.items()},
    }
    (output_dir / OUTPUTS["run_summary"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] 月频清洗后行数: {len(monthly_clean)}；删除说明行: {monthly_removed}")
    print(f"[OK] 季频清洗后行数: {len(quarterly_clean)}；删除说明行: {quarterly_removed}")
    print(f"[OK] 宏观因子数: definition={len(definitions)}, monthly={len(monthly_features)}, quarterly={len(quarterly_features)}, available={summary['available_factor_count']}")
    print(f"[OK] 已生成: {output_dir / OUTPUTS['clean_monthly']}")
    print(f"[OK] 已生成: {output_dir / OUTPUTS['clean_quarterly']}")
    print(f"[OK] 已生成: {output_dir / OUTPUTS['aligned_quarterly']}")
    print(f"[OK] 已生成: {output_dir / OUTPUTS['feature_catalog']}")
    print(f"[OK] 已生成: {output_dir / OUTPUTS['time_mapping_report']}")
    if conflict_rows:
        print(f"[WARN] 发现 {len(conflict_rows)} 条宏观字段冲突，详见: {output_dir / OUTPUTS['conflict_report']}")
    if time_exceptions:
        print(f"[WARN] 发现 {len(time_exceptions)} 条时间/数值清洗异常记录，详见: {output_dir / OUTPUTS['time_parse_exceptions']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗宏观月频/季频数据，并生成季度对齐审计报告。")
    parser.add_argument("--macro-workbook", default="原始数据/宏观因子数据.xlsx", help="宏观因子工作簿路径")
    parser.add_argument("--macro-definition-sheet", default="宏观因子", help="宏观因子说明 sheet")
    parser.add_argument("--macro-month-sheet", default="宏观因子具体数据", help="月频宏观数据 sheet")
    parser.add_argument("--macro-quarter-sheet", default="Sheet1", help="季频宏观数据 sheet")
    parser.add_argument("--time-col", default="时间", help="时间列名，格式应为 YYYYMM")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--alignment-mode", default="prefer_quarterly", choices=sorted(VALID_ALIGNMENT_MODES), help="季度对齐模式")
    parser.add_argument("--level-agg", default="quarter_end", choices=["quarter_end", "quarter_mean"], help="level 类变量月转季规则")
    parser.add_argument("--growth-agg", default="quarter_end", choices=["quarter_end", "quarter_mean"], help="growth 类变量月转季规则")
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
