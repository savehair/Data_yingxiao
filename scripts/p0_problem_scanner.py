#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scan P0 project artifacts and produce a project-wide problem summary.

Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_COLUMNS = [
    "issue_id",
    "issue_title",
    "issue_type",
    "evidence",
    "severity",
    "priority",
    "suggested_fix",
    "success_metric",
    "success_assertion",
]


@dataclass
class Issue:
    issue_id: str
    issue_title: str
    issue_type: str
    evidence: str
    severity: str
    priority: str
    suggested_fix: str
    success_metric: str
    success_assertion: str

    def to_row(self) -> dict[str, str]:
        return {
            "issue_id": self.issue_id,
            "issue_title": self.issue_title,
            "issue_type": self.issue_type,
            "evidence": self.evidence,
            "severity": self.severity,
            "priority": self.priority,
            "suggested_fix": self.suggested_fix,
            "success_metric": self.success_metric,
            "success_assertion": self.success_assertion,
        }


def fail(message: str) -> SystemExit:
    return SystemExit(f"[ERROR] {message}")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def next_issue_id(counter: int) -> str:
    return f"ISSUE-{counter:03d}"


def add_issue(
    issues: list[Issue],
    title: str,
    issue_type: str,
    evidence: str,
    severity: str,
    priority: str,
    suggested_fix: str,
    success_metric: str,
    success_assertion: str,
) -> None:
    issues.append(
        Issue(
            issue_id=next_issue_id(len(issues) + 1),
            issue_title=title,
            issue_type=issue_type,
            evidence=evidence,
            severity=severity,
            priority=priority,
            suggested_fix=suggested_fix,
            success_metric=success_metric,
            success_assertion=success_assertion,
        )
    )


def load_stock_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise fail(f"stock_panel_path 不存在: {path}")
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise fail(f"读取 stock panel parquet 失败: {path}；原因: {exc}") from exc


def scan_doc_data_factor_conflicts(issues: list[Issue], macro_dir: Path) -> None:
    conflict_path = macro_dir / "macro_conflict_report.csv"
    conflicts = read_csv_if_exists(conflict_path)
    if conflicts.empty:
        return
    count_mismatch = conflicts[conflicts.get("issue", pd.Series(dtype=str)).astype(str) == "factor_count_mismatch"]
    if not count_mismatch.empty:
        evidence = "; ".join(count_mismatch["recommendation"].astype(str).tolist())
        add_issue(
            issues,
            "文档口径与宏观数据字段数量不一致",
            "doc_data_schema_conflict",
            evidence,
            "high",
            "P0",
            "确认缺失的第 12 个宏观因子是否应补充、派生或在报告中说明剔除。",
            "macro_factor_count_matches_requirement",
            "available_macro_factor_count == documented_macro_factor_count",
        )
    other_conflicts = conflicts[conflicts.get("issue", pd.Series(dtype=str)).astype(str) != "factor_count_mismatch"]
    if not other_conflicts.empty:
        evidence = f"{len(other_conflicts)} 个宏观字段只出现在部分表中: " + ", ".join(other_conflicts["feature"].astype(str).head(8).tolist())
        add_issue(
            issues,
            "宏观说明 sheet、月频表、季频表字段覆盖不一致",
            "macro_schema_conflict",
            evidence,
            "medium",
            "P1",
            "在 macro_feature_catalog 中明确每个字段来源；月频-only 字段用聚合规则转季度或标记为不进季度模型。",
            "macro_catalog_conflict_count",
            "macro_catalog_conflict_count == 0 or all_conflicts_have_documented_resolution == True",
        )


def scan_label_leakage(issues: list[Issue], panel: pd.DataFrame, label_col: str, id_col: str) -> None:
    if label_col not in panel.columns:
        add_issue(
            issues,
            "标签列缺失",
            "label_definition",
            f"stock panel 中未找到 label_col={label_col}",
            "critical",
            "P0",
            "确认收益标签列名并重新构建 stock_panel。",
            "label_col_exists",
            "label_col in stock_panel.columns",
        )
        return
    if "quarter_idx" not in panel.columns or id_col not in panel.columns:
        return
    label_non_null_rate = float(panel[label_col].notna().mean())
    duplicate_label_key_count = int(panel[[id_col, "quarter_idx"]].duplicated().sum())
    evidence = f"label_col={label_col}; non_null_rate={label_non_null_rate:.4f}; duplicate_key_count={duplicate_label_key_count}"
    add_issue(
        issues,
        "标签需要验证是否严格来自下一期收益",
        "label_leakage_risk",
        evidence,
        "high",
        "P0",
        "构造显式 next_quarter_idx = quarter_idx + 1 的标签匹配审计，禁止同季度收益直接作为训练标签。",
        "label_shift_match_rate",
        "label_shift_match_rate > 0.99",
    )


def scan_macro_frequency(issues: list[Issue], macro_dir: Path) -> None:
    run_summary = read_json_if_exists(macro_dir / "macro_clean_run_summary.json")
    catalog = read_csv_if_exists(macro_dir / "macro_feature_catalog.csv")
    if run_summary:
        monthly_count = run_summary.get("monthly_factor_count")
        quarterly_count = run_summary.get("quarterly_factor_count")
        alignment_mode = run_summary.get("alignment_mode")
        if monthly_count != quarterly_count:
            add_issue(
                issues,
                "宏观月频与季频字段数量不一致",
                "macro_frequency_mismatch",
                f"monthly_factor_count={monthly_count}; quarterly_factor_count={quarterly_count}; alignment_mode={alignment_mode}",
                "medium",
                "P1",
                "对季频缺失字段使用明确聚合规则，或在建模配置中固定使用 dual_track 并防止混用。",
                "macro_frequency_alignment_documented",
                "all_macro_features_have_frequency_and_alignment_note == True",
            )
    if not catalog.empty:
        missing_notes = catalog[catalog["frequency"].isna() | catalog["alignment_note"].isna()]
        if not missing_notes.empty:
            add_issue(
                issues,
                "宏观字段缺少频率或对齐说明",
                "macro_catalog_incomplete",
                f"missing_note_count={len(missing_notes)}",
                "high",
                "P0",
                "补齐 macro_feature_catalog.csv 中每个字段的 frequency 和 alignment_note。",
                "macro_catalog_completeness",
                "all_macro_features_have_frequency_and_alignment_note == True",
            )


def scan_desc_rows(issues: list[Issue], macro_dir: Path) -> None:
    run_summary = read_json_if_exists(macro_dir / "macro_clean_run_summary.json")
    if not run_summary:
        return
    monthly_removed = run_summary.get("monthly_explanation_rows_removed", 0)
    quarterly_removed = run_summary.get("quarterly_explanation_rows_removed", 0)
    if monthly_removed or quarterly_removed:
        add_issue(
            issues,
            "宏观表存在首行说明导致的伪数据风险",
            "description_row_pseudodata",
            f"monthly_explanation_rows_removed={monthly_removed}; quarterly_explanation_rows_removed={quarterly_removed}",
            "medium",
            "P1",
            "保留说明行删除审计；后续所有宏观输入只使用 clean 文件，不直接读取原始 sheet。",
            "macro_desc_rows_removed",
            "macro_desc_rows_removed == True",
        )


def scan_missingness(issues: list[Issue], panel: pd.DataFrame, threshold: float) -> None:
    missing_rates = panel.isna().mean().sort_values(ascending=False)
    high_missing = missing_rates[missing_rates > threshold]
    if high_missing.empty:
        return
    evidence = "; ".join(f"{col}={rate:.2%}" for col, rate in high_missing.head(10).items())
    add_issue(
        issues,
        "存在缺失率过高字段",
        "missingness",
        evidence,
        "medium",
        "P1",
        "按训练窗口统计缺失率，删除超过阈值字段或配置稳定填补策略，并在报告中记录字段处理结果。",
        "max_missing_rate_after_drop",
        "max_missing_rate_after_drop <= threshold",
    )


def scan_outliers_and_skew(issues: list[Issue], panel: pd.DataFrame, id_cols: set[str], skew_threshold: float, outlier_ratio_threshold: float) -> None:
    numeric_cols = [col for col in panel.select_dtypes(include="number").columns if col not in id_cols]
    skew_hits: list[str] = []
    outlier_hits: list[str] = []
    for col in numeric_cols:
        series = panel[col].dropna()
        if len(series) < 20:
            continue
        skew = float(series.skew())
        if abs(skew) > skew_threshold:
            skew_hits.append(f"{col}:skew={skew:.2f}")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_ratio = float(((series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)).mean())
        if outlier_ratio > outlier_ratio_threshold:
            outlier_hits.append(f"{col}:outlier_ratio={outlier_ratio:.2%}")
    if skew_hits or outlier_hits:
        evidence = "skew: " + ", ".join(skew_hits[:8]) + " | outliers: " + ", ".join(outlier_hits[:8])
        add_issue(
            issues,
            "部分数值字段存在严重偏态或异常值",
            "outlier_skew",
            evidence,
            "medium",
            "P1",
            "在训练窗口内执行缩尾、稳健标准化或对数变换，避免全样本预处理造成泄漏。",
            "extreme_feature_share_after_winsorize",
            "extreme_feature_share_after_winsorize <= outlier_ratio_threshold",
        )


def scan_financial_statement_anomalies(issues: list[Issue], panel: pd.DataFrame) -> None:
    checks: list[str] = []
    for col in panel.columns:
        if col not in panel.select_dtypes(include="number").columns:
            continue
        series = panel[col].dropna()
        if series.empty:
            continue
        if "市盈率PE" in col:
            neg_ratio = float((series < 0).mean())
            if neg_ratio > 0:
                checks.append(f"{col}: negative_ratio={neg_ratio:.2%}")
        if "市现率PCF" in col:
            neg_ratio = float((series < 0).mean())
            if neg_ratio > 0:
                checks.append(f"{col}: negative_ratio={neg_ratio:.2%}")
        if "增长率" in col or "增速" in col:
            explosion_ratio = float((series.abs() > 300).mean())
            if explosion_ratio > 0:
                checks.append(f"{col}: abs_gt_300_ratio={explosion_ratio:.2%}")
    if checks:
        add_issue(
            issues,
            "财务口径存在负 PE/负 PCF 或爆炸增长率",
            "financial_metric_anomaly",
            "; ".join(checks[:12]),
            "medium",
            "P1",
            "为估值和增长率字段建立口径规则：负估值单独标记或截尾，爆炸增长率用缩尾/缺失处理并记录原因。",
            "financial_anomaly_policy_applied",
            "negative_pe_pcf_policy_confirmed == True and explosive_growth_rate_handled == True",
        )


def scan_duplicate_keys(issues: list[Issue], panel: pd.DataFrame, id_col: str) -> None:
    if id_col not in panel.columns or "quarter_idx" not in panel.columns:
        return
    duplicate_count = int(panel[[id_col, "quarter_idx"]].duplicated().sum())
    if duplicate_count > 0:
        add_issue(
            issues,
            "股票季度面板存在重复 key",
            "duplicate_key",
            f"duplicated({id_col}, quarter_idx)={duplicate_count}",
            "high",
            "P0",
            "定位重复证券代码-季度组合，确认是否多市场代码、重复导入或名称变更导致。",
            "duplicate_stock_quarter_key_count",
            "duplicate_stock_quarter_key_count == 0",
        )


def scan_time_mapping(issues: list[Issue], panel: pd.DataFrame, macro_dir: Path) -> None:
    aligned_path = macro_dir / "macro_quarterly_aligned.csv"
    macro_aligned = read_csv_if_exists(aligned_path)
    if macro_aligned.empty or "quarter_idx" not in panel.columns:
        add_issue(
            issues,
            "宏观季度与股票 quarter_idx 映射缺少可机读校验",
            "time_mapping",
            f"macro_aligned_exists={aligned_path.exists()}; stock_has_quarter_idx={'quarter_idx' in panel.columns}",
            "medium",
            "P1",
            "输出显式 quarter_idx 到 macro_quarter_period 的映射表，并在合并前校验覆盖率。",
            "macro_stock_quarter_mapping_coverage",
            "macro_stock_quarter_mapping_coverage == 1.0",
        )
        return
    stock_quarters = set(panel["quarter_idx"].dropna().astype(int).unique().tolist())
    macro_quarter_count = len(macro_aligned["quarter_period"].dropna().astype(str).unique()) if "quarter_period" in macro_aligned.columns else len(macro_aligned)
    expected = set(range(1, macro_quarter_count + 1))
    missing = sorted(stock_quarters - expected)
    if missing:
        add_issue(
            issues,
            "宏观季度覆盖不完整",
            "time_mapping",
            f"stock_quarter_count={len(stock_quarters)}; macro_quarter_count={macro_quarter_count}; missing_stock_quarters={missing[:10]}",
            "high",
            "P0",
            "补齐宏观季度或缩小股票面板时间范围，确保每个股票季度都有宏观特征。",
            "macro_stock_quarter_mapping_coverage",
            "macro_stock_quarter_mapping_coverage == 1.0",
        )


def scan_feature_roles(issues: list[Issue], macro_dir: Path) -> None:
    role_map = read_csv_if_exists(macro_dir / "feature_role_map.csv")
    if role_map.empty:
        return
    suspicious = []
    for _, row in role_map.iterrows():
        col = str(row.get("original_column", ""))
        role = str(row.get("role", ""))
        if role == "factor" and any(token in col for token in ["平均收益率", "下一期收益", "未来收益", "区间涨跌幅"]):
            suspicious.append(col)
        if role == "return_like" and col != "月平均收益率":
            suspicious.append(col)
    if suspicious:
        add_issue(
            issues,
            "feature role 可能混淆收益类字段与普通因子",
            "feature_role_confusion",
            "suspicious_columns=" + ", ".join(suspicious[:12]),
            "high",
            "P0",
            "检查 feature_role_map.csv，把历史收益/未来收益/标签类字段从普通因子候选集中隔离。",
            "return_like_as_factor_count",
            "return_like_as_factor_count == 0",
        )


def scan_train_test_time_risk(issues: list[Issue], task_requirements_path: Path, macro_dir: Path) -> None:
    assumption = read_csv_if_exists(macro_dir / "assumption_ledger_final.csv")
    scheme_confirmed = False
    if not assumption.empty and {"item", "value", "status"}.issubset(assumption.columns):
        row = assumption[assumption["item"].astype(str) == "train_valid_test_scheme"]
        if not row.empty:
            value = str(row.iloc[0]["value"])
            status = str(row.iloc[0]["status"])
            scheme_confirmed = status == "confirmed" and "time" in value.lower()
    task_text = task_requirements_path.read_text(encoding="utf-8") if task_requirements_path.exists() else ""
    evidence = f"train_valid_test_scheme_time_confirmed={scheme_confirmed}; task_requirements_exists={task_requirements_path.exists()}; task_text_len={len(task_text)}"
    add_issue(
        issues,
        "训练/验证/测试仍需后续脚本执行时间顺序断言",
        "temporal_split_risk",
        evidence,
        "high",
        "P0",
        "后续建模前输出 split manifest，确保训练、验证、测试按时间递增且预处理只在训练窗口拟合。",
        "temporal_split_order",
        "train_end_date < valid_start_date <= test_start_date",
    )


def write_csv(path: Path, issues: list[Issue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows([issue.to_row() for issue in issues])


def write_markdown(path: Path, issues: list[Issue]) -> None:
    lines = [
        "# Problem Summary",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- issue_count: `{len(issues)}`",
        "",
        "| issue_id | priority | severity | issue_type | issue_title | success_assertion |",
        "|---|---|---|---|---|---|",
    ]
    for issue in issues:
        lines.append(
            f"| {issue.issue_id} | {issue.priority} | {issue.severity} | {issue.issue_type} | "
            f"{escape_md(issue.issue_title)} | `{escape_md(issue.success_assertion)}` |"
        )
    lines.extend(["", "## Details", ""])
    for issue in issues:
        lines.extend(
            [
                f"### {issue.issue_id} {issue.issue_title}",
                "",
                f"- type: `{issue.issue_type}`",
                f"- severity: `{issue.severity}`",
                f"- priority: `{issue.priority}`",
                f"- evidence: {issue.evidence}",
                f"- suggested_fix: {issue.suggested_fix}",
                f"- success_metric: `{issue.success_metric}`",
                f"- success_assertion: `{issue.success_assertion}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def run(args: argparse.Namespace) -> int:
    base_dir = Path.cwd()
    stock_panel_path = resolve_path(args.stock_panel_path, base_dir)
    macro_clean_dir = resolve_path(args.macro_clean_dir, base_dir, must_exist=True, is_dir=True)
    task_requirements_path = resolve_path(args.task_requirements_path, base_dir, must_exist=False)
    output_dir = resolve_path(args.output_dir, base_dir, must_exist=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_stock_panel(stock_panel_path)
    issues: list[Issue] = []

    scan_doc_data_factor_conflicts(issues, macro_clean_dir)
    scan_label_leakage(issues, panel, args.label_col, args.id_col)
    scan_macro_frequency(issues, macro_clean_dir)
    scan_desc_rows(issues, macro_clean_dir)
    scan_missingness(issues, panel, args.missing_threshold)
    scan_outliers_and_skew(issues, panel, {args.id_col, "quarter_idx"}, args.skew_threshold, args.outlier_ratio_threshold)
    scan_financial_statement_anomalies(issues, panel)
    scan_duplicate_keys(issues, panel, args.id_col)
    scan_time_mapping(issues, panel, macro_clean_dir)
    scan_feature_roles(issues, macro_clean_dir)
    scan_train_test_time_risk(issues, task_requirements_path, macro_clean_dir)

    csv_path = output_dir / "problem_summary.csv"
    md_path = output_dir / "problem_summary.md"
    write_csv(csv_path, issues)
    write_markdown(md_path, issues)

    print(f"[OK] 已生成问题汇总表: {csv_path}")
    print(f"[OK] 已生成问题汇总 Markdown: {md_path}")
    print(f"[OK] 问题数量: {len(issues)}")
    for priority in ["P0", "P1", "P2"]:
        count = sum(1 for issue in issues if issue.priority == priority)
        print(f"  - {priority}: {count}")
    return 0


def resolve_path(path_text: str, base_dir: Path, must_exist: bool = True, is_dir: bool = False) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise fail(f"路径不存在: {path}")
    if must_exist and is_dir and not path.is_dir():
        raise fail(f"路径不是目录: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描项目 P0 阶段产物，输出问题统计总表。")
    parser.add_argument("--stock-panel-path", default="outputs/stock_panel.parquet", help="股票长表 parquet")
    parser.add_argument("--macro-clean-dir", default="outputs", help="宏观清洗和审计输出目录")
    parser.add_argument("--task-requirements-path", default="outputs/task_requirements.md", help="任务要求摘要文件")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--id-col", default="证券代码", help="股票代码列")
    parser.add_argument("--label-col", default="月平均收益率", help="标签列")
    parser.add_argument("--missing-threshold", type=float, default=0.30, help="缺失率高风险阈值")
    parser.add_argument("--skew-threshold", type=float, default=5.0, help="严重偏态阈值")
    parser.add_argument("--outlier-ratio-threshold", type=float, default=0.05, help="IQR 极端值比例阈值")
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
