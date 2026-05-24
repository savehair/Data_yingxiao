#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Maintain the project assumption ledger.

The script reads the seed ledger, fills required unspecified assumptions,
applies YAML overrides, and writes a final ledger plus the adopted YAML config.
It uses only the Python standard library so it can run before project
dependencies are installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_ITEMS = [
    "business_goal",
    "task_type",
    "label_col",
    "label_is_shifted",
    "date_mapping_rule",
    "macro_frequency_alignment",
    "quarterly_rebalance_rule",
    "train_valid_test_scheme",
    "rating_bin_rule",
    "benchmark_series",
    "transaction_cost_bps",
    "slippage_bps",
    "industry_neutralization",
    "missing_value_policy",
    "outlier_policy",
    "leakage_guard_rules",
]

ITEM_ALIASES = {
    "label_definition": "label_col",
    "macro_alignment_mode": "macro_frequency_alignment",
    "rebalancing_freq": "quarterly_rebalance_rule",
    "validation_scheme": "train_valid_test_scheme",
    "rating_bins": "rating_bin_rule",
    "transaction_cost": "transaction_cost_bps",
    "benchmark": "benchmark_series",
    "missing_value_strategy": "missing_value_policy",
    "outlier_treatment": "outlier_policy",
    "leakage_checks": "leakage_guard_rules",
}


@dataclass(frozen=True)
class AssumptionSpec:
    default_assumption: str
    alternatives: str
    risk_note: str
    high_risk: bool


ASSUMPTION_CATALOG: dict[str, AssumptionSpec] = {
    "business_goal": AssumptionSpec(
        "用多因子、宏观因子和变量选择方法支持下一期收益预测、评级与回测评估",
        "收益预测；风险评级；组合排序；解释性变量筛选",
        "业务目标不明确会导致标签、评价指标和回测口径不一致。",
        True,
    ),
    "task_type": AssumptionSpec(
        "ranking",
        "regression；classification；ranking",
        "任务类型会影响标签构造、损失函数、评级规则和最终评价指标。",
        True,
    ),
    "label_col": AssumptionSpec(
        "next_period_return",
        "月平均收益率；季度收益率；超额收益率；评级标签",
        "标签列选错会直接造成训练目标偏离作业要求。",
        True,
    ),
    "label_is_shifted": AssumptionSpec(
        "true",
        "true；false",
        "若没有使用下一期收益率作为标签，模型可能学习到同期信息并产生未来函数。",
        True,
    ),
    "date_mapping_rule": AssumptionSpec(
        "因子期 t 匹配收益期 t+1，宏观数据只使用预测时点已发布数据",
        "同月匹配；滞后一月匹配；季度末匹配；公告日可得性匹配",
        "日期映射错误是时间序列建模中最常见的数据泄露来源。",
        True,
    ),
    "macro_frequency_alignment": AssumptionSpec(
        "宏观因子按可得日期向前填充到股票样本期，并至少滞后一档",
        "月频前向填充；季度末对齐；发布日滞后；不用宏观因子",
        "宏观数据若按事后完整月份对齐，可能把预测时点未知的信息带入模型。",
        True,
    ),
    "quarterly_rebalance_rule": AssumptionSpec(
        "每个季度末基于当期可得因子形成评级，下一季度持有",
        "月度调仓；季度调仓；年度调仓；只做样本外预测不调仓",
        "调仓频率不清会影响换手率、交易成本和回测收益。",
        False,
    ),
    "train_valid_test_scheme": AssumptionSpec(
        "按时间顺序切分训练、验证、测试集，禁止随机打乱",
        "滚动窗口；扩展窗口；固定时间切分；留出最后若干期",
        "随机切分会把未来时期分布泄露到训练集，夸大预测效果。",
        True,
    ),
    "rating_bin_rule": AssumptionSpec(
        "按预测分数横截面五分位分为 5 档评级",
        "三档；五档；十分位；按绝对阈值；PROMETHEE 净流量分档",
        "评级分箱规则不同会改变评级稳定性、样本占比和回测组合构成。",
        False,
    ),
    "benchmark_series": AssumptionSpec(
        "全股票池等权收益",
        "沪深300；中证500；行业中性基准；全样本市值加权",
        "基准选择会显著影响超额收益、信息比率和胜率解释。",
        False,
    ),
    "transaction_cost_bps": AssumptionSpec(
        "10",
        "0；5；10；20；自定义券商费率",
        "未计交易成本会高估高换手策略的回测表现。",
        True,
    ),
    "slippage_bps": AssumptionSpec(
        "5",
        "0；5；10；按成交额或流动性估计",
        "滑点假设过低会高估组合可实施性。",
        False,
    ),
    "industry_neutralization": AssumptionSpec(
        "false",
        "true；false；仅标准化不中性化；行业内排序",
        "若行业暴露未受控，模型可能主要反映行业轮动而非因子有效性。",
        False,
    ),
    "missing_value_policy": AssumptionSpec(
        "按期中位数填补，并保留缺失率审计",
        "删除样本；均值填补；行业中位数填补；模型填补",
        "缺失处理会改变样本池，极端情况下引入幸存者偏差。",
        True,
    ),
    "outlier_policy": AssumptionSpec(
        "按期 1%/99% 分位缩尾",
        "不处理；MAD 缩尾；标准差截断；按行业缩尾",
        "异常值处理不一致会影响回归变量选择和 PROMETHEE 权重稳定性。",
        False,
    ),
    "leakage_guard_rules": AssumptionSpec(
        "所有标准化、填补、缩尾、变量选择和权重学习仅在训练窗口拟合，再应用到验证/测试",
        "全样本预处理；滚动拟合；扩展窗口拟合；按季度横截面拟合",
        "若预处理或变量选择使用全样本，会造成严重样本外绩效高估。",
        True,
    ),
}


OUTPUT_COLUMNS = [
    "item",
    "value",
    "status",
    "reason",
    "configurable",
    "high_risk",
    "risk_note",
    "downstream_usage",
    "default_assumption",
    "alternatives",
    "source_type",
    "note",
]

PROBLEM_SUMMARY_COLUMNS = [
    "problem_type",
    "item",
    "status",
    "severity",
    "risk_note",
    "downstream_usage",
    "recommended_action",
]

DOWNSTREAM_USAGE = {
    "business_goal": "决定建模目标、报告叙事和评价指标选择",
    "task_type": "决定预测输出、评级转换和模型评价方式",
    "label_col": "用于构造训练标签、验证目标和收益回测",
    "label_is_shifted": "用于检查下一期收益率匹配和未来信息泄露",
    "date_mapping_rule": "用于合并因子、收益和宏观数据的时间键",
    "macro_frequency_alignment": "用于宏观因子重采样、滞后和可得性处理",
    "quarterly_rebalance_rule": "用于组合构建、调仓频率和换手率计算",
    "train_valid_test_scheme": "用于样本切分、样本外验证和报告可信度",
    "rating_bin_rule": "用于股票评级分档和分组收益分析",
    "benchmark_series": "用于超额收益、胜率和回撤指标对比",
    "transaction_cost_bps": "用于回测净收益、换手成本和策略可实施性评估",
    "slippage_bps": "用于回测净收益和交易冲击估计",
    "industry_neutralization": "用于控制行业暴露和横截面可比性",
    "missing_value_policy": "用于特征矩阵构建和样本保留规则",
    "outlier_policy": "用于因子预处理、回归稳定性和权重稳定性",
    "leakage_guard_rules": "用于预处理拟合窗口、变量选择和样本外评估防泄露",
}


def fail(message: str) -> SystemExit:
    return SystemExit(f"[ERROR] {message}")


def read_seed_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"[WARN] 初始 assumption ledger 不存在，将仅使用标准目录和 YAML 配置生成最终台账: {path}")
        return []
    if not path.is_file():
        raise fail(f"初始 assumption ledger 不是文件: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise fail(f"初始 assumption ledger 为空或缺少表头: {path}")
            rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    except UnicodeDecodeError as exc:
        raise fail(f"无法按 UTF-8/UTF-8-SIG 读取 CSV: {path}") from exc

    if not rows:
        print(f"[WARN] 初始 assumption ledger 没有数据行，将仅使用标准目录和 YAML 配置生成最终台账: {path}")
        return []
    if "item" not in reader.fieldnames:
        raise fail(f"初始 assumption ledger 缺少必需列 item: {path}")
    return rows


def strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return value[:index].strip()
    return value.strip()


def parse_scalar(value: str) -> Any:
    value = strip_inline_comment(value)
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return "true" if lowered == "true" else "false"
    if lowered in {"null", "none", "~"}:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return ""
        return "；".join(parse_scalar(part.strip()) for part in inner.split(","))
    return value


def read_override_yaml(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    if not path.is_file():
        raise fail(f"override_yaml 不是文件: {path}")

    overrides: dict[str, dict[str, str]] = {}
    current_section: str | None = None
    current_item: str | None = None
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise fail(f"YAML 第 {line_no} 行格式无法解析，需使用 key: value: {raw_line}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise fail(f"YAML 第 {line_no} 行 key 为空: {raw_line}")

        if indent == 0 and not value:
            current_section = key
            current_item = None
            continue
        if indent == 2 and current_section in {"assumptions", "overrides"} and not value:
            current_item = ITEM_ALIASES.get(key, key)
            overrides.setdefault(current_item, {})
            continue
        if indent == 2 and current_section in {"assumptions", "overrides"}:
            item = ITEM_ALIASES.get(key, key)
            overrides[item] = {"value": str(parse_scalar(value))}
            current_item = None
            continue
        if indent == 4 and current_section in {"assumptions", "overrides"} and current_item:
            if key not in {"value", "status", "reason", "note"}:
                raise fail(f"YAML 第 {line_no} 行仅支持 value/status/reason/note 字段: {raw_line}")
            overrides.setdefault(current_item, {})[key] = str(parse_scalar(value))
        elif indent == 0:
            if key in REQUIRED_ITEMS or key in ASSUMPTION_CATALOG:
                item = ITEM_ALIASES.get(key, key)
                overrides[item] = {"value": str(parse_scalar(value))}
            else:
                current_section = None
                current_item = None
        else:
            raise fail(f"YAML 第 {line_no} 行仅支持 assumptions/overrides 下的 key: value 或 item/value/status/reason/note 结构: {raw_line}")
    return overrides


def yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def yaml_format(value: str) -> str:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered
    try:
        float(value)
    except ValueError:
        return yaml_quote(value)
    return value


def write_adopted_yaml(path: Path, adopted_config: dict[str, dict[str, str]], source_yaml: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Project assumptions adopted by scripts/p0_assumption_ledger.py",
        f"# generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"# source_override_yaml: {source_yaml.as_posix()}",
        "assumptions:",
    ]
    for item in REQUIRED_ITEMS:
        config = adopted_config[item]
        lines.append(f"  {item}:")
        lines.append(f"    value: {yaml_format(config['value'])}")
        lines.append(f"    status: {yaml_quote(config['status'])}")
        if config.get("reason"):
            lines.append(f"    reason: {yaml_quote(config['reason'])}")
        if config.get("note"):
            lines.append(f"    note: {yaml_quote(config['note'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_bps_unit(item: str, value: str) -> None:
    if "bps" not in item:
        return
    try:
        numeric_value = float(value)
    except ValueError as exc:
        raise fail(f"{item} 字段名包含 bps，值应为基点数字，例如 10；当前值无法解析为数字: {value}") from exc
    if 0 < numeric_value < 1:
        suggested = item.replace("_bps", "_rate")
        raise fail(
            f"{item} 字段名包含 bps，值应为基点数字，例如 10。当前值 {value} 可能是 rate 口径；"
            f"请改为 {item}: 10，或新增/改用 {suggested}: {value}。"
        )


def normalize_rows(
    seed_rows: list[dict[str, str]],
    overrides: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    by_item: dict[str, dict[str, str]] = {}

    for row in seed_rows:
        raw_item = row.get("item", "").strip()
        if not raw_item:
            continue
        item = ITEM_ALIASES.get(raw_item, raw_item)
        normalized = dict(row)
        normalized["item"] = item
        if item not in by_item:
            by_item[item] = normalized

    for item in REQUIRED_ITEMS:
        by_item.setdefault(
            item,
            {
                "item": item,
                "value": "",
                "source_type": "default_assumption",
                "status": "to_confirm",
                "configurable": "true",
                "rationale": "",
            },
        )

    for item in sorted(key for key in overrides if key not in by_item):
        by_item[item] = {
            "item": item,
            "value": "",
            "source_type": "yaml_override",
            "status": "configured",
            "configurable": "true",
            "rationale": "YAML 中提供的项目自定义配置项。",
        }

    normalized_rows: list[dict[str, str]] = []
    adopted_config: dict[str, dict[str, str]] = {}

    for item in REQUIRED_ITEMS + sorted(key for key in by_item if key not in REQUIRED_ITEMS):
        row = by_item[item]
        override = overrides.get(item, {})
        spec = ASSUMPTION_CATALOG.get(
            item,
            AssumptionSpec(
                row.get("value", ""),
                "按项目需要自定义",
                "该项未在标准目录中定义，请确认是否会影响数据、建模或回测口径。",
                False,
            ),
        )
        default_value = spec.default_assumption
        seed_value = row.get("value", "").strip()
        adopted_value = override.get("value") or seed_value or default_value
        validate_bps_unit(item, adopted_value)

        source_type = "yaml_override" if item in overrides else row.get("source_type", "default_assumption") or "default_assumption"
        status = override.get("status") or ("configured" if item in overrides else row.get("status", "to_confirm") or "to_confirm")
        if adopted_value == default_value and item not in overrides:
            status = "defaulted"
        reason = override.get("reason") or row.get("reason", "") or row.get("rationale", "") or ""
        note = override.get("note") or row.get("note", "") or ""
        if status == "confirmed" and not reason:
            reason = "人工确认配置"

        adopted_config[item] = {
            "value": adopted_value,
            "status": status,
            "reason": reason,
            "note": note,
        }

        final_row = {
            "item": item,
            "value": adopted_value,
            "status": status,
            "reason": reason,
            "configurable": row.get("configurable", "true") or "true",
            "high_risk": "true" if spec.high_risk else "false",
            "risk_note": spec.risk_note,
            "downstream_usage": DOWNSTREAM_USAGE.get(item, "用于项目自定义配置和后续处理逻辑"),
            "default_assumption": default_value,
            "alternatives": spec.alternatives,
            "source_type": source_type,
            "note": note,
        }
        normalized_rows.append(final_row)

    return normalized_rows, adopted_config


def validate_final_rows(rows: list[dict[str, str]]) -> None:
    by_item = {row["item"]: row for row in rows}
    missing = [item for item in REQUIRED_ITEMS if item not in by_item]
    if missing:
        raise fail(f"最终台账缺少必备未指定项: {', '.join(missing)}")

    incomplete: list[str] = []
    for item in REQUIRED_ITEMS:
        row = by_item[item]
        required_cols = ["value", "default_assumption", "alternatives", "risk_note"]
        empty_cols = [col for col in required_cols if not row.get(col, "").strip()]
        if empty_cols:
            incomplete.append(f"{item}({', '.join(empty_cols)})")
    if incomplete:
        raise fail(f"以下未指定项缺少默认值、备选方案或风险提示: {'; '.join(incomplete)}")


def write_final_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows([{col: row.get(col, "") for col in OUTPUT_COLUMNS} for row in rows])


def unconfirmed_high_risk_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("high_risk") == "true" and row.get("status", "").strip().lower() != "confirmed"
    ]


def write_problem_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    problem_rows = []
    for row in unconfirmed_high_risk_rows(rows):
        problem_rows.append(
            {
                "problem_type": "高风险未指定项",
                "item": row["item"],
                "status": row["status"],
                "severity": "high",
                "risk_note": row["risk_note"],
                "downstream_usage": row["downstream_usage"],
                "recommended_action": "请在 configs/project_assumptions.yaml 中设置 value 并将 status 标记为 confirmed。",
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PROBLEM_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(problem_rows)


def write_summary_json(path: Path, adopted_config: dict[str, dict[str, str]], rows: list[dict[str, str]]) -> None:
    high_risk_items = [row["item"] for row in unconfirmed_high_risk_rows(rows)]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "adopted_config": {item: adopted_config[item] for item in REQUIRED_ITEMS},
        "unconfirmed_high_risk_items": high_risk_items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_run_summary(adopted_config: dict[str, dict[str, str]], rows: list[dict[str, str]]) -> None:
    print("\n本次运行实际采用的配置摘要")
    print("-" * 40)
    for item in REQUIRED_ITEMS:
        config = adopted_config[item]
        print(f"{item}: value={config['value']} status={config['status']}")
        if config.get("note"):
            print(f"  note: {config['note']}")

    print("\n高风险未指定项")
    print("-" * 40)
    high_risk_rows = unconfirmed_high_risk_rows(rows)
    if not high_risk_rows:
        print("(无)")
        return
    for row in high_risk_rows:
        marker = row.get("status") or "unknown"
        print(f"- {row['item']} [{marker}]: {row['risk_note']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="维护项目未指定项假设台账，并生成最终配置。")
    parser.add_argument("--assumption-file", default="outputs/assumption_ledger_seed.csv", help="初始 assumption ledger CSV")
    parser.add_argument("--override-yaml", default="configs/project_assumptions.yaml", help="用户覆盖 YAML；不存在时使用默认假设生成")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--final-csv-name", default="assumption_ledger_final.csv", help="最终台账文件名")
    parser.add_argument("--problem-csv-name", default="problem_summary.csv", help="高风险未指定项摘要文件名")
    parser.add_argument("--summary-json-name", default="assumption_config_summary.json", help="配置摘要 JSON 文件名")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path.cwd()
    assumption_file = Path(args.assumption_file)
    override_yaml = Path(args.override_yaml)
    output_dir = Path(args.output_dir)
    if not assumption_file.is_absolute():
        assumption_file = base_dir / assumption_file
    if not override_yaml.is_absolute():
        override_yaml = base_dir / override_yaml
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    seed_rows = read_seed_ledger(assumption_file)
    overrides = read_override_yaml(override_yaml)
    unknown_overrides = sorted(key for key in overrides if key not in ASSUMPTION_CATALOG)
    if unknown_overrides:
        print(f"[WARN] YAML 中存在标准目录外配置项，将保留但不纳入必备项验收: {', '.join(unknown_overrides)}")

    final_rows, adopted_config = normalize_rows(seed_rows, overrides)
    validate_final_rows(final_rows)

    final_csv = output_dir / args.final_csv_name
    problem_csv = output_dir / args.problem_csv_name
    summary_json = output_dir / args.summary_json_name
    write_final_csv(final_csv, final_rows)
    write_problem_summary_csv(problem_csv, final_rows)
    write_adopted_yaml(override_yaml, adopted_config, override_yaml)
    write_summary_json(summary_json, adopted_config, final_rows)

    print(f"[OK] 已生成最终台账: {final_csv}")
    print(f"[OK] 已生成问题摘要: {problem_csv}")
    print(f"[OK] 已写回采用配置: {override_yaml}")
    print(f"[OK] 已生成配置摘要: {summary_json}")
    print_run_summary(adopted_config, final_rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("[ERROR] 用户中断")
    except BrokenPipeError:
        sys.stderr.close()
