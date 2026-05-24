#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the model-ready panel for the stagewise 200+ composite-factor route."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stagewise_utils import (
    FIGURE_DIR,
    LABEL_COL,
    MACRO_FACTORS,
    META_FIELDS,
    OUTPUT_DIR,
    RETURN_FIELDS,
    ensure_dirs,
    configure_matplotlib_chinese,
    fail,
    get_stock_id_col,
    get_stock_name_col,
    is_return_like,
    safe_feature_name,
    split_name,
)


ROOT = Path(__file__).resolve().parents[1]


def find_stock_excel() -> Path | None:
    raw_dir = OUTPUT_DIR / "extracted_project" / "原始数据"
    candidates = sorted(raw_dir.glob("因子汇总+月平均收益率*24个因子*.xlsx"))
    return candidates[0] if candidates else None


def read_stock_panel() -> pd.DataFrame:
    p0_path = OUTPUT_DIR / "stock_panel.parquet"
    if p0_path.exists():
        panel = pd.read_parquet(p0_path)
        source = str(p0_path.relative_to(ROOT))
    else:
        excel_path = find_stock_excel()
        if excel_path is None:
            raise fail("未找到 P0 stock_panel.parquet 或原始个股 Excel。")
        xls = pd.ExcelFile(excel_path)
        parts = []
        for sheet in xls.sheet_names:
            match = re.search(r"(\d+)", sheet)
            if not match:
                continue
            frame = pd.read_excel(excel_path, sheet_name=sheet)
            frame = frame.rename(columns={"Unnamed: 0": "row_id_raw"})
            frame.insert(0, "quarter_idx", int(match.group(1)))
            frame.insert(1, "sheet_name", sheet)
            parts.append(frame)
        if not parts:
            raise fail("原始个股 Excel 未识别到季度 sheet。")
        panel = pd.concat(parts, ignore_index=True)
        source = str(excel_path.relative_to(ROOT))
    panel.attrs["source"] = source
    return panel


def read_macro_quarterly() -> pd.DataFrame:
    aligned_path = OUTPUT_DIR / "macro_quarterly_aligned.csv"
    if aligned_path.exists():
        macro = pd.read_csv(aligned_path, encoding="utf-8-sig")
        source = str(aligned_path.relative_to(ROOT))
    else:
        excel_path = OUTPUT_DIR / "extracted_project" / "原始数据" / "宏观因子数据.xlsx"
        if not excel_path.exists():
            raise fail("未找到宏观季度对齐表或原始宏观 Excel。")
        sheets = pd.ExcelFile(excel_path).sheet_names
        data_sheet = "宏观因子具体数据" if "宏观因子具体数据" in sheets else sheets[-1]
        macro = pd.read_excel(excel_path, sheet_name=data_sheet)
        source = str(excel_path.relative_to(ROOT))
    macro = macro.copy()
    missing = [c for c in MACRO_FACTORS if c not in macro.columns]
    if missing:
        raise fail(f"宏观数据缺少可验证字段: {missing}")
    if "quarter_idx" not in macro.columns:
        macro.insert(0, "quarter_idx", np.arange(1, len(macro) + 1))
    macro.attrs["source"] = source
    return macro[["quarter_idx"] + [c for c in ["quarter_period", "date"] if c in macro.columns] + MACRO_FACTORS]


def audit_stock_factors(panel: pd.DataFrame) -> tuple[list[str], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    stock_factors: list[str] = []
    for col in panel.columns:
        reason = ""
        included = False
        if col in META_FIELDS or col in RETURN_FIELDS:
            reason = "meta_or_explicit_return_field"
        elif is_return_like(col):
            reason = "return_like_or_future_like_field"
        elif not pd.api.types.is_numeric_dtype(panel[col]):
            reason = "non_numeric"
        elif col.startswith("macro_") or col in MACRO_FACTORS:
            reason = "macro_field"
        else:
            included = True
            reason = "base_stock_factor"
            stock_factors.append(col)
        rows.append({"column": col, "included_as_stock_factor": included, "reason": reason})
    return stock_factors, rows


def fit_transform_train_only(panel: pd.DataFrame, cols: list[str], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = panel["quarter_idx"].isin(range(1, 22))
    params = []
    out = pd.DataFrame(index=panel.index)
    for col in cols:
        raw = pd.to_numeric(panel[col], errors="coerce")
        train = raw.loc[train_mask]
        median = float(train.median())
        filled = raw.fillna(median)
        lo = float(filled.loc[train_mask].quantile(0.01))
        hi = float(filled.loc[train_mask].quantile(0.99))
        clipped = filled.clip(lo, hi)
        mean = float(clipped.loc[train_mask].mean())
        std = float(clipped.loc[train_mask].std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        std_col = f"{prefix}{safe_feature_name(col)}"
        out[std_col] = (clipped - mean) / std
        params.append(
            {
                "source_column": col,
                "standardized_column": std_col,
                "median_train": median,
                "winsor_p01_train": lo,
                "winsor_p99_train": hi,
                "mean_train": mean,
                "std_train": std,
            }
        )
    return out, pd.DataFrame(params)


def plot_factor_count(stock_count: int, macro_count: int, composite_count: int) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib_chinese()
    FIGURE_DIR.mkdir(exist_ok=True)
    labels = ["stock_factors", "macro_factors", "composite_factors"]
    values = [stock_count, macro_count, composite_count]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Stagewise factor count")
    ax.set_ylabel("count")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_composite_factor_count.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    stock = read_stock_panel()
    macro = read_macro_quarterly()
    if LABEL_COL not in stock.columns:
        if "月平均收益率" not in stock.columns:
            raise fail("无法构造 future_return_1q：缺少 月平均收益率。")
        stock[LABEL_COL] = pd.to_numeric(stock["月平均收益率"], errors="coerce")
    stock["split"] = stock["quarter_idx"].map(lambda q: split_name(int(q)))
    stock_id_col = get_stock_id_col(stock)
    stock_name_col = get_stock_name_col(stock)

    stock_factors, audit_rows = audit_stock_factors(stock)
    if len(stock_factors) < 18:
        raise fail(f"可用个股因子过少: {len(stock_factors)}")

    panel = stock.merge(macro, on="quarter_idx", how="left", validate="many_to_one")
    missing_macro = panel[MACRO_FACTORS].isna().sum().sum()
    if missing_macro:
        raise fail(f"宏观因子合并后存在缺失值数量: {missing_macro}")

    stock_std, stock_params = fit_transform_train_only(panel, stock_factors, "std_stock_")
    macro_std, macro_params = fit_transform_train_only(panel, MACRO_FACTORS, "std_macro_")
    model_panel = pd.concat([panel.reset_index(drop=True), stock_std.reset_index(drop=True), macro_std.reset_index(drop=True)], axis=1)

    composite_rows = []
    composite_data = {}
    for stock_factor in stock_factors:
        stock_std_col = stock_params.loc[stock_params["source_column"].eq(stock_factor), "standardized_column"].iloc[0]
        for macro_factor in MACRO_FACTORS:
            macro_std_col = macro_params.loc[macro_params["source_column"].eq(macro_factor), "standardized_column"].iloc[0]
            comp_col = f"composite_{safe_feature_name(stock_factor)}_x_{macro_factor}"
            composite_data[comp_col] = model_panel[stock_std_col] * model_panel[macro_std_col]
            composite_rows.append(
                {
                    "composite_feature": comp_col,
                    "stock_factor": stock_factor,
                    "macro_factor": macro_factor,
                    "stock_standardized_column": stock_std_col,
                    "macro_standardized_column": macro_std_col,
                    "formula": f"{stock_std_col} * {macro_std_col}",
                    "is_composite": True,
                }
            )

    composite_list = pd.DataFrame(composite_rows)
    model_panel = pd.concat([model_panel, pd.DataFrame(composite_data, index=model_panel.index)], axis=1)
    factor_audit = pd.DataFrame(audit_rows)
    scaler_params = pd.concat(
        [
            stock_params.assign(group="stock_factor"),
            macro_params.assign(group="macro_factor"),
        ],
        ignore_index=True,
    )
    out_cols = [
        "quarter_idx",
        "split",
        stock_id_col,
        *([stock_name_col] if stock_name_col else []),
        LABEL_COL,
        *[c for c in ["sheet_name", "row_id_raw", "quarter_period", "date"] if c in model_panel.columns],
        *stock_factors,
        *MACRO_FACTORS,
        *stock_std.columns.tolist(),
        *macro_std.columns.tolist(),
        *composite_list["composite_feature"].tolist(),
    ]
    model_panel = model_panel.loc[:, list(dict.fromkeys(out_cols))]
    model_panel.to_parquet(OUTPUT_DIR / "model_ready_stagewise_panel.parquet", index=False)
    composite_list.to_csv(OUTPUT_DIR / "composite_factor_list.csv", index=False, encoding="utf-8-sig")
    factor_audit.to_csv(OUTPUT_DIR / "stagewise_factor_audit.csv", index=False, encoding="utf-8-sig")
    scaler_params.to_csv(OUTPUT_DIR / "stagewise_preprocess_params.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stock_source": stock.attrs.get("source"),
        "macro_source": macro.attrs.get("source"),
        "stock_rows": int(len(stock)),
        "quarter_count": int(model_panel["quarter_idx"].nunique()),
        "stock_factor_count": int(len(stock_factors)),
        "macro_factor_count": int(len(MACRO_FACTORS)),
        "composite_factor_count": int(len(composite_list)),
        "label_definition": "future_return_1q = 原始月平均收益率，不做二次 shift",
        "macro_factor_note": "作业说明中提到 12 个宏观因子，但实际提供文件仅能识别 11 个具名宏观变量。",
    }
    (OUTPUT_DIR / "stagewise_data_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_factor_count(len(stock_factors), len(MACRO_FACTORS), len(composite_list))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
