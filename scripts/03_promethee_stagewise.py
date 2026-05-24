#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entropy-weighted PROMETHEE scores from stagewise-selected factors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stagewise_utils import LABEL_COL, OUTPUT_DIR, ensure_dirs, fail, get_stock_id_col, get_stock_name_col, minmax_high_good


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n-promethee-features", type=int, default=20)
    return parser.parse_args()


def entropy_weights(x: pd.DataFrame) -> pd.DataFrame:
    n = len(x)
    if n <= 1 or x.empty:
        return pd.DataFrame(columns=["feature", "entropy", "weight"])
    rows = []
    d_values = []
    for col in x.columns:
        values = pd.to_numeric(x[col], errors="coerce").fillna(0.5).clip(0.0, 1.0).to_numpy(float)
        total = values.sum()
        if total <= 1e-12:
            entropy = 1.0
        else:
            p = values / total
            p = p[p > 0]
            entropy = float(-(p * np.log(p)).sum() / np.log(n))
        d = max(0.0, 1.0 - entropy)
        rows.append({"feature": col, "entropy": entropy})
        d_values.append(d)
    d_sum = float(np.sum(d_values))
    weights = np.ones(len(rows)) / len(rows) if d_sum <= 1e-12 else np.asarray(d_values) / d_sum
    for row, weight in zip(rows, weights):
        row["weight"] = float(weight)
    return pd.DataFrame(rows)


def promethee_flows(beneficial: pd.DataFrame, weights: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(beneficial)
    phi_plus = np.zeros(n, dtype=float)
    phi_minus = np.zeros(n, dtype=float)
    if n <= 1:
        return phi_plus, phi_minus, phi_plus - phi_minus
    for col in beneficial.columns:
        w = float(weights[col])
        values = beneficial[col].to_numpy(float)
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        cumsum = np.cumsum(sorted_values)
        total = cumsum[-1]
        k = np.arange(n)
        below_sum = np.concatenate([[0.0], cumsum[:-1]])
        above_sum = total - cumsum
        plus_sorted = sorted_values * k - below_sum
        minus_sorted = above_sum - sorted_values * (n - k - 1)
        plus = np.empty(n)
        minus = np.empty(n)
        plus[order] = plus_sorted
        minus[order] = minus_sorted
        phi_plus += w * plus / (n - 1)
        phi_minus += w * minus / (n - 1)
    return phi_plus, phi_minus, phi_plus - phi_minus


def selected_features_for_quarter(selected: pd.DataFrame, q: int, top_n: int) -> pd.DataFrame:
    sq = selected.loc[selected["quarter_idx"].eq(q)].copy()
    if sq.empty:
        past = selected.loc[selected["quarter_idx"].lt(q)].copy()
        if past.empty:
            return sq
        freq = past.groupby("selected_feature").agg(
            selected_count=("quarter_idx", "count"),
            avg_coef=("coefficient", "mean"),
            avg_abs_coef=("coefficient", lambda x: float(np.mean(np.abs(x)))),
        )
        sq = freq.sort_values(["selected_count", "avg_abs_coef"], ascending=[False, False]).head(top_n).reset_index()
        sq["coefficient"] = sq["avg_coef"]
    else:
        sq = sq.sort_values(["step"]).drop_duplicates("selected_feature", keep="last").head(top_n)
    return sq


def main() -> None:
    args = parse_args()
    ensure_dirs()
    panel_path = OUTPUT_DIR / "model_ready_stagewise_panel.parquet"
    selected_path = OUTPUT_DIR / "stagewise_selected_features_by_quarter.csv"
    pred_path = OUTPUT_DIR / "stagewise_prediction_panel.csv"
    if not panel_path.exists() or not selected_path.exists() or not pred_path.exists():
        raise fail("请先运行 01_build_stagewise_composite_factors.py 和 02_forward_stagewise_selection.py。")
    panel = pd.read_parquet(panel_path)
    selected = pd.read_csv(selected_path, encoding="utf-8-sig")
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    stock_id_col = get_stock_id_col(panel)
    stock_name_col = get_stock_name_col(panel)

    score_rows = []
    weight_rows = []
    for q in sorted(pred["quarter_idx"].unique()):
        q = int(q)
        group = panel.loc[panel["quarter_idx"].eq(q)].copy()
        if group.empty:
            continue
        sq = selected_features_for_quarter(selected, q, args.top_n_promethee_features)
        features = [f for f in sq["selected_feature"].astype(str).tolist() if f in group.columns]
        if not features:
            continue
        coef_by_feature = sq.set_index("selected_feature")["coefficient"].astype(float).to_dict()
        beneficial = pd.DataFrame(index=group.index)
        for feature in features:
            raw = pd.to_numeric(group[feature], errors="coerce")
            median = raw.median()
            raw = raw.fillna(median if np.isfinite(median) else 0.0)
            direction = "positive" if coef_by_feature.get(feature, 0.0) >= 0 else "negative"
            signed = raw if direction == "positive" else -raw
            beneficial[feature] = minmax_high_good(signed)
        weights = entropy_weights(beneficial)
        if weights.empty:
            continue
        weight_series = weights.set_index("feature")["weight"]
        plus, minus, net = promethee_flows(beneficial[weights["feature"].tolist()], weight_series)
        out = group[["quarter_idx", stock_id_col] + ([stock_name_col] if stock_name_col else [])].copy()
        out["promethee_phi_plus"] = plus
        out["promethee_phi_minus"] = minus
        out["promethee_net_flow"] = net
        out["promethee_rank"] = out["promethee_net_flow"].rank(method="first", ascending=False)
        out = out.rename(columns={stock_id_col: "stock_id"})
        if stock_name_col:
            out = out.rename(columns={stock_name_col: "stock_name"})
        score_rows.append(out)
        for _, row in weights.iterrows():
            coef = coef_by_feature.get(row["feature"], 0.0)
            weight_rows.append(
                {
                    "quarter_idx": q,
                    "feature": row["feature"],
                    "entropy": row["entropy"],
                    "weight": row["weight"],
                    "direction": "positive" if coef >= 0 else "negative",
                    "coefficient_for_direction": coef,
                }
            )
        print(f"quarter={q} promethee_features={len(features)}")

    if not score_rows:
        raise fail("未生成 PROMETHEE 评分。")
    pd.concat(score_rows, ignore_index=True).to_csv(OUTPUT_DIR / "promethee_scores_stagewise.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weight_rows).to_csv(OUTPUT_DIR / "entropy_weights_stagewise.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
