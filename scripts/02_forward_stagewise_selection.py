#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rolling-window forward stepwise regression on 200+ composite factors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stagewise_utils import (
    FIGURE_DIR,
    LABEL_COL,
    OUTPUT_DIR,
    configure_matplotlib_chinese,
    ensure_dirs,
    fail,
    get_stock_id_col,
    get_stock_name_col,
    rank_ic,
    regression_metrics,
    split_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--max-features", type=int, default=15)
    parser.add_argument("--min-improvement", type=float, default=1e-6)
    parser.add_argument("--min-abs-corr", type=float, default=0.0)
    return parser.parse_args()


def standardize_window(train_x: pd.DataFrame, pred_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    return ((train_x - mean) / std).to_numpy(float), ((pred_x - mean) / std).to_numpy(float), mean, std


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ beta
    return beta, pred


def select_for_quarter(
    panel: pd.DataFrame,
    q: int,
    candidate_features: list[str],
    window_size: int,
    max_features: int,
    min_improvement: float,
    min_abs_corr: float,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    train_quarters = list(range(q - window_size, q))
    train = panel.loc[panel["quarter_idx"].isin(train_quarters)].copy()
    pred_panel = panel.loc[panel["quarter_idx"].eq(q)].copy()
    if train.empty or pred_panel.empty:
        return pd.DataFrame(), [], {}
    train = train.dropna(subset=[LABEL_COL])
    y = pd.to_numeric(train[LABEL_COL], errors="coerce").to_numpy(float)
    valid_y = np.isfinite(y)
    train = train.loc[valid_y].copy()
    y = y[valid_y]
    if len(y) < 20:
        return pd.DataFrame(), [], {}

    train_x = train[candidate_features].apply(pd.to_numeric, errors="coerce")
    pred_x = pred_panel[candidate_features].apply(pd.to_numeric, errors="coerce")
    med = train_x.median(axis=0).fillna(0.0)
    train_x = train_x.fillna(med)
    pred_x = pred_x.fillna(med)
    x_std, pred_std, _, _ = standardize_window(train_x, pred_x)

    selected: list[int] = []
    remaining = list(range(len(candidate_features)))
    residual = y - y.mean()
    prev_sse = float(np.sum(residual**2))
    step_rows: list[dict[str, object]] = []
    final_beta = np.array([y.mean()])
    final_train_pred = np.full_like(y, y.mean(), dtype=float)

    for step in range(1, max_features + 1):
        if not remaining:
            break
        residual_centered = residual - residual.mean()
        x_rem = x_std[:, remaining]
        x_rem_centered = x_rem - x_rem.mean(axis=0)
        denom = np.sqrt((x_rem_centered**2).sum(axis=0) * np.sum(residual_centered**2))
        corr = np.divide(x_rem_centered.T @ residual_centered, denom, out=np.zeros(len(remaining)), where=denom > 1e-12)
        best_pos = int(np.nanargmax(np.abs(corr)))
        best_corr = float(corr[best_pos])
        if abs(best_corr) < min_abs_corr:
            break
        best_feature_idx = remaining[best_pos]
        trial_selected = selected + [best_feature_idx]
        beta, train_pred = fit_ols(x_std[:, trial_selected], y)
        sse = float(np.sum((y - train_pred) ** 2))
        improvement = (prev_sse - sse) / prev_sse if prev_sse > 1e-12 else 0.0
        if step > 1 and improvement < min_improvement:
            break
        selected = trial_selected
        remaining.remove(best_feature_idx)
        residual = y - train_pred
        prev_sse = sse
        final_beta = beta
        final_train_pred = train_pred
        metrics = regression_metrics(y, train_pred)
        step_rows.append(
            {
                "quarter_idx": q,
                "window_start": q - window_size,
                "window_end": q - 1,
                "step": step,
                "selected_feature": candidate_features[best_feature_idx],
                "coefficient": float(beta[-1]),
                "train_r2": metrics["r2"],
                "train_rankic": rank_ic(y, train_pred),
                "selection_abs_corr": abs(best_corr),
                "sse_improvement": improvement,
                "valid_rankic_if_available": np.nan,
            }
        )

    if not selected:
        pred = np.full(len(pred_panel), y.mean(), dtype=float)
        selected_features: list[str] = []
    else:
        pred_design = np.column_stack([np.ones(len(pred_panel)), pred_std[:, selected]])
        pred = pred_design @ final_beta
        selected_features = [candidate_features[i] for i in selected]
    pred_out = pred_panel.copy()
    pred_out["stagewise_pred"] = pred
    pred_out["stagewise_rank"] = pred_out["stagewise_pred"].rank(method="first", ascending=False)
    pred_out["stagewise_rank_pct"] = pred_out["stagewise_pred"].rank(method="average", pct=True)
    pred_out["selected_feature_count"] = len(selected_features)
    pred_out["selected_features"] = ";".join(selected_features)
    q_rankic = rank_ic(pred_out[LABEL_COL], pred_out["stagewise_pred"])
    if split_name(q) == "valid":
        for row in step_rows:
            row["valid_rankic_if_available"] = q_rankic
    quarter_summary = {
        "quarter_idx": q,
        "split": split_name(q),
        "rankic": q_rankic,
        "n_stocks": int(pred_out[LABEL_COL].notna().sum()),
        "selected_feature_count": len(selected_features),
    }
    return pred_out, step_rows, quarter_summary


def feature_frequency(selected: pd.DataFrame, composite_list: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    coef_stats = selected.groupby("selected_feature").agg(
        selected_count=("quarter_idx", "count"),
        avg_abs_coefficient=("coefficient", lambda x: float(np.mean(np.abs(x)))),
        first_selected_avg_step=("step", "mean"),
    )
    freq = coef_stats.reset_index().rename(columns={"selected_feature": "feature"})
    meta = composite_list.rename(columns={"composite_feature": "feature"})
    freq = freq.merge(meta[["feature", "stock_factor", "macro_factor", "is_composite"]], on="feature", how="left")
    freq["is_composite"] = freq["is_composite"].fillna(freq["feature"].str.startswith("composite_"))
    freq["feature_group"] = np.where(freq["is_composite"], "macro_micro_composite", "other")
    return freq.sort_values(["selected_count", "avg_abs_coefficient"], ascending=[False, False])


def build_model_summary(pred: pd.DataFrame, rankic_q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in pred.groupby("split", sort=False):
        metrics = regression_metrics(group[LABEL_COL], group["stagewise_pred"])
        rows.extend({"split": split, "metric": k, "value": v} for k, v in metrics.items())
        rows.append({"split": split, "metric": "mean_rankic", "value": float(rankic_q.loc[rankic_q["split"].eq(split), "rankic"].mean())})
        for k in [10, 20, 30]:
            top = group.sort_values(["quarter_idx", "stagewise_pred"], ascending=[True, False]).groupby("quarter_idx").head(k)
            rows.append({"split": split, "metric": f"top{k}_avg_return", "value": float(top[LABEL_COL].mean())})
    return pd.DataFrame(rows)


def plot_outputs(selected: pd.DataFrame, rankic: pd.DataFrame, pred: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib_chinese()
    FIGURE_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    steps = ["P0 data audit", "231 composites", "rolling stepwise", "PROMETHEE", "rating & backtest"]
    ax.set_axis_off()
    for i, step in enumerate(steps):
        x = i / (len(steps) - 1)
        ax.text(x, 0.55, step, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#F6F8FA", ec="#4C78A8"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 0.13, 0.55), xytext=(x + 0.06, 0.55), arrowprops=dict(arrowstyle="->", color="#4C78A8"))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_stagewise_pipeline.png", dpi=180)
    plt.close(fig)

    top = selected["selected_feature"].value_counts().head(20).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot(kind="barh", ax=ax, color="#4C78A8")
    ax.set_title("Top selected composite factors")
    ax.set_xlabel("selected count")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_stagewise_top_feature_frequency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for split, group in rankic.groupby("split"):
        ax.plot(group["quarter_idx"], group["rankic"], marker="o", label=split)
    ax.axhline(0, color="#999999", linewidth=1)
    ax.set_title("Quarterly RankIC")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("RankIC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_stagewise_quarterly_rankic.png", dpi=180)
    plt.close(fig)

    heat = selected.pivot_table(index="selected_feature", columns="quarter_idx", values="coefficient", aggfunc="mean").fillna(0)
    heat = heat.loc[selected["selected_feature"].value_counts().head(20).index.intersection(heat.index)]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(heat))))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="RdBu_r")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([str(x)[:42] for x in heat.index], fontsize=7)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, fontsize=7)
    ax.set_title("Stagewise coefficient heatmap")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_stagewise_coefficient_heatmap.png", dpi=180)
    plt.close(fig)

    ret_rows = []
    for k in [10, 20, 30]:
        r = pred.sort_values(["quarter_idx", "stagewise_pred"], ascending=[True, False]).groupby("quarter_idx").head(k)
        qret = r.groupby("quarter_idx")[LABEL_COL].mean().reset_index(name=f"top{k}")
        ret_rows.append(qret)
    perf = ret_rows[0]
    for frame in ret_rows[1:]:
        perf = perf.merge(frame, on="quarter_idx", how="outer")
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in [c for c in perf.columns if c != "quarter_idx"]:
        ax.plot(perf["quarter_idx"], (1 + perf[col].fillna(0) / 100).cumprod() - 1, marker="o", label=col)
    ax.set_title("Stagewise Top-K cumulative performance")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("cumulative return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_stagewise_topk_performance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    panel_path = OUTPUT_DIR / "model_ready_stagewise_panel.parquet"
    list_path = OUTPUT_DIR / "composite_factor_list.csv"
    if not panel_path.exists() or not list_path.exists():
        raise fail("请先运行 scripts/01_build_stagewise_composite_factors.py")
    panel = pd.read_parquet(panel_path)
    composite_list = pd.read_csv(list_path, encoding="utf-8-sig")
    candidates = composite_list["composite_feature"].astype(str).tolist()
    missing = [c for c in candidates if c not in panel.columns]
    if missing:
        raise fail(f"模型面板缺少综合因子列: {missing[:5]}")
    stock_id_col = get_stock_id_col(panel)
    stock_name_col = get_stock_name_col(panel)

    pred_parts = []
    selected_rows = []
    rankic_rows = []
    start_q = 1 + args.window_size
    for q in range(start_q, 31):
        pred_q, selected_q, summary_q = select_for_quarter(
            panel,
            q,
            candidates,
            args.window_size,
            args.max_features,
            args.min_improvement,
            args.min_abs_corr,
        )
        if not pred_q.empty:
            pred_parts.append(pred_q)
            selected_rows.extend(selected_q)
            rankic_rows.append(summary_q)
            print(f"quarter={q} split={summary_q['split']} rankic={summary_q['rankic']:.4f} features={summary_q['selected_feature_count']}")

    if not pred_parts:
        raise fail("未生成任何滚动预测。")
    pred = pd.concat(pred_parts, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    rankic = pd.DataFrame(rankic_rows)
    if selected.empty:
        raise fail("前向分步回归未选择任何变量，请调低阈值。")

    keep_cols = ["quarter_idx", "split", stock_id_col]
    if stock_name_col:
        keep_cols.append(stock_name_col)
    keep_cols += [LABEL_COL, "stagewise_pred", "stagewise_rank", "stagewise_rank_pct", "selected_feature_count", "selected_features"]
    pred_out = pred[keep_cols].copy()
    pred_out = pred_out.rename(columns={stock_id_col: "stock_id"})
    if stock_name_col:
        pred_out = pred_out.rename(columns={stock_name_col: "stock_name"})
    selected.to_csv(OUTPUT_DIR / "stagewise_selected_features_by_quarter.csv", index=False, encoding="utf-8-sig")
    feature_frequency(selected, composite_list).to_csv(OUTPUT_DIR / "stagewise_feature_frequency.csv", index=False, encoding="utf-8-sig")
    pred_out.to_csv(OUTPUT_DIR / "stagewise_prediction_panel.csv", index=False, encoding="utf-8-sig")
    rankic.to_csv(OUTPUT_DIR / "stagewise_quarterly_rankic.csv", index=False, encoding="utf-8-sig")
    build_model_summary(pred_out, rankic).to_csv(OUTPUT_DIR / "stagewise_model_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window_size": args.window_size,
        "max_features": args.max_features,
        "candidate_feature_count": len(candidates),
        "candidate_scope": "composite factors only",
    }
    (OUTPUT_DIR / "stagewise_model_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_outputs(selected, rankic, pred_out)


if __name__ == "__main__":
    main()
