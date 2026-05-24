#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hybrid scores, Top-K backtests, and 13-grade ratings for stagewise outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stagewise_utils import (
    FIGURE_DIR,
    LABEL_COL,
    OUTPUT_DIR,
    RATING_LABELS,
    configure_matplotlib_chinese,
    ensure_dirs,
    fail,
    max_drawdown,
    performance_metrics,
)


TOPK_PLAN = {
    "pure_stagewise_score": [10, 20, 30],
    "pure_promethee_score": [10, 20, 30],
    "hybrid_score_50_50": [10, 20, 30],
    "hybrid_score_30_70": [10, 20],
    "hybrid_score_70_30": [10, 20],
}


def short_strategy_name(strategy: str) -> str:
    return (
        str(strategy)
        .replace("_score", "")
        .replace("pure_", "")
        .replace("hybrid_", "hybrid ")
        .replace("_top", " top")
        .replace("_", " ")
    )


def normalize_stock_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stock_id"] = out["stock_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return out


def build_hybrid_scores() -> pd.DataFrame:
    pred_path = OUTPUT_DIR / "stagewise_prediction_panel.csv"
    pro_path = OUTPUT_DIR / "promethee_scores_stagewise.csv"
    if not pred_path.exists() or not pro_path.exists():
        raise fail("请先运行 02_forward_stagewise_selection.py 和 03_promethee_stagewise.py。")
    pred = normalize_stock_id(pd.read_csv(pred_path, encoding="utf-8-sig", dtype={"stock_id": str}))
    pro = normalize_stock_id(pd.read_csv(pro_path, encoding="utf-8-sig", dtype={"stock_id": str}))
    panel = pred.merge(pro, on=["quarter_idx", "stock_id", "stock_name"], how="inner", validate="one_to_one")
    panel["pure_stagewise_score"] = panel.groupby("quarter_idx")["stagewise_pred"].rank(method="average", pct=True)
    panel["pure_promethee_score"] = panel.groupby("quarter_idx")["promethee_net_flow"].rank(method="average", pct=True)
    panel["hybrid_score_70_30"] = 0.7 * panel["pure_promethee_score"] + 0.3 * panel["pure_stagewise_score"]
    panel["hybrid_score_50_50"] = 0.5 * panel["pure_promethee_score"] + 0.5 * panel["pure_stagewise_score"]
    panel["hybrid_score_30_70"] = 0.3 * panel["pure_promethee_score"] + 0.7 * panel["pure_stagewise_score"]
    score_cols = list(TOPK_PLAN)
    keep = [
        "quarter_idx",
        "split",
        "stock_id",
        "stock_name",
        LABEL_COL,
        "stagewise_pred",
        "promethee_net_flow",
        *score_cols,
    ]
    panel[keep].to_csv(OUTPUT_DIR / "hybrid_scores_stagewise.csv", index=False, encoding="utf-8-sig")
    return panel


def build_topk(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings = []
    returns = []
    benchmark = panel.groupby("quarter_idx", as_index=False)[LABEL_COL].mean().rename(columns={LABEL_COL: "benchmark_return"})
    for score_col, topks in TOPK_PLAN.items():
        for k in topks:
            strategy = f"{score_col}_top{k}"
            prev_weights: dict[str, float] = {}
            for q, group in panel.groupby("quarter_idx", sort=True):
                ranked = group.dropna(subset=[score_col, LABEL_COL]).sort_values([score_col, "stock_id"], ascending=[False, True]).head(k).copy()
                if ranked.empty:
                    continue
                ranked["portfolio_rank"] = np.arange(1, len(ranked) + 1)
                ranked["weight"] = 1.0 / len(ranked)
                current_weights = dict(zip(ranked["stock_id"], ranked["weight"]))
                names = set(prev_weights) | set(current_weights)
                turnover = 0.5 * sum(abs(current_weights.get(name, 0.0) - prev_weights.get(name, 0.0)) for name in names)
                prev_weights = current_weights
                port_ret = float((ranked["weight"] * ranked[LABEL_COL]).sum())
                returns.append(
                    {
                        "quarter_idx": q,
                        "split": ranked["split"].iloc[0],
                        "strategy": strategy,
                        "score_col": score_col,
                        "topk": k,
                        "portfolio_return": port_ret,
                        "benchmark_return": float(benchmark.loc[benchmark["quarter_idx"].eq(q), "benchmark_return"].iloc[0]),
                        "excess_return": port_ret - float(benchmark.loc[benchmark["quarter_idx"].eq(q), "benchmark_return"].iloc[0]),
                        "turnover": turnover,
                        "holding_count": int(len(ranked)),
                    }
                )
                holdings.append(
                    ranked[["quarter_idx", "split", "stock_id", "stock_name", LABEL_COL, score_col, "portfolio_rank", "weight"]]
                    .assign(strategy=strategy, score_col=score_col, topk=k)
                    .rename(columns={score_col: "score"})
                )
    holdings_df = pd.concat(holdings, ignore_index=True)
    returns_df = pd.DataFrame(returns)
    metrics_rows = []
    for split_name, frame in [("all", returns_df), *list(returns_df.groupby("split"))]:
        wide = frame.pivot_table(index="quarter_idx", columns="strategy", values="portfolio_return")
        bench = frame.groupby("quarter_idx")["benchmark_return"].first()
        wide = wide.merge(bench, left_index=True, right_index=True, how="left")
        strategy_cols = [c for c in wide.columns if c != "benchmark_return"]
        metric = performance_metrics(wide.reset_index(), strategy_cols)
        if metric.empty:
            continue
        turn = frame.groupby("strategy")["turnover"].mean().rename("average_turnover")
        metric = metric.merge(turn, on="strategy", how="left")
        metric.insert(0, "split", split_name)
        metrics_rows.append(metric)
    metrics_df = pd.concat(metrics_rows, ignore_index=True)
    return holdings_df, returns_df, metrics_df


def build_rating(panel: pd.DataFrame, score_col: str = "hybrid_score_50_50") -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for q, group in panel.groupby("quarter_idx", sort=True):
        ranked = group.sort_values([score_col, "stock_id"], ascending=[False, True]).copy()
        n = len(ranked)
        order = np.arange(n)
        grade_idx = np.floor(order * len(RATING_LABELS) / n).astype(int).clip(0, len(RATING_LABELS) - 1)
        ranked["rating"] = [RATING_LABELS[i] for i in grade_idx]
        ranked["rating_rank_in_quarter"] = order + 1
        ranked["rating_score_source"] = score_col
        parts.append(ranked)
    rating = pd.concat(parts, ignore_index=True)
    perf = (
        rating.groupby(["split", "rating"], as_index=False)
        .agg(
            avg_future_return=(LABEL_COL, "mean"),
            median_future_return=(LABEL_COL, "median"),
            stock_count=("stock_id", "count"),
            positive_rate=(LABEL_COL, lambda x: float((x > 0).mean())),
        )
    )
    order = {label: i for i, label in enumerate(RATING_LABELS)}
    perf["rating_order"] = perf["rating"].map(order)
    perf = perf.sort_values(["split", "rating_order"])
    return rating, perf


def add_heatmap_labels(ax, values: np.ndarray, fmt: str = ".2f", fontsize: int = 7) -> None:
    finite = values[np.isfinite(values)]
    threshold = float(np.nanmean(finite)) if finite.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if value > threshold else "#222222"
            ax.text(j, i, format(value, fmt), ha="center", va="center", color=color, fontsize=fontsize)


def plot_outputs(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    metrics: pd.DataFrame,
    rating: pd.DataFrame,
    rating_perf: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib_chinese()
    FIGURE_DIR.mkdir(exist_ok=True)
    focus = [
        "pure_stagewise_score_top10",
        "pure_promethee_score_top10",
        "hybrid_score_50_50_top10",
        "hybrid_score_30_70_top10",
        "hybrid_score_70_30_top10",
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    for strategy in focus:
        r = returns.loc[returns["strategy"].eq(strategy)].sort_values("quarter_idx")
        if not r.empty:
            ax.plot(r["quarter_idx"], (1 + r["portfolio_return"].fillna(0) / 100).cumprod() - 1, marker="o", label=strategy.replace("_score", ""))
    ax.set_title("Top-K cumulative return")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("cumulative return")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_topk_cumulative_return_stagewise.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for strategy in focus[:3]:
        r = returns.loc[returns["strategy"].eq(strategy)].sort_values("quarter_idx")
        if not r.empty:
            ax.plot(r["quarter_idx"], r["portfolio_return"], marker="o", label=strategy.replace("_score", ""))
    ax.axhline(0, color="#999999", linewidth=1)
    ax.set_title("Top-K quarterly return")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("return (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_topk_quarterly_return_stagewise.png", dpi=180)
    plt.close(fig)

    all_metrics = metrics.loc[metrics["split"].eq("all")].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(all_metrics["annualized_volatility"], all_metrics["annualized_return"], s=35, color="#4C78A8")
    for _, row in all_metrics.iterrows():
        ax.text(row["annualized_volatility"], row["annualized_return"], row["strategy"].replace("_score", ""), fontsize=6)
    ax.set_title("Risk-return scatter")
    ax.set_xlabel("annualized volatility")
    ax.set_ylabel("annualized return")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_topk_risk_return_scatter_stagewise.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for strategy in focus:
        r = returns.loc[returns["strategy"].eq(strategy)].sort_values("quarter_idx")
        if not r.empty:
            ax.plot(r["quarter_idx"], r["turnover"], marker="o", label=strategy.replace("_score", ""))
    ax.set_title("Top-K turnover")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("one-way turnover")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_turnover_stagewise.png", dpi=180)
    plt.close(fig)

    test_perf = rating_perf.loc[rating_perf["split"].eq("test")].copy()
    if test_perf.empty:
        test_perf = rating_perf.groupby("rating", as_index=False)["avg_future_return"].mean()
        test_perf["rating_order"] = test_perf["rating"].map({label: i for i, label in enumerate(RATING_LABELS)})
    test_perf = test_perf.sort_values("rating_order")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(test_perf["rating"], test_perf["avg_future_return"], color="#54A24B")
    ax.set_title("Rating return by grade")
    ax.set_xlabel("rating")
    ax.set_ylabel("avg future return (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_rating_return_by_grade.png", dpi=180)
    plt.close(fig)

    count = rating.groupby(["rating"]).size().reindex(RATING_LABELS)
    fig, ax = plt.subplots(figsize=(9, 4))
    count.plot(kind="bar", ax=ax, color="#F58518")
    ax.set_title("Rating count by grade")
    ax.set_xlabel("rating")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_rating_count_by_grade.png", dpi=180)
    plt.close(fig)

    focus_returns = returns.loc[returns["strategy"].isin(focus)].copy()
    if not focus_returns.empty:
        heat = focus_returns.pivot_table(index="strategy", columns="quarter_idx", values="excess_return", aggfunc="mean")
        heat = heat.reindex(focus)
        values = heat.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        lim = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-lim, vmax=lim)
        ax.set_title("Top-K excess return heatmap")
        ax.set_xlabel("quarter_idx")
        ax.set_ylabel("strategy")
        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels(heat.columns, fontsize=7)
        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels([short_strategy_name(x) for x in heat.index], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.025, label="excess return (%)")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fig_topk_excess_return_heatmap_stagewise.png", dpi=180)
        plt.close(fig)

    all_metrics = metrics.loc[metrics["split"].eq("all")].copy()
    metric_cols = [
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "average_turnover",
    ]
    metric_cols = [col for col in metric_cols if col in all_metrics.columns]
    if not all_metrics.empty and metric_cols:
        heat = all_metrics.set_index("strategy")[metric_cols].copy()
        heat = heat.loc[[s for s in focus if s in heat.index]]
        if not heat.empty:
            display = heat.copy()
            for col in ["annualized_return", "annualized_volatility", "max_drawdown"]:
                if col in display.columns:
                    display[col] = display[col] * 100
            normalized = display.apply(lambda col: (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0.5, axis=0)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            im = ax.imshow(normalized.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
            ax.set_title("Strategy metric profile")
            ax.set_xticks(np.arange(len(display.columns)))
            ax.set_xticklabels(display.columns, rotation=25, ha="right", fontsize=7)
            ax.set_yticks(np.arange(len(display.index)))
            ax.set_yticklabels([short_strategy_name(x) for x in display.index], fontsize=7)
            add_heatmap_labels(ax, display.to_numpy(dtype=float), ".2f", fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.025, label="column-normalized level")
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "fig_strategy_metric_heatmap_stagewise.png", dpi=180)
            plt.close(fig)

    positive_perf = rating_perf.copy()
    if not positive_perf.empty:
        test_positive = positive_perf.loc[positive_perf["split"].eq("test")].copy()
        if test_positive.empty:
            test_positive = positive_perf.groupby("rating", as_index=False).agg(
                positive_rate=("positive_rate", "mean"),
                rating_order=("rating_order", "first"),
            )
        test_positive = test_positive.sort_values("rating_order")
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(test_positive["rating"], test_positive["positive_rate"], color="#72B7B2")
        ax.axhline(0.5, color="#999999", linewidth=1, linestyle="--")
        ax.set_title("Rating positive rate by grade")
        ax.set_xlabel("rating")
        ax.set_ylabel("positive rate")
        ax.set_ylim(0, max(1.0, float(test_positive["positive_rate"].max()) * 1.15))
        ax.tick_params(axis="x", rotation=45)
        for bar, value in zip(bars, test_positive["positive_rate"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0%}", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fig_rating_positive_rate_by_grade.png", dpi=180)
        plt.close(fig)

    rating_heat = rating.pivot_table(index="rating", columns="quarter_idx", values=LABEL_COL, aggfunc="mean")
    rating_heat = rating_heat.reindex(RATING_LABELS)
    if not rating_heat.empty:
        values = rating_heat.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 5.6))
        lim = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-lim, vmax=lim)
        ax.set_title("Rating quarterly return heatmap")
        ax.set_xlabel("quarter_idx")
        ax.set_ylabel("rating")
        ax.set_xticks(np.arange(len(rating_heat.columns)))
        ax.set_xticklabels(rating_heat.columns, fontsize=7)
        ax.set_yticks(np.arange(len(rating_heat.index)))
        ax.set_yticklabels(rating_heat.index, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.025, label="avg future return (%)")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fig_rating_quarterly_return_heatmap.png", dpi=180)
        plt.close(fig)

    scatter_cols = ["pure_stagewise_score", "pure_promethee_score", LABEL_COL]
    if all(col in panel.columns for col in scatter_cols):
        scatter = panel.dropna(subset=scatter_cols).copy()
        if not scatter.empty:
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            sample = scatter.sample(n=min(6000, len(scatter)), random_state=7) if len(scatter) > 6000 else scatter
            color = sample[LABEL_COL].clip(sample[LABEL_COL].quantile(0.02), sample[LABEL_COL].quantile(0.98))
            im = ax.scatter(
                sample["pure_stagewise_score"],
                sample["pure_promethee_score"],
                c=color,
                cmap="RdYlGn",
                s=10,
                alpha=0.45,
                linewidths=0,
            )
            corr = scatter["pure_stagewise_score"].corr(scatter["pure_promethee_score"])
            ax.set_title(f"Stagewise vs PROMETHEE score agreement (corr={corr:.2f})")
            ax.set_xlabel("stagewise rank percentile")
            ax.set_ylabel("PROMETHEE rank percentile")
            ax.plot([0, 1], [0, 1], color="#666666", linewidth=1, linestyle="--")
            fig.colorbar(im, ax=ax, fraction=0.035, label="future return (%)")
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "fig_score_agreement_scatter_stagewise.png", dpi=180)
            plt.close(fig)


def main() -> None:
    ensure_dirs()
    panel = build_hybrid_scores()
    holdings, returns, metrics = build_topk(panel)
    rating, rating_perf = build_rating(panel)

    returns.to_csv(OUTPUT_DIR / "topk_quarterly_returns_stagewise.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "topk_performance_metrics_stagewise.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(OUTPUT_DIR / "topk_holdings_stagewise.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "hybrid_topk_performance_summary.csv", index=False, encoding="utf-8-sig")
    rating[[
        "quarter_idx",
        "split",
        "stock_id",
        "stock_name",
        "rating",
        "rating_rank_in_quarter",
        "rating_score_source",
        "hybrid_score_50_50",
        LABEL_COL,
    ]].to_csv(OUTPUT_DIR / "rating_panel_stagewise.csv", index=False, encoding="utf-8-sig")
    rating_perf.to_csv(OUTPUT_DIR / "rating_performance_stagewise.csv", index=False, encoding="utf-8-sig")
    plot_outputs(panel, returns, metrics, rating, rating_perf)


if __name__ == "__main__":
    main()
