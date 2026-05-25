#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate additional diagnostic figures for the optimized stagewise project."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_optimized"
FIG = ROOT / "figures_optimized"
LABEL = "future_return_1q"
RATING_HIGH = ["AAA", "AA+", "AA"]
RATING_MID = ["A+", "A", "A-", "BBB+", "BBB", "BBB-"]
RATING_LOW = ["BB+", "BB", "B"]


def configure_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def pct(x: float) -> float:
    return float(x) * 100.0 if pd.notna(x) else np.nan


def load_config() -> dict:
    p = OUT / "final_selected_algorithm_config.json"
    return json.loads(p.read_text(encoding="utf-8"))


def fig_optimization_valid_score_top10_fixed() -> Path:
    plt = configure_matplotlib()
    log_path = OUT / "optimization_experiment_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path, encoding="utf-8-sig")
    else:
        log = pd.read_csv(OUT / "final_algorithm_comparison_summary.csv", encoding="utf-8-sig")
    top = log.sort_values("valid_score", ascending=False).head(10).copy().reset_index(drop=True)
    top["plot_label"] = [
        f"#{i + 1}\n{row.hybrid_score_name}\nTop{int(row.top_k)} | P{int(row.promethee_top_n) if pd.notna(row.promethee_top_n) else '-'} | {row.entropy_weight_method}"
        for i, row in top.iterrows()
    ]
    colors = np.where(top.get("selected_as_final", False).astype(bool), "#59A14F", "#4C78A8")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    bars = ax.bar(np.arange(len(top)), top["valid_score"], color=colors)
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(top["plot_label"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("valid_score")
    ax.set_title("Top 10 validation composite scores")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, top["valid_score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out = FIG / "fig_optimization_valid_score_top10.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_rankic_split_summary() -> Path:
    plt = configure_matplotlib()
    rankic = pd.read_csv(OUT / "stagewise_quarterly_rankic.csv", encoding="utf-8-sig")
    split_order = ["train", "valid", "test"]
    colors = {"train": "#4C78A8", "valid": "#59A14F", "test": "#E15759"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    for split in split_order:
        g = rankic.loc[rankic["split"].eq(split)].sort_values("quarter_idx")
        if not g.empty:
            ax.plot(g["quarter_idx"], g["rankic"], marker="o", label=split, color=colors[split])
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_title("Quarterly RankIC by split")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("RankIC")
    ax.legend()

    mean_rankic = rankic.groupby("split")["rankic"].mean().reindex(split_order)
    ax = axes[1]
    bars = ax.bar(mean_rankic.index, mean_rankic.values, color=[colors[s] for s in mean_rankic.index])
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_title("Mean RankIC")
    ax.set_ylabel("RankIC")
    for bar, value in zip(bars, mean_rankic.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout()
    out = FIG / "fig_rankic_split_summary_optimized.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_top20_return_vs_benchmark() -> Path:
    plt = configure_matplotlib()
    returns = pd.read_csv(OUT / "topk_quarterly_returns_stagewise.csv", encoding="utf-8-sig").sort_values("quarter_idx")
    x = np.arange(len(returns))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x - width / 2, returns["portfolio_return"], width, label="Top20 portfolio", color="#4C78A8")
    ax.bar(x + width / 2, returns["benchmark_return"], width, label="Equal-weight benchmark", color="#F28E2B")
    ax.plot(x, returns["excess_return"], color="#59A14F", marker="o", linewidth=2, label="Excess return")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(returns["quarter_idx"].astype(str), fontsize=8)
    ax.set_title("Top20 quarterly return vs benchmark")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("return (%)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out = FIG / "fig_top20_quarterly_return_vs_benchmark_optimized.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_top20_equity_drawdown() -> Path:
    plt = configure_matplotlib()
    returns = pd.read_csv(OUT / "topk_quarterly_returns_stagewise.csv", encoding="utf-8-sig").sort_values("quarter_idx")
    port = (1.0 + returns["portfolio_return"] / 100.0).cumprod()
    bench = (1.0 + returns["benchmark_return"] / 100.0).cumprod()
    drawdown = port / port.cummax() - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(returns["quarter_idx"], port, marker="o", label="Top20 portfolio", color="#4C78A8")
    axes[0].plot(returns["quarter_idx"], bench, marker="o", label="Benchmark", color="#F28E2B")
    axes[0].set_title("Top20 cumulative equity and drawdown")
    axes[0].set_ylabel("equity index")
    axes[0].legend()
    axes[1].fill_between(returns["quarter_idx"].astype(float).to_numpy(), drawdown.to_numpy() * 100, 0, color="#E15759", alpha=0.45)
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].set_xlabel("quarter_idx")
    axes[1].set_ylabel("drawdown (%)")
    fig.tight_layout()
    out = FIG / "fig_top20_equity_drawdown_optimized.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_score_decile_return() -> Path:
    plt = configure_matplotlib()
    config = load_config()
    score_col = config["hybrid_score_name"]
    panel = pd.read_csv(OUT / "hybrid_scores_stagewise.csv", dtype={"stock_id": str}, encoding="utf-8-sig")
    rows = []
    for (split, q), group in panel.loc[panel["split"].isin(["valid", "test"])].groupby(["split", "quarter_idx"]):
        ranked = group.dropna(subset=[score_col, LABEL]).sort_values(score_col, ascending=False).copy()
        if ranked.empty:
            continue
        ranked["decile"] = (np.floor(np.arange(len(ranked)) * 10 / len(ranked)).astype(int) + 1).clip(1, 10)
        dec = ranked.groupby("decile")[LABEL].mean().reset_index()
        dec["split"] = split
        dec["quarter_idx"] = q
        rows.append(dec)
    decile = pd.concat(rows, ignore_index=True)
    avg = decile.groupby(["split", "decile"], as_index=False)[LABEL].mean()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    valid = avg.loc[avg["split"].eq("valid")].sort_values("decile")
    test = avg.loc[avg["split"].eq("test")].sort_values("decile")
    x = np.arange(1, 11)
    width = 0.38
    ax.bar(x - width / 2, valid[LABEL], width, label="valid", color="#59A14F")
    ax.bar(x + width / 2, test[LABEL], width, label="test", color="#E15759")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_title(f"Average future return by score decile ({score_col})")
    ax.set_xlabel("score decile (1=highest score, 10=lowest score)")
    ax.set_ylabel("avg future return (%)")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    out = FIG / "fig_score_decile_return_valid_test_optimized.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_rating_group_separation() -> Path:
    plt = configure_matplotlib()
    rating = pd.read_csv(OUT / "rating_performance_stagewise.csv", encoding="utf-8-sig")
    rows = []
    for split, g in rating.groupby("split"):
        rows.extend(
            [
                {"split": split, "group": "High: AAA/AA+/AA", "avg_return": g.loc[g["rating"].isin(RATING_HIGH), "avg_future_return"].mean()},
                {"split": split, "group": "Middle: A to BBB-", "avg_return": g.loc[g["rating"].isin(RATING_MID), "avg_future_return"].mean()},
                {"split": split, "group": "Low: BB+/BB/B", "avg_return": g.loc[g["rating"].isin(RATING_LOW), "avg_future_return"].mean()},
            ]
        )
    grouped = pd.DataFrame(rows)
    split_order = ["train", "valid", "test"]
    group_order = ["High: AAA/AA+/AA", "Middle: A to BBB-", "Low: BB+/BB/B"]
    colors = ["#4C78A8", "#F28E2B", "#E15759"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(split_order))
    width = 0.24
    for i, group in enumerate(group_order):
        vals = [grouped.loc[grouped["split"].eq(split) & grouped["group"].eq(group), "avg_return"].iloc[0] for split in split_order]
        ax.bar(x + (i - 1) * width, vals, width, label=group, color=colors[i])
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(split_order)
    ax.set_ylabel("avg future return (%)")
    ax.set_title("Rating group return separation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig_rating_group_separation_optimized.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_feature_macro_heatmap() -> Path:
    plt = configure_matplotlib()
    freq = pd.read_csv(OUT / "stagewise_feature_frequency.csv", encoding="utf-8-sig")
    top_stocks = freq.groupby("stock_factor")["selected_count"].sum().sort_values(ascending=False).head(12).index
    heat = freq.loc[freq["stock_factor"].isin(top_stocks)].pivot_table(
        index="macro_factor",
        columns="stock_factor",
        values="selected_count",
        aggfunc="sum",
        fill_value=0,
    )
    heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
    heat = heat.loc[:, heat.sum(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
    ax.set_title("Selected feature frequency by stock factor and macro factor")
    ax.set_xlabel("stock factor")
    ax.set_ylabel("macro factor")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(c)[:16] for c in heat.columns], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iat[i, j]
            if value:
                ax.text(j, i, str(int(value)), ha="center", va="center", fontsize=7, color="#222222")
    fig.colorbar(im, ax=ax, fraction=0.025, label="selected count")
    fig.tight_layout()
    out = FIG / "fig_stagewise_macro_factor_selection_heatmap_optimized.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_preference_function_valid_metrics() -> Path:
    plt = configure_matplotlib()
    pref = pd.read_csv(OUT / "promethee_preference_function_comparison.csv", encoding="utf-8-sig")
    valid = pref.loc[pref["split"].eq("valid")].copy()
    order = ["usual", "u_shape", "v_shape", "level", "linear", "gaussian"]
    valid["preference_function"] = pd.Categorical(valid["preference_function"], categories=order, ordered=True)
    valid = valid.sort_values("preference_function")

    metrics = [
        ("mean_rankic", "RankIC"),
        ("top20_annualized_return", "Top20 annual return"),
        ("top20_sharpe", "Top20 Sharpe"),
        ("rating_separation", "Rating separation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        vals = valid[col].astype(float)
        bars = ax.bar(valid["preference_function"].astype(str), vals, color=np.where(valid["selected_as_final"], "#59A14F", "#4C78A8"))
        ax.axhline(0, color="#777777", linewidth=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    fig.suptitle("PROMETHEE preference function sensitivity on validation split", y=1.02)
    fig.tight_layout()
    out = FIG / "fig_promethee_preference_function_valid_metrics_optimized.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_promethee_preference_function_comparison() -> Path:
    plt = configure_matplotlib()
    pref = pd.read_csv(OUT / "promethee_preference_function_comparison.csv", encoding="utf-8-sig")
    valid = pref.loc[pref["split"].eq("valid")].copy()
    order = ["usual", "u_shape", "v_shape", "level", "linear", "gaussian"]
    valid["preference_function"] = pd.Categorical(valid["preference_function"], categories=order, ordered=True)
    valid = valid.sort_values("preference_function")

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(valid))
    bars = ax1.bar(
        x,
        valid["top20_annualized_return"] * 100,
        color=np.where(valid["selected_as_final"], "#59A14F", "#4C78A8"),
        alpha=0.85,
        label="Top20 annualized return",
    )
    ax1.axhline(0, color="#777777", linewidth=1)
    ax1.set_ylabel("Top20 annualized return (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid["preference_function"].astype(str), rotation=25)
    ax2 = ax1.twinx()
    ax2.plot(x, valid["mean_rankic"], color="#E15759", marker="o", linewidth=2, label="RankIC")
    ax2.set_ylabel("RankIC")
    ax1.set_title("PROMETHEE preference function comparison on validation split")
    for bar, value in zip(bars, valid["top20_annualized_return"] * 100):
        ax1.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    out = FIG / "fig_promethee_preference_function_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_promethee_net_flow_distribution() -> Path:
    plt = configure_matplotlib()
    pro = pd.read_csv(OUT / "promethee_scores_stagewise.csv", encoding="utf-8-sig")
    pred = pd.read_csv(OUT / "stagewise_prediction_panel.csv", encoding="utf-8-sig", usecols=["quarter_idx", "split"])
    split_map = pred.drop_duplicates("quarter_idx").set_index("quarter_idx")["split"].to_dict()
    pro["split"] = pro["quarter_idx"].map(split_map)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = 40
    for split, color in [("train", "#4C78A8"), ("valid", "#59A14F"), ("test", "#E15759")]:
        values = pro.loc[pro["split"].eq(split), "promethee_net_flow"].dropna()
        if not values.empty:
            ax.hist(values, bins=bins, alpha=0.45, label=split, color=color, density=True)
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_title("PROMETHEE net flow distribution")
    ax.set_xlabel("net_flow")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    out = FIG / "fig_promethee_net_flow_distribution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_topk_quarterly_return_stagewise() -> Path:
    plt = configure_matplotlib()
    returns = pd.read_csv(OUT / "topk_quarterly_returns_stagewise.csv", encoding="utf-8-sig").sort_values("quarter_idx")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(returns["quarter_idx"], returns["portfolio_return"], marker="o", linewidth=2, label="Top20 portfolio", color="#4C78A8")
    ax.plot(returns["quarter_idx"], returns["benchmark_return"], marker="o", linewidth=2, label="Benchmark", color="#F28E2B")
    ax.bar(returns["quarter_idx"], returns["excess_return"], alpha=0.25, label="Excess return", color="#59A14F")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_title("Top-K quarterly return stagewise")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("return (%)")
    ax.legend()
    fig.tight_layout()
    out = FIG / "fig_topk_quarterly_return_stagewise.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_turnover_and_holding_count() -> Path:
    plt = configure_matplotlib()
    returns = pd.read_csv(OUT / "topk_quarterly_returns_stagewise.csv", encoding="utf-8-sig").sort_values("quarter_idx")
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(returns["quarter_idx"], returns["turnover"], marker="o", color="#4C78A8", label="turnover")
    ax1.set_ylabel("turnover")
    ax1.set_xlabel("quarter_idx")
    ax1.set_ylim(0, max(1.05, float(returns["turnover"].max()) * 1.15))
    ax2 = ax1.twinx()
    ax2.bar(returns["quarter_idx"], returns["holding_count"], alpha=0.25, color="#F28E2B", label="holding count")
    ax2.set_ylabel("holding count")
    ax1.set_title("Top20 turnover and holding count by quarter")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    out = FIG / "fig_top20_turnover_holding_count_optimized.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    FIG.mkdir(exist_ok=True)
    generated = [
        fig_optimization_valid_score_top10_fixed(),
        fig_rankic_split_summary(),
        fig_top20_return_vs_benchmark(),
        fig_top20_equity_drawdown(),
        fig_score_decile_return(),
        fig_rating_group_separation(),
        fig_feature_macro_heatmap(),
        fig_preference_function_valid_metrics(),
        fig_promethee_preference_function_comparison(),
        fig_promethee_net_flow_distribution(),
        fig_topk_quarterly_return_stagewise(),
        fig_turnover_and_holding_count(),
    ]
    manifest = pd.DataFrame({"figure": [str(p.relative_to(ROOT)).replace("\\", "/") for p in generated]})
    manifest.to_csv(OUT / "additional_visualization_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
