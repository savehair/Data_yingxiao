#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate robust v3 figures from outputs_robust."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_robust"
FIG = ROOT / "figures_robust"
ORIG = ROOT / "outputs_optimized"


def setup():
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    return plt


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")


def pct(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:.1%}"


def fig_original_vs_robust_valid_test(plt):
    comp = read_csv(OUT / "robust_vs_original_comparison.csv")
    if comp.empty:
        return
    metrics = [
        ("annualized_return", "Annualized Return"),
        ("alpha", "Alpha"),
        ("sharpe", "Sharpe"),
        ("turnover", "Turnover"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        plot = comp[["split", f"original_{col}", f"robust_{col}"]].melt("split", var_name="model", value_name="value")
        plot["model"] = plot["model"].str.replace(f"_{col}", "", regex=False)
        pivot = plot.pivot(index="split", columns="model", values="value").reindex(["valid", "test"])
        pivot.plot(kind="bar", ax=ax, color=["#7f8c8d", "#2ca25f"])
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("")
        ax.legend(title="")
    save(fig, "fig_original_vs_robust_valid_test.png")
    plt.close(fig)


def fig_turnover_original_vs_robust(plt):
    comp = read_csv(OUT / "robust_vs_original_comparison.csv")
    if comp.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = comp[["split", "original_turnover", "robust_turnover"]].set_index("split").reindex(["valid", "test"])
    pivot.plot(kind="bar", ax=ax, color=["#9e9e9e", "#1b9e77"])
    ax.set_title("Original vs Robust Turnover")
    ax.set_ylabel("Average Turnover")
    ax.axhline(0, color="black", linewidth=0.8)
    save(fig, "fig_turnover_original_vs_robust.png")
    plt.close(fig)


def fig_topk_cumulative_return_robust(plt):
    ret = read_csv(OUT / "topk_quarterly_returns_robust.csv")
    if ret.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    r = ret.sort_values("quarter_idx").copy()
    r["portfolio_equity"] = (1.0 + r["portfolio_return"] / 100.0).cumprod()
    r["benchmark_equity"] = (1.0 + r["benchmark_return"] / 100.0).cumprod()
    ax.plot(r["quarter_idx"], r["portfolio_equity"], marker="o", label="Robust portfolio", color="#1b9e77")
    ax.plot(r["quarter_idx"], r["benchmark_equity"], marker="o", label="Equal-weight benchmark", color="#7570b3", linestyle="--")
    ax.set_title("Robust Top-K Cumulative Return")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("Equity")
    ax.legend()
    save(fig, "fig_topk_cumulative_return_robust.png")
    plt.close(fig)


def fig_rankic_original_vs_robust(plt):
    robust = read_csv(OUT / "stagewise_quarterly_rankic_robust.csv")
    orig = read_csv(ORIG / "stagewise_quarterly_rankic.csv")
    if robust.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    if not orig.empty:
        ax.plot(orig["quarter_idx"], orig["rankic"], marker="o", label="Original", color="#9e9e9e")
    ax.plot(robust["quarter_idx"], robust["rankic"], marker="o", label="Robust", color="#1b9e77")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Quarterly RankIC: Original vs Robust")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("RankIC")
    ax.legend()
    save(fig, "fig_rankic_original_vs_robust.png")
    plt.close(fig)


def fig_feature_stability_robust(plt):
    stab = read_csv(OUT / "stagewise_feature_stability.csv")
    if stab.empty:
        return
    top = stab.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = top["sign_stability"].clip(0, 1)
    bars = ax.barh(top["feature"], top["selected_count"], color=plt.cm.Greens(colors))
    ax.set_title("Robust Feature Stability")
    ax.set_xlabel("Selected Count")
    for bar, val in zip(bars, top["sign_stability"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f"sign={val:.2f}", va="center", fontsize=8)
    save(fig, "fig_feature_stability_robust.png")
    plt.close(fig)


def fig_drawdown_original_vs_robust(plt):
    ret = read_csv(OUT / "topk_quarterly_returns_robust.csv")
    orig = read_csv(ORIG / "topk_quarterly_returns_stagewise.csv")
    if ret.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame, label, color in [(orig, "Original", "#9e9e9e"), (ret, "Robust", "#1b9e77")]:
        if frame.empty:
            continue
        f = frame.sort_values("quarter_idx")
        eq = (1.0 + f["portfolio_return"] / 100.0).cumprod()
        dd = eq / eq.cummax() - 1.0
        ax.plot(f["quarter_idx"], dd, marker="o", label=label, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Drawdown: Original vs Robust")
    ax.set_ylabel("Drawdown")
    ax.legend()
    save(fig, "fig_drawdown_original_vs_robust.png")
    plt.close(fig)


def fig_alpha_original_vs_robust(plt):
    comp = read_csv(OUT / "robust_vs_original_alpha_comparison.csv")
    if comp.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = comp.set_index("split")[["original_alpha", "robust_alpha"]].reindex(["valid", "test", "all"])
    pivot.plot(kind="bar", ax=ax, color=["#9e9e9e", "#1b9e77"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Alpha: Original vs Robust")
    ax.set_ylabel("Annualized alpha")
    save(fig, "fig_alpha_original_vs_robust.png")
    plt.close(fig)


def fig_excess_return_by_quarter(plt):
    ret = read_csv(OUT / "excess_return_by_quarter.csv")
    if ret.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = np.where(ret["quarterly_excess_return"] >= 0, "#1b9e77", "#d95f02")
    ax.bar(ret["quarter_idx"], ret["quarterly_excess_return"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Robust Quarterly Excess Return")
    ax.set_xlabel("quarter_idx")
    ax.set_ylabel("Portfolio - benchmark return")
    save(fig, "fig_excess_return_by_quarter.png")
    plt.close(fig)


def fig_information_ratio_comparison(plt):
    comp = read_csv(OUT / "robust_vs_original_alpha_comparison.csv")
    if comp.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    comp.set_index("split")["robust_information_ratio"].reindex(["valid", "test", "all"]).plot(kind="bar", ax=ax, color="#1b9e77")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Robust Information Ratio")
    ax.set_ylabel("Mean excess / excess std")
    save(fig, "fig_information_ratio_comparison.png")
    plt.close(fig)


def fig_rankic_vs_portfolio_return(plt):
    conv = read_csv(OUT / "signal_to_portfolio_conversion.csv")
    if conv.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for split, group in conv.groupby("split"):
        ax.scatter(group["rankic"], group["annualized_return"], label=split, alpha=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("RankIC vs Portfolio Return")
    ax.set_xlabel("RankIC")
    ax.set_ylabel("Annualized return")
    ax.legend()
    save(fig, "fig_rankic_vs_portfolio_return.png")
    plt.close(fig)


def fig_valid_test_gap(plt):
    comp = read_csv(OUT / "robust_vs_original_comparison.csv")
    if comp.empty:
        return
    valid = comp.loc[comp["split"].eq("valid")].iloc[0]
    test = comp.loc[comp["split"].eq("test")].iloc[0]
    rows = []
    for m in ["annualized_return", "alpha", "sharpe"]:
        rows.append({"metric": m, "original_gap": valid[f"original_{m}"] - test[f"original_{m}"], "robust_gap": valid[f"robust_{m}"] - test[f"robust_{m}"]})
    gap = pd.DataFrame(rows).set_index("metric")
    fig, ax = plt.subplots(figsize=(8, 5))
    gap.plot(kind="bar", ax=ax, color=["#9e9e9e", "#1b9e77"])
    ax.set_title("Valid-Test Gap: Original vs Robust")
    ax.set_ylabel("valid - test")
    ax.axhline(0, color="black", linewidth=0.8)
    save(fig, "fig_valid_test_gap.png")
    plt.close(fig)


def fig_turnover_vs_alpha(plt):
    log = read_csv(OUT / "robust_experiment_log.csv")
    if log.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(log["valid_turnover"], log["valid_alpha"], c=log["robust_valid_score"], cmap="viridis", alpha=0.65)
    ax.set_title("Validation Turnover vs Alpha")
    ax.set_xlabel("Validation turnover")
    ax.set_ylabel("Validation alpha")
    fig.colorbar(sc, ax=ax, label="robust_valid_score")
    save(fig, "fig_turnover_vs_alpha.png")
    plt.close(fig)


def fig_signal_conversion_comparison(plt):
    conv = read_csv(OUT / "signal_to_portfolio_conversion.csv")
    if conv.empty:
        return
    top = conv.sort_values(["split", "information_ratio"], ascending=[True, False]).groupby("split").head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = top["split"] + "_" + top["score_name"].astype(str) + "_K" + top["top_k"].astype(str)
    ax.bar(np.arange(len(top)), top["information_ratio"], color=np.where(top["split"].eq("test"), "#7570b3", "#1b9e77"))
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Signal-to-Portfolio Conversion")
    ax.set_ylabel("Information ratio")
    save(fig, "fig_signal_conversion_comparison.png")
    plt.close(fig)


def fig_promethee_preference_function_comparison(plt):
    pref = read_csv(OUT / "promethee_preference_function_comparison.csv")
    if pref.empty:
        return
    valid = pref.loc[pref["split"].eq("valid")].copy()
    cols = [c for c in valid.columns if c.startswith("top20_annualized_return") or c == "top20_sharpe"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if "top20_annualized_return" in valid:
        valid.plot(x="preference_function", y="top20_annualized_return", kind="bar", ax=axes[0], legend=False, color="#1b9e77")
        axes[0].set_title("Top20 Annualized Return")
    if "top20_sharpe" in valid:
        valid.plot(x="preference_function", y="top20_sharpe", kind="bar", ax=axes[1], legend=False, color="#7570b3")
        axes[1].set_title("Top20 Sharpe")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    save(fig, "fig_promethee_preference_function_comparison.png")
    plt.close(fig)


def fig_promethee_net_flow_distribution(plt):
    pro = read_csv(OUT / "promethee_scores_robust.csv")
    if pro.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    pro["promethee_net_flow"].hist(ax=ax, bins=40, color="#1b9e77", alpha=0.8)
    ax.set_title("PROMETHEE Net Flow Distribution")
    ax.set_xlabel("net_flow")
    ax.set_ylabel("count")
    save(fig, "fig_promethee_net_flow_distribution.png")
    plt.close(fig)


def fig_promethee_preference_function_valid_metrics(plt):
    pref = read_csv(OUT / "promethee_preference_function_comparison.csv")
    if pref.empty:
        return
    valid = pref.loc[pref["split"].eq("valid")].set_index("preference_function")
    cols = [c for c in ["top10_annualized_return", "top20_annualized_return", "top30_annualized_return", "top10_sharpe", "top20_sharpe", "top30_sharpe"] if c in valid.columns]
    if not cols:
        return
    data = valid[cols].astype(float)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("PROMETHEE Preference Valid Metrics")
    fig.colorbar(im, ax=ax)
    save(fig, "fig_promethee_preference_function_valid_metrics.png")
    plt.close(fig)


def main() -> None:
    plt = setup()
    fig_original_vs_robust_valid_test(plt)
    fig_turnover_original_vs_robust(plt)
    fig_topk_cumulative_return_robust(plt)
    fig_rankic_original_vs_robust(plt)
    fig_feature_stability_robust(plt)
    fig_drawdown_original_vs_robust(plt)
    fig_alpha_original_vs_robust(plt)
    fig_excess_return_by_quarter(plt)
    fig_information_ratio_comparison(plt)
    fig_rankic_vs_portfolio_return(plt)
    fig_valid_test_gap(plt)
    fig_turnover_vs_alpha(plt)
    fig_signal_conversion_comparison(plt)
    fig_promethee_preference_function_comparison(plt)
    fig_promethee_net_flow_distribution(plt)
    fig_promethee_preference_function_valid_metrics(plt)
    print(f"[figures] wrote robust figures to {FIG}")


if __name__ == "__main__":
    main()
