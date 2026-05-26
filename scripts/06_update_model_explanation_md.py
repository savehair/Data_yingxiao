#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Append robust v3 explanation sections to the beginner Markdown document."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "model_explanation_for_beginners.md"
OUT = ROOT / "outputs_robust"
START = "<!-- ROBUST_V3_SECTION_START -->"
END = "<!-- ROBUST_V3_SECTION_END -->"


def read_csv(name: str) -> pd.DataFrame:
    path = OUT / name
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def read_json(name: str) -> dict:
    path = OUT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(v, pct: bool = False) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if pd.isna(x):
        return "NA"
    return f"{x:.2%}" if pct else f"{x:.4f}"


def metric_table() -> str:
    comp = read_csv("robust_vs_original_comparison.csv")
    alpha = read_csv("robust_vs_original_alpha_comparison.csv")
    if comp.empty:
        return "当前项目结果文件中未找到 `robust_vs_original_comparison.csv`，因此本文不对原模型和稳健模型作定量对比。\n"
    test = comp.loc[comp["split"].eq("test")].iloc[0]
    valid = comp.loc[comp["split"].eq("valid")].iloc[0]
    rows = [
        ("test RankIC", test.get("original_rankic"), test.get("robust_rankic"), False, "越高越好"),
        ("test 年化收益", test.get("original_annualized_return"), test.get("robust_annualized_return"), True, "越高越好"),
        ("test Alpha", test.get("original_alpha"), test.get("robust_alpha"), True, "越高越好"),
        ("test Sharpe", test.get("original_sharpe"), test.get("robust_sharpe"), False, "越高越好"),
        ("test Information Ratio", test.get("original_information_ratio"), test.get("robust_information_ratio"), False, "越高越好"),
        ("test 最大回撤", test.get("original_max_drawdown"), test.get("robust_max_drawdown"), True, "越接近 0 越好"),
        ("test 换手率", test.get("original_turnover"), test.get("robust_turnover"), True, "越低越好"),
        ("excess win rate", test.get("original_excess_win_rate"), test.get("robust_excess_win_rate"), True, "越高越好"),
        (
            "valid/test Sharpe gap",
            valid.get("original_sharpe") - test.get("original_sharpe"),
            valid.get("robust_sharpe") - test.get("robust_sharpe"),
            False,
            "越小越稳",
        ),
        (
            "valid/test return gap",
            valid.get("original_annualized_return") - test.get("original_annualized_return"),
            valid.get("robust_annualized_return") - test.get("robust_annualized_return"),
            True,
            "越小越稳",
        ),
    ]
    out = ["| 指标 | 原模型 | 稳健模型 | 是否改善 |", "|---|---:|---:|---|"]
    for name, old, new, as_pct, rule in rows:
        improved = "NA"
        if pd.notna(old) and pd.notna(new):
            if "换手" in name or "gap" in name:
                improved = "是" if abs(float(new)) < abs(float(old)) else "否"
            elif "回撤" in name:
                improved = "是" if float(new) > float(old) else "否"
            else:
                improved = "是" if float(new) > float(old) else "否"
        out.append(f"| {name} | {fmt(old, as_pct)} | {fmt(new, as_pct)} | {improved} |")
    return "\n".join(out) + "\n"


def config_table() -> str:
    cfg = read_json("final_robust_algorithm_config.json")
    if not cfg:
        return "当前项目结果文件中未找到 `final_robust_algorithm_config.json`。\n"
    keys = [
        "experiment_id",
        "rolling_window",
        "max_selected_features",
        "selection_objective",
        "refit_method",
        "ridge_alpha",
        "min_improvement",
        "feature_corr_threshold",
        "promethee_top_n",
        "entropy_weight_method",
        "preference_function",
        "hybrid_score_name",
        "top_k",
        "rank_smoothing",
        "buffer_rule",
        "stable_core_feature_count",
    ]
    lines = ["| 参数 | 稳健模型取值 |", "|---|---|"]
    for key in keys:
        lines.append(f"| `{key}` | `{cfg.get(key, 'NA')}` |")
    return "\n".join(lines) + "\n"


def image_block(title: str, path: str) -> str:
    return f"""### {title}

![{title}](../figures_robust/{path})

**这张图展示什么：**  
这张图用于展示稳健优化相关结果，帮助比较原模型、稳健模型或不同参数方案的差异。

**应该怎么看：**  
重点看横轴和图例中的模型、split 或参数组合，再看纵轴指标是收益、Alpha、Information Ratio、换手率还是回撤。收益和 Alpha 通常越高越好，换手率和回撤通常越低越稳。

**从图中能得到什么结论：**  
如果稳健模型在测试集 Alpha、Information Ratio、回撤或换手率上更好，说明稳健化提高了可交易性或样本外稳定性；如果收益没有提高，也要如实说明收益转化仍有限。

**需要注意什么：**  
这些图不能证明模型未来一定稳定盈利。测试期季度数量有限，所有 test 指标只能作为 holdout 检验，不能反向用于调参。
"""


def build_section() -> str:
    cfg = read_json("final_robust_algorithm_config.json")
    comp = read_csv("robust_vs_original_comparison.csv")
    metrics = read_csv("topk_performance_metrics_robust.csv")
    test_metrics = metrics.loc[metrics["split"].eq("test")].iloc[0] if not metrics.empty and not metrics.loc[metrics["split"].eq("test")].empty else {}
    pref = cfg.get("preference_function", "linear")
    section = f"""
{START}

## 25. 当前模型的问题：为什么需要稳健优化

上一版模型已经完成 231 个宏微观综合因子、移动窗口前向分步回归、熵权-PROMETHEE、Hybrid Score、13 级评级和 Top-K 回测，但结果也暴露出几个问题。第一，测试集 RankIC 为正，说明模型确实有一定横截面排序能力；第二，测试集年化收益、Alpha 和 Sharpe 较弱，说明排序信号转化为真实组合收益的能力有限；第三，验证集明显好于测试集，说明参数对验证区间较敏感；第四，Top20 换手率偏高，如果考虑交易成本，真实收益可能进一步被压缩。

因此，本轮 v3 不是推翻原模型，而是在原主线下做稳健优化：降低模型复杂度，减少换手，关注 Alpha 稳定性和 valid/test 差距，而不是继续追求验证集收益最大化。

## 26. 稳健优化思路

稳健优化主要做了八件事：

1. 将 `max_selected_features` 降到 5、6、8、10，减少噪声因子进入模型；
2. 提高 `min_improvement`，要求新因子必须带来更明显边际改善；
3. 比较 OLS 和 Ridge，降低小样本窗口下的系数波动；
4. 统计稳定因子池，测试期优先使用更稳定的因子；
5. 比较 Top20、Top30、Top50，不再强制 Top20；
6. 加入 buffer rule，降低持仓频繁进出；
7. 加入 rank smoothing，只用过去评分平滑当前评分；
8. 使用 `robust_valid_score`，提高 RankIC、ICIR、Alpha、Information Ratio 的权重，同时惩罚换手率和回撤。

整个过程仍然只用 train 和 valid 做参数选择，test 只用于最终 holdout 检验。

## 27. 稳健优化后的最终参数

最终稳健模型参数如下：

{config_table()}

## 28. 原模型 vs 稳健模型结果对比

原模型和稳健模型的核心指标对比如下：

{metric_table()}

这张表的重点不是单看收益，而是看测试集 RankIC、Alpha、Information Ratio、最大回撤、换手率，以及 valid/test 差距是否改善。如果稳健模型牺牲了部分验证集收益，但降低换手或缩小 valid/test 差距，这仍然是有意义的稳健化结果。

{image_block("图 25：原模型与稳健模型 valid/test 对比", "fig_original_vs_robust_valid_test.png")}

## 29. 排序信号向投资收益的转化分析

RankIC 为正只能说明模型排序方向有一定参考价值，不自动等于组合收益高。排序信号要转化为投资收益，还会受到持仓数量、换手率、噪声股票、极端收益和基准波动影响。

本轮新增 `signal_to_portfolio_conversion.csv`，专门记录 RankIC、年化收益、Alpha、Sharpe、Information Ratio、最大回撤、换手率和超额收益胜率。它的作用是回答：模型排序信号到底有没有转化成更好的组合表现。

{image_block("图 26：RankIC 与组合收益转化", "fig_rankic_vs_portfolio_return.png")}

{image_block("图 27：信号转化能力对比", "fig_signal_conversion_comparison.png")}

{image_block("图 28：换手率与 Alpha 的关系", "fig_turnover_vs_alpha.png")}

## 30. 换手率与交易可实现性

原模型 Top20 换手率偏高，说明每个季度持仓变化较大。即使回测没有计入交易成本，真实交易也会受到佣金、冲击成本和买卖价差影响。

v3 加入 buffer rule：Top30 下，新买入需要进入 Top25，原持仓只要仍在 Top40 内就保留；Top20 下，新买入需要进入 Top15，原持仓仍在 Top30 内就保留。这个规则的目标是减少因为排名小幅波动造成的频繁交易。

{image_block("图 29：原模型与稳健模型换手率对比", "fig_turnover_original_vs_robust.png")}

## 31. Alpha 稳定性分析

本轮新增 Alpha 稳定性指标，包括季度超额收益、超额收益胜率、平均季度 Alpha、Alpha t 值、Information Ratio、tracking error 和 alpha stability score。

当前稳健模型测试集核心指标为：

| 指标 | test 取值 |
|---|---:|
| 年化收益 | {fmt(test_metrics.get("annualized_return", float("nan")), True)} |
| Alpha | {fmt(test_metrics.get("alpha", float("nan")), True)} |
| Sharpe | {fmt(test_metrics.get("sharpe_ratio", float("nan")))} |
| Information Ratio | {fmt(test_metrics.get("information_ratio", float("nan")))} |
| 最大回撤 | {fmt(test_metrics.get("max_drawdown", float("nan")), True)} |
| 换手率 | {fmt(test_metrics.get("average_turnover", float("nan")), True)} |
| 超额收益胜率 | {fmt(test_metrics.get("excess_win_rate", float("nan")), True)} |

{image_block("图 30：Alpha 原模型与稳健模型对比", "fig_alpha_original_vs_robust.png")}

{image_block("图 31：季度超额收益", "fig_excess_return_by_quarter.png")}

{image_block("图 32：Information Ratio 对比", "fig_information_ratio_comparison.png")}

## 32. PROMETHEE 偏好函数选择与数据比较

稳健版本继续采用 PROMETHEE 作为多属性投资前景评价工具。最终偏好函数为 `{pref}`。本项目重点解释 linear preference function：

设 `d = g_j(a) - g_j(b)`，其中 `g_j(a)` 是股票 a 在指标 j 上的标准化得分，`g_j(b)` 是股票 b 在指标 j 上的标准化得分。

Linear Preference Function 为：

```text
P_j(a,b) = 0, 如果 d <= q
P_j(a,b) = (d - q) / (p - q), 如果 q < d < p
P_j(a,b) = 1, 如果 d >= p
```

本项目默认 `q = 0.05`，`p = 0.25`，`sigma = 0.20`。其中 sigma 只用于 Gaussian 对照，linear 不使用 sigma。

linear 偏好函数适合本项目，是因为综合因子经过标准化后是连续变量。两个股票之间差距很小时，不应该直接判定一方完全胜出；差距逐渐变大时，偏好强度也应该逐渐增加。

{image_block("图 33：PROMETHEE 偏好函数比较", "fig_promethee_preference_function_comparison.png")}

{image_block("图 34：PROMETHEE 净流分布", "fig_promethee_net_flow_distribution.png")}

{image_block("图 35：PROMETHEE 偏好函数验证集指标", "fig_promethee_preference_function_valid_metrics.png")}

PROMETHEE-only 如果表现偏弱，不能强行写成 PROMETHEE 单独排序能力很强。更稳妥的定位是：PROMETHEE 提供多属性解释、投资前景评价、评级依据和偏好函数敏感性分析，而不是必然替代前向分步回归排序。

## 33. 稳健优化后的结论

稳健优化后的结论应根据实际指标谨慎表述。如果测试集收益、Alpha、Sharpe 或 Information Ratio 改善，同时换手率下降，可以写“样本外稳定性有所提升”。如果收益没有明显提升，但换手或回撤下降，可以写“模型更适合作为候选池筛选和投资前景评价框架”。如果测试集仍然较弱，则必须如实说明现有因子体系在测试期的收益转化能力有限。

本项目不能宣称模型可以准确预测未来收益率，也不能宣称策略稳定盈利。更合适的最终表述是：v3 稳健优化在保持前向分步回归主线的基础上，系统检查了模型复杂度、Ridge 稳健重拟合、稳定因子池、排名平滑、buffer rule、Alpha 稳定性和 PROMETHEE 偏好函数敏感性，使模型解释和风险检验更完整。

{END}
"""
    return section


def main() -> None:
    DOC.parent.mkdir(exist_ok=True)
    if DOC.exists():
        text = DOC.read_text(encoding="utf-8")
    else:
        text = "# 股票多因子建模与投资前景评价模型说明书\n"
    section = build_section()
    if START in text and END in text:
        before = text.split(START)[0].rstrip()
        after = text.split(END, 1)[1].lstrip()
        text = before + "\n\n" + section + "\n" + after
    else:
        text = text.rstrip() + "\n\n" + section
    DOC.write_text(text, encoding="utf-8")
    print(f"[docs] updated {DOC}")


if __name__ == "__main__":
    main()
