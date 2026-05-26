#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare PROMETHEE preference functions on the optimized stagewise outputs.

The experiment fixes the selected stagewise model, PROMETHEE feature pool,
entropy-weight method, time split, and Top-K evaluation protocol. Only the
PROMETHEE preference function changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_stagewise_optimization as opt  # noqa: E402


OUT = ROOT / "outputs_optimized"
PREFERENCE_FUNCTIONS = ["usual", "u_shape", "v_shape", "level", "linear", "gaussian"]
Q = 0.05
P = 0.25
SIGMA = 0.20


def split_rankic(panel: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for split, group in panel.groupby("split"):
        rows.append(
            {
                "split": split,
                "mean_rankic": opt.rank_ic(group[opt.LABEL], group[score_col]),
            }
        )
    return pd.DataFrame(rows)


def rating_separation_frame(panel: pd.DataFrame, score_col: str) -> pd.DataFrame:
    _rating, perf = opt.rating_outputs(panel, score_col)
    rows = []
    for split, group in perf.groupby("split"):
        hi = group.loc[group["rating"].isin(["AAA", "AA+", "AA"]), "avg_future_return"].mean()
        lo = group.loc[group["rating"].isin(["BB+", "BB", "B"]), "avg_future_return"].mean()
        rows.append({"split": split, "rating_separation": float(hi - lo)})
    return pd.DataFrame(rows)


def topk_metrics(panel: pd.DataFrame, score_col: str) -> pd.DataFrame:
    metrics = []
    for top_k in [10, 20, 30]:
        _hold, _ret, m = opt.evaluate_topk(panel, score_col, top_k)
        metrics.append(m)
    all_metrics = pd.concat(metrics, ignore_index=True)
    rows = []
    for split, group in all_metrics.groupby("split"):
        row = {"split": split}
        for top_k in [10, 20, 30]:
            one = group.loc[group["topk"].eq(top_k)]
            if one.empty:
                continue
            one = one.iloc[0]
            row[f"top{top_k}_annualized_return"] = float(one["annualized_return"])
            row[f"top{top_k}_sharpe"] = float(one["sharpe_ratio"])
            if top_k == 20:
                row["alpha"] = float(one["alpha"])
                row["beta"] = float(one["beta"])
                row["max_drawdown"] = float(one["max_drawdown"])
                row["turnover"] = float(one["average_turnover"])
        rows.append(row)
    return pd.DataFrame(rows)


def top20_overlap(panel: pd.DataFrame, score_col: str, baseline: pd.DataFrame, baseline_score_col: str) -> pd.DataFrame:
    rows = []
    base_sets = {
        int(q): set(g.sort_values(baseline_score_col, ascending=False).head(20)["stock_id"].astype(str))
        for q, g in baseline.groupby("quarter_idx")
    }
    for q, group in panel.groupby("quarter_idx"):
        chosen = set(group.sort_values(score_col, ascending=False).head(20)["stock_id"].astype(str))
        base = base_sets.get(int(q), set())
        overlap = len(chosen & base) / 20.0 if base else np.nan
        rows.append({"quarter_idx": int(q), "split": opt.split_name(int(q)), "top20_overlap": overlap})
    return pd.DataFrame(rows).groupby("split", as_index=False)["top20_overlap"].mean().rename(columns={"top20_overlap": "avg_top20_overlap_with_selected"})


def main() -> None:
    config_path = OUT / "final_selected_algorithm_config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    panel, _composite, _features = opt.load_model_panel()
    pred = pd.read_csv(OUT / "stagewise_prediction_panel.csv", dtype={"stock_id": str}, encoding="utf-8-sig")
    selected = pd.read_csv(OUT / "stagewise_selected_features_by_quarter.csv", encoding="utf-8-sig")

    prom_base = {
        "promethee_top_n": int(config.get("promethee_top_n", 20)),
        "entropy_weight_method": str(config.get("entropy_weight_method", "smooth_80_20")),
        "min_selected_count": int(config.get("min_selected_count", 2)),
        "sign_stability_threshold": float(config.get("sign_stability_threshold", 0.55)),
    }
    final_pref = str(config.get("preference_function", "linear"))
    score_col = "pure_promethee_score"

    panels: dict[str, pd.DataFrame] = {}
    net_std_rows = []
    for pref in PREFERENCE_FUNCTIONS:
        prom = opt.PromConfig(preference_function=pref, **prom_base)
        pro, _weights = opt.build_promethee_scores(panel, pred, selected, prom)
        hp = opt.build_hybrid_panel(pred, pro)
        panels[pref] = hp
        net_std = pro.groupby("quarter_idx")["promethee_net_flow"].std(ddof=0).mean()
        net_std_rows.append({"preference_function": pref, "avg_net_flow_std": float(net_std)})

    baseline = panels[final_pref]
    rows = []
    for pref, hp in panels.items():
        rankic = split_rankic(hp, score_col)
        perf = topk_metrics(hp, score_col)
        rating = rating_separation_frame(hp, score_col)
        overlap = top20_overlap(hp, score_col, baseline, score_col)
        merged = rankic.merge(perf, on="split", how="outer").merge(rating, on="split", how="outer").merge(overlap, on="split", how="outer")
        merged["preference_function"] = pref
        merged["q"] = Q if pref in ["u_shape", "level", "linear"] else (0.0 if pref == "v_shape" else np.nan)
        merged["p"] = P if pref in ["v_shape", "level", "linear"] else np.nan
        merged["sigma"] = SIGMA if pref == "gaussian" else np.nan
        merged["avg_net_flow_std"] = dict((r["preference_function"], r["avg_net_flow_std"]) for r in net_std_rows)[pref]
        merged["selected_as_final"] = pref == final_pref
        if pref == final_pref:
            reason = "Final PROMETHEE preference function from validation-selected optimized configuration."
        else:
            reason = "Sensitivity comparison only; test metrics were not used to select the final preference function."
        merged["reason"] = reason
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    ordered_cols = [
        "preference_function",
        "q",
        "p",
        "sigma",
        "split",
        "mean_rankic",
        "top10_annualized_return",
        "top20_annualized_return",
        "top30_annualized_return",
        "top10_sharpe",
        "top20_sharpe",
        "top30_sharpe",
        "alpha",
        "beta",
        "max_drawdown",
        "turnover",
        "rating_separation",
        "avg_net_flow_std",
        "avg_top20_overlap_with_selected",
        "selected_as_final",
        "reason",
    ]
    out = out.loc[:, ordered_cols]
    out.to_csv(OUT / "promethee_preference_function_comparison.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
