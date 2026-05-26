#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Robust v3 stagewise + PROMETHEE optimization pipeline.

This entrypoint keeps the fixed course route and adds robustness controls:
lower stagewise complexity, Ridge refit comparison, stable feature pool,
rank smoothing, Top-K buffer rules, alpha stability diagnostics, and
original-vs-robust comparison. Test data is never used for parameter choice.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import run_stagewise_optimization as opt


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"E:\JetBrains\Anaconda3\envs\pytorch\python.exe")
OUT = ROOT / "outputs_robust"
FIG = ROOT / "figures_robust"
ORIG = ROOT / "outputs_optimized"
LABEL = opt.LABEL
HYBRIDS = list(opt.HYBRIDS.keys())
TOPKS = [20, 30, 50]
SMOOTHING_METHODS = ["none", "prev_70_30", "trailing2_60_40"]


def ensure_dirs() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)


def configure_opt_globals() -> None:
    opt.OUT = OUT
    opt.FIG = FIG


def verify_python() -> None:
    actual = subprocess.check_output([str(PYTHON), "-c", "import sys; print(sys.executable)"], text=True).strip()
    if Path(actual).resolve() != PYTHON.resolve():
        raise RuntimeError(f"Unexpected Python interpreter: {actual}")
    print(f"[env] python={actual}")


def split_series_value(frame: pd.DataFrame, split: str, col: str) -> float:
    row = frame.loc[frame["split"].eq(split)]
    if row.empty or col not in row:
        return np.nan
    return float(row[col].iloc[0])


def standardize(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (x - x.mean()) / std


def robust_stagewise_configs() -> list[opt.StagewiseConfig]:
    configs: list[opt.StagewiseConfig] = []
    refits = [("OLS", 1.0), ("Ridge", 1.0), ("Ridge", 3.0), ("Ridge", 10.0), ("Ridge", 30.0)]
    idx = 1
    for max_features in [5, 6, 8, 10]:
        for min_improvement in [0.001, 0.002, 0.005]:
            for refit_method, ridge_alpha in refits:
                configs.append(
                    opt.StagewiseConfig(
                        experiment_id=f"rb_{idx:03d}",
                        preprocessing_method="window_median",
                        winsorize_quantile=0.05,
                        scaling_method="quarter_rank",
                        rolling_window=12,
                        max_selected_features=max_features,
                        selection_objective="residual_correlation",
                        refit_method=refit_method,
                        ridge_alpha=ridge_alpha,
                        feature_corr_threshold=0.90,
                        min_improvement=min_improvement,
                    )
                )
                idx += 1
    return configs


def rankic_icir(rankic: pd.DataFrame, split: str) -> float:
    vals = pd.to_numeric(rankic.loc[rankic["split"].eq(split), "rankic"], errors="coerce").dropna()
    if len(vals) < 2:
        return np.nan
    std = vals.std(ddof=1)
    return float(vals.mean() / std) if std > 1e-12 else np.nan


def feature_stability(selected: pd.DataFrame, composite: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    meta = composite.rename(columns={"composite_feature": "feature"})
    rows = (
        selected.groupby("selected_feature")
        .agg(
            selected_count=("quarter_idx", "count"),
            avg_coefficient=("coefficient", "mean"),
            avg_abs_coefficient=("coefficient", lambda x: float(np.mean(np.abs(pd.to_numeric(x, errors="coerce"))))),
            positive_coef_ratio=("coefficient", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            first_selected_avg_step=("step", "mean"),
        )
        .reset_index()
        .rename(columns={"selected_feature": "feature"})
    )
    rows["negative_coef_ratio"] = 1.0 - rows["positive_coef_ratio"]
    rows["sign_stability"] = rows[["positive_coef_ratio", "negative_coef_ratio"]].max(axis=1)
    rows = rows.merge(meta[["feature", "stock_factor", "macro_factor", "is_composite"]], on="feature", how="left")
    return rows.sort_values(["selected_count", "sign_stability", "avg_abs_coefficient"], ascending=[False, False, False])


def stable_core_features(stability: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if stability.empty:
        return stability.copy(), "empty"
    strict = stability.loc[(stability["selected_count"] >= 3) & (stability["sign_stability"] >= 0.60)].copy()
    if len(strict) >= 5:
        strict["stable_rule"] = "selected_count>=3 and sign_stability>=0.60"
        return strict, "strict"
    relaxed = stability.loc[(stability["selected_count"] >= 2) & (stability["sign_stability"] >= 0.55)].copy()
    relaxed["stable_rule"] = "fallback: selected_count>=2 and sign_stability>=0.55"
    return relaxed, "relaxed"


def add_rank_smoothing(panel: pd.DataFrame, score_col: str, method: str, out_col: str) -> pd.DataFrame:
    p = panel.copy()
    p = p.sort_values(["stock_id", "quarter_idx"])
    current = pd.to_numeric(p[score_col], errors="coerce")
    if method == "none":
        p[out_col] = current
    elif method == "prev_70_30":
        prev = p.groupby("stock_id")[score_col].shift(1)
        p[out_col] = 0.7 * current + 0.3 * pd.to_numeric(prev, errors="coerce").fillna(current)
    elif method == "trailing2_60_40":
        hist = (
            p.groupby("stock_id")[score_col]
            .shift(1)
            .groupby(p["stock_id"])
            .rolling(2, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        p[out_col] = 0.6 * current + 0.4 * pd.to_numeric(hist, errors="coerce").fillna(current)
    else:
        raise ValueError(method)
    p[out_col] = p.groupby("quarter_idx")[out_col].rank(method="average", pct=True)
    return p.sort_values(["quarter_idx", "stock_id"]).reset_index(drop=True)


def buffer_thresholds(top_k: int) -> tuple[int, int] | None:
    if top_k == 20:
        return 15, 30
    if top_k == 30:
        return 25, 40
    return None


def evaluate_topk_rule(panel: pd.DataFrame, score_col: str, top_k: int, buffer_rule: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings, returns = [], []
    bench = panel.groupby("quarter_idx")[LABEL].mean().rename("benchmark_return")
    prev_weights: dict[str, float] = {}
    prev_holdings: set[str] = set()
    thresholds = buffer_thresholds(top_k) if buffer_rule == "buffer" else None
    for q, group in panel.groupby("quarter_idx", sort=True):
        ranked_all = group.dropna(subset=[score_col, LABEL]).sort_values([score_col, "stock_id"], ascending=[False, True]).copy()
        ranked_all["portfolio_rank"] = np.arange(1, len(ranked_all) + 1)
        if ranked_all.empty:
            continue
        if thresholds is None or not prev_holdings:
            chosen = ranked_all.head(top_k).copy()
        else:
            entry_rank, hold_rank = thresholds
            keep = ranked_all.loc[ranked_all["stock_id"].isin(prev_holdings) & (ranked_all["portfolio_rank"] <= hold_rank)].copy()
            if len(keep) > top_k:
                chosen = keep.sort_values("portfolio_rank").head(top_k).copy()
            else:
                entry_pool = ranked_all.loc[(ranked_all["portfolio_rank"] <= entry_rank) & (~ranked_all["stock_id"].isin(keep["stock_id"]))].copy()
                chosen = pd.concat([keep, entry_pool.sort_values("portfolio_rank").head(max(0, top_k - len(keep)))], ignore_index=True)
                if len(chosen) < top_k:
                    filler = ranked_all.loc[~ranked_all["stock_id"].isin(chosen["stock_id"])].sort_values("portfolio_rank").head(top_k - len(chosen))
                    chosen = pd.concat([chosen, filler], ignore_index=True)
                chosen = chosen.sort_values("portfolio_rank").head(top_k).copy()
        if chosen.empty:
            continue
        chosen["portfolio_rank"] = np.arange(1, len(chosen) + 1)
        chosen["weight"] = 1.0 / len(chosen)
        current = dict(zip(chosen["stock_id"], chosen["weight"]))
        names = set(prev_weights) | set(current)
        turnover = 0.5 * sum(abs(current.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in names)
        prev_weights = current
        prev_holdings = set(current)
        port_ret = float((chosen["weight"] * chosen[LABEL]).sum())
        b = float(bench.loc[q])
        strategy = f"{score_col}_top{top_k}_{buffer_rule}"
        returns.append(
            {
                "quarter_idx": q,
                "split": opt.split_name(int(q)),
                "strategy": strategy,
                "score_col": score_col,
                "topk": top_k,
                "buffer_rule": buffer_rule,
                "portfolio_return": port_ret,
                "benchmark_return": b,
                "quarterly_excess_return": port_ret - b,
                "excess_return": port_ret - b,
                "turnover": turnover,
                "holding_count": int(len(chosen)),
            }
        )
        holdings.append(
            chosen[["quarter_idx", "split", "stock_id", "stock_name", LABEL, score_col, "portfolio_rank", "weight"]]
            .assign(strategy=strategy, score_col=score_col, topk=top_k, buffer_rule=buffer_rule)
            .rename(columns={score_col: "score"})
        )
    returns_df = pd.DataFrame(returns)
    holdings_df = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    metrics_df = topk_metrics_ext(returns_df)
    return holdings_df, returns_df, metrics_df


def max_drawdown_pct(r_pct: pd.Series) -> float:
    if r_pct.empty:
        return np.nan
    equity = (1.0 + r_pct.fillna(0.0) / 100.0).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def topk_metrics_ext(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if returns.empty:
        return pd.DataFrame()
    frames = [("all", returns), *list(returns.groupby("split"))]
    for split, frame in frames:
        if frame.empty:
            continue
        r = pd.to_numeric(frame["portfolio_return"], errors="coerce").dropna()
        sr = r / 100.0
        b = pd.to_numeric(frame.loc[r.index, "benchmark_return"], errors="coerce") / 100.0
        ex = pd.to_numeric(frame.loc[r.index, "quarterly_excess_return"], errors="coerce") / 100.0
        n = len(sr)
        cumulative = float((1.0 + sr).prod() - 1.0) if n else np.nan
        annual = float((1.0 + cumulative) ** (4.0 / n) - 1.0) if n else np.nan
        vol = float(sr.std(ddof=1) * math.sqrt(4)) if n > 1 else np.nan
        sharpe = annual / vol if np.isfinite(vol) and vol > 0 else np.nan
        beta, alpha = np.nan, np.nan
        mask = sr.notna() & b.notna()
        if int(mask.sum()) >= 2 and float(b.loc[mask].var()) > 1e-12:
            beta = float(np.cov(sr.loc[mask], b.loc[mask], ddof=1)[0, 1] / b.loc[mask].var())
            alpha = float((sr.loc[mask].mean() - beta * b.loc[mask].mean()) * 4)
        ex_std = float(ex.std(ddof=1)) if len(ex) > 1 else np.nan
        ex_mean = float(ex.mean()) if len(ex) else np.nan
        information_ratio = ex_mean / ex_std if np.isfinite(ex_std) and ex_std > 1e-12 else np.nan
        alpha_t_stat = ex_mean / (ex_std / math.sqrt(len(ex))) if np.isfinite(ex_std) and ex_std > 1e-12 and len(ex) > 1 else np.nan
        excess_win_rate = float((ex > 0).mean()) if len(ex) else np.nan
        alpha_stability_score = information_ratio * excess_win_rate if np.isfinite(information_ratio) and np.isfinite(excess_win_rate) else np.nan
        rows.append(
            {
                "split": split,
                "strategy": frame["strategy"].iloc[0],
                "score_col": frame["score_col"].iloc[0],
                "topk": int(frame["topk"].iloc[0]),
                "buffer_rule": frame["buffer_rule"].iloc[0] if "buffer_rule" in frame else "none",
                "average_period_return": float(r.mean()) if n else np.nan,
                "cumulative_return": cumulative,
                "annualized_return": annual,
                "annualized_volatility": vol,
                "sharpe_ratio": sharpe,
                "alpha": alpha,
                "beta": beta,
                "max_drawdown": max_drawdown_pct(r),
                "win_rate": float((r > 0).mean()) if n else np.nan,
                "average_turnover": float(frame["turnover"].mean()),
                "average_holding_count": float(frame["holding_count"].mean()),
                "mean_quarterly_alpha": ex_mean,
                "alpha_t_stat": alpha_t_stat,
                "information_ratio": information_ratio,
                "tracking_error": ex_std,
                "excess_win_rate": excess_win_rate,
                "alpha_stability_score": alpha_stability_score,
            }
        )
    return pd.DataFrame(rows)


def metric(metrics: pd.DataFrame, split: str, col: str) -> float:
    return split_series_value(metrics, split, col)


def append_robust_log(
    rows: list[dict],
    sw: opt.StagewiseConfig,
    prom: opt.PromConfig | None,
    hybrid: str,
    top_k: int,
    smoothing: str,
    buffer_rule: str,
    rankic: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    base = asdict(sw)
    valid_rankic_vals = pd.to_numeric(rankic.loc[rankic["split"].eq("valid"), "rankic"], errors="coerce").dropna()
    test_rankic_vals = pd.to_numeric(rankic.loc[rankic["split"].eq("test"), "rankic"], errors="coerce").dropna()
    base.update(
        {
            "promethee_top_n": prom.promethee_top_n if prom else np.nan,
            "entropy_weight_method": prom.entropy_weight_method if prom else "none",
            "preference_function": prom.preference_function if prom else "none",
            "min_selected_count": prom.min_selected_count if prom else np.nan,
            "sign_stability_threshold": prom.sign_stability_threshold if prom else np.nan,
            "hybrid_score_name": hybrid,
            "top_k": top_k,
            "rank_smoothing": smoothing,
            "buffer_rule": buffer_rule,
            "train_rankic": float(rankic.loc[rankic["split"].eq("train"), "rankic"].mean()),
            "valid_rankic": float(valid_rankic_vals.mean()) if len(valid_rankic_vals) else np.nan,
            "test_rankic": float(test_rankic_vals.mean()) if len(test_rankic_vals) else np.nan,
            "valid_icir": rankic_icir(rankic, "valid"),
            "test_icir": rankic_icir(rankic, "test"),
            "valid_annualized_return": metric(metrics, "valid", "annualized_return"),
            "valid_sharpe": metric(metrics, "valid", "sharpe_ratio"),
            "valid_alpha": metric(metrics, "valid", "alpha"),
            "valid_information_ratio": metric(metrics, "valid", "information_ratio"),
            "valid_max_drawdown": metric(metrics, "valid", "max_drawdown"),
            "valid_turnover": metric(metrics, "valid", "average_turnover"),
            "valid_excess_win_rate": metric(metrics, "valid", "excess_win_rate"),
            "test_annualized_return": metric(metrics, "test", "annualized_return"),
            "test_sharpe": metric(metrics, "test", "sharpe_ratio"),
            "test_alpha": metric(metrics, "test", "alpha"),
            "test_information_ratio": metric(metrics, "test", "information_ratio"),
            "test_max_drawdown": metric(metrics, "test", "max_drawdown"),
            "test_turnover": metric(metrics, "test", "average_turnover"),
            "test_excess_win_rate": metric(metrics, "test", "excess_win_rate"),
            "selected_as_final": False,
            "reason": "",
        }
    )
    rows.append(base)


def robust_valid_score(log: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=log.index)
    total = 0.0
    weights = {
        "valid_rankic": 0.25,
        "valid_icir": 0.20,
        "valid_alpha": 0.20,
        "valid_information_ratio": 0.15,
    }
    for col, weight in weights.items():
        if col in log and log[col].notna().any():
            score += weight * standardize(log[col]).fillna(0.0)
            total += weight
    if "valid_turnover" in log and log["valid_turnover"].notna().any():
        score -= 0.15 * standardize(log["valid_turnover"]).fillna(0.0)
        total += 0.15
    if "valid_max_drawdown" in log and log["valid_max_drawdown"].notna().any():
        score -= 0.05 * standardize(log["valid_max_drawdown"].abs()).fillna(0.0)
        total += 0.05
    return score / total if total > 0 else score


def evaluate_panel_variants(
    hp: pd.DataFrame,
    rankic: pd.DataFrame,
    sw: opt.StagewiseConfig,
    prom: opt.PromConfig | None,
    score_names: Iterable[str],
    rows: list[dict],
    max_variants: bool = True,
) -> dict[tuple[str, int, str, str], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    cache: dict[tuple[str, int, str, str], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for score_name in score_names:
        for smoothing in SMOOTHING_METHODS:
            smoothed_col = f"robust__{score_name}__{smoothing}"
            work = add_rank_smoothing(hp, score_name, smoothing, smoothed_col)
            for top_k in TOPKS:
                rules = ["none"]
                if top_k in (20, 30):
                    rules.append("buffer")
                for buffer_rule in rules:
                    holdings, returns, metrics = evaluate_topk_rule(work, smoothed_col, top_k, buffer_rule)
                    append_robust_log(rows, sw, prom, score_name, top_k, smoothing, buffer_rule, rankic, metrics)
                    cache[(score_name, top_k, smoothing, buffer_rule)] = (holdings, returns, metrics)
    return cache


def select_final(log: pd.DataFrame) -> pd.Series:
    candidates = log.copy()
    candidates["robust_valid_score"] = robust_valid_score(candidates)
    preferred = candidates.loc[
        candidates["top_k"].isin([20, 30])
        & candidates["valid_rankic"].gt(0)
        & candidates["valid_alpha"].gt(-0.05)
    ].copy()
    if preferred.empty:
        preferred = candidates.copy()
    # Choose among validation-near-best candidates by lower validation turnover.
    # This keeps selection validation-only while making the robust version
    # actually prefer more tradable, lower-churn portfolios.
    best_score = float(preferred["robust_valid_score"].max())
    near_best = preferred.loc[preferred["robust_valid_score"] >= best_score - 0.02].copy()
    if not near_best.empty:
        preferred = near_best
    preferred["_prom_ready"] = (preferred.get("preference_function", "none").astype(str) == "linear").astype(int)
    preferred = preferred.sort_values(["valid_turnover", "_prom_ready", "robust_valid_score"], ascending=[True, False, False])
    return preferred.iloc[0], candidates


def final_stagewise_with_stable_test(
    panel: pd.DataFrame,
    features: list[str],
    config: opt.StagewiseConfig,
    stable_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not stable_features:
        return opt.run_stagewise_one(panel, features, config)
    stock_col = opt.get_stock_id_col(panel)
    name_col = opt.get_stock_name_col(panel)
    pred_parts, selected_rows, rankic_rows = [], [], []
    for q in range(config.rolling_window + 1, 31):
        active_features = stable_features if q >= 26 else features
        train_quarters = list(range(q - config.rolling_window, q))
        train = panel.loc[panel["quarter_idx"].isin(train_quarters)].dropna(subset=[LABEL]).copy()
        pred_panel = panel.loc[panel["quarter_idx"].eq(q)].copy()
        if train.empty or pred_panel.empty:
            continue
        y = pd.to_numeric(train[LABEL], errors="coerce").to_numpy(float)
        good = np.isfinite(y)
        train = train.loc[good].copy()
        y = y[good]
        tx, px = opt.preprocess_window(train[active_features], pred_panel[active_features], train["quarter_idx"], pred_panel["quarter_idx"], config)
        kept_features = opt.corr_filter(tx, config.feature_corr_threshold)
        tx = tx[kept_features]
        px = px[kept_features]
        x = tx.to_numpy(float)
        x_pred = px.to_numpy(float)
        selected: list[int] = []
        remaining = list(range(x.shape[1]))
        current_pred = np.full(len(y), y.mean(), dtype=float)
        residual = y - current_pred
        prev_sse = float(np.sum(residual * residual))
        final_beta = np.array([y.mean()])
        train_quarter = train["quarter_idx"].to_numpy(int)
        for step in range(1, config.max_selected_features + 1):
            if not remaining:
                break
            idx, objective_score = opt.choose_feature(x, y, selected, remaining, current_pred, residual, config, train_quarter)
            if idx is None:
                break
            trial = selected + [idx]
            beta, trial_pred = opt.fit_linear(x[:, trial], y, config.refit_method, config.ridge_alpha)
            sse = float(np.sum((y - trial_pred) ** 2))
            improvement = (prev_sse - sse) / prev_sse if prev_sse > 1e-12 else 0.0
            if step > 1 and improvement < config.min_improvement:
                break
            selected = trial
            remaining.remove(idx)
            current_pred = trial_pred
            residual = y - current_pred
            prev_sse = sse
            final_beta = beta
            selected_rows.append(
                {
                    "experiment_id": config.experiment_id,
                    "quarter_idx": q,
                    "rolling_window_start": q - config.rolling_window,
                    "rolling_window_end": q - 1,
                    "step": step,
                    "selected_feature": kept_features[idx],
                    "coefficient": float(beta[-1]),
                    "residual_correlation": float(objective_score),
                    "train_r2": opt.r2_score(y, current_pred),
                    "train_rankic": opt.rank_ic(y, current_pred),
                    "marginal_improvement": improvement,
                    "stable_core_restricted": bool(q >= 26),
                }
            )
        if selected:
            pred = np.column_stack([np.ones(len(pred_panel)), x_pred[:, selected]]) @ final_beta
            selected_features = [kept_features[i] for i in selected]
        else:
            pred = np.full(len(pred_panel), y.mean(), dtype=float)
            selected_features = []
        out_cols = ["quarter_idx", "split", stock_col, LABEL]
        if name_col:
            out_cols.insert(3, name_col)
        out = pred_panel[out_cols].copy().rename(columns={stock_col: "stock_id"})
        if name_col:
            out = out.rename(columns={name_col: "stock_name"})
        else:
            out["stock_name"] = ""
        out["stock_id"] = out["stock_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        out["stagewise_pred"] = pred
        out["stagewise_rank"] = out.groupby("quarter_idx")["stagewise_pred"].rank(method="first", ascending=False)
        out["stagewise_rank_pct"] = out.groupby("quarter_idx")["stagewise_pred"].rank(method="average", pct=True)
        out["selected_feature_count"] = len(selected_features)
        out["selected_features"] = ";".join(selected_features)
        pred_parts.append(out)
        rankic_rows.append(
            {
                "experiment_id": config.experiment_id,
                "quarter_idx": q,
                "split": opt.split_name(q),
                "rankic": opt.rank_ic(out[LABEL], out["stagewise_pred"]),
                "n_stocks": int(out[LABEL].notna().sum()),
                "selected_feature_count": int(len(selected_features)),
                "stable_core_restricted": bool(q >= 26),
            }
        )
    pred_df = pd.concat(pred_parts, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    rankic_df = pd.DataFrame(rankic_rows)
    summary_rows = []
    for split, group in pred_df.groupby("split"):
        summary_rows.append(
            {
                "experiment_id": config.experiment_id,
                "split": split,
                "mean_rankic": float(rankic_df.loc[rankic_df["split"].eq(split), "rankic"].mean()),
                "mean_top20_return": float(group.sort_values(["quarter_idx", "stagewise_pred"], ascending=[True, False]).groupby("quarter_idx").head(20)[LABEL].mean()),
                "mean_top30_return": float(group.sort_values(["quarter_idx", "stagewise_pred"], ascending=[True, False]).groupby("quarter_idx").head(30)[LABEL].mean()),
            }
        )
    return pred_df, selected_df, rankic_df, pd.DataFrame(summary_rows)


def alpha_stability(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "split",
        "strategy",
        "score_col",
        "topk",
        "buffer_rule",
        "mean_quarterly_alpha",
        "alpha",
        "alpha_t_stat",
        "information_ratio",
        "tracking_error",
        "excess_win_rate",
        "alpha_stability_score",
    ]
    return metrics[[c for c in cols if c in metrics.columns]].copy()


def read_original_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = ORIG / "topk_performance_metrics_stagewise.csv"
    rankic_path = ORIG / "stagewise_model_summary.csv"
    metrics = pd.read_csv(metrics_path, encoding="utf-8-sig") if metrics_path.exists() else pd.DataFrame()
    rankic = pd.read_csv(rankic_path, encoding="utf-8-sig") if rankic_path.exists() else pd.DataFrame()
    return metrics, rankic


def comparison_tables(
    final_metrics: pd.DataFrame,
    final_rankic: pd.DataFrame,
    final_returns: pd.DataFrame,
    log: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orig_metrics, orig_rankic = read_original_metrics()
    rows = []
    for split in ["valid", "test"]:
        orig_rankic_value = np.nan
        if not orig_rankic.empty and "mean_rankic" in orig_rankic:
            row = orig_rankic.loc[orig_rankic["split"].eq(split)]
            orig_rankic_value = float(row["mean_rankic"].iloc[0]) if not row.empty else np.nan
        robust_rankic_value = float(final_rankic.loc[final_rankic["split"].eq(split), "rankic"].mean())
        orig = orig_metrics.loc[orig_metrics["split"].eq(split)].iloc[0] if not orig_metrics.empty and not orig_metrics.loc[orig_metrics["split"].eq(split)].empty else pd.Series(dtype=float)
        robust = final_metrics.loc[final_metrics["split"].eq(split)].iloc[0]
        rows.append(
            {
                "split": split,
                "original_rankic": orig_rankic_value,
                "robust_rankic": robust_rankic_value,
                "original_annualized_return": orig.get("annualized_return", np.nan),
                "robust_annualized_return": robust.get("annualized_return", np.nan),
                "original_alpha": orig.get("alpha", np.nan),
                "robust_alpha": robust.get("alpha", np.nan),
                "original_sharpe": orig.get("sharpe_ratio", np.nan),
                "robust_sharpe": robust.get("sharpe_ratio", np.nan),
                "original_information_ratio": np.nan,
                "robust_information_ratio": robust.get("information_ratio", np.nan),
                "original_max_drawdown": orig.get("max_drawdown", np.nan),
                "robust_max_drawdown": robust.get("max_drawdown", np.nan),
                "original_turnover": orig.get("average_turnover", np.nan),
                "robust_turnover": robust.get("average_turnover", np.nan),
                "original_excess_win_rate": np.nan,
                "robust_excess_win_rate": robust.get("excess_win_rate", np.nan),
            }
        )
    comp = pd.DataFrame(rows)
    alpha_rows = []
    for split in ["valid", "test", "all"]:
        robust = final_metrics.loc[final_metrics["split"].eq(split)]
        orig = orig_metrics.loc[orig_metrics["split"].eq(split)] if not orig_metrics.empty else pd.DataFrame()
        alpha_rows.append(
            {
                "split": split,
                "original_alpha": float(orig["alpha"].iloc[0]) if not orig.empty and "alpha" in orig else np.nan,
                "robust_alpha": float(robust["alpha"].iloc[0]) if not robust.empty else np.nan,
                "original_sharpe": float(orig["sharpe_ratio"].iloc[0]) if not orig.empty and "sharpe_ratio" in orig else np.nan,
                "robust_sharpe": float(robust["sharpe_ratio"].iloc[0]) if not robust.empty else np.nan,
                "original_turnover": float(orig["average_turnover"].iloc[0]) if not orig.empty and "average_turnover" in orig else np.nan,
                "robust_turnover": float(robust["average_turnover"].iloc[0]) if not robust.empty else np.nan,
                "robust_information_ratio": float(robust["information_ratio"].iloc[0]) if not robust.empty else np.nan,
                "robust_excess_win_rate": float(robust["excess_win_rate"].iloc[0]) if not robust.empty else np.nan,
            }
        )
    alpha_comp = pd.DataFrame(alpha_rows)
    conversion = []
    for _, row in log.iterrows():
        conversion.append(
            {
                "split": "valid",
                "score_name": row["hybrid_score_name"],
                "top_k": row["top_k"],
                "rankic": row["valid_rankic"],
                "annualized_return": row["valid_annualized_return"],
                "alpha": row["valid_alpha"],
                "sharpe_ratio": row["valid_sharpe"],
                "information_ratio": row["valid_information_ratio"],
                "max_drawdown": row["valid_max_drawdown"],
                "turnover": row["valid_turnover"],
                "excess_win_rate": row["valid_excess_win_rate"],
                "signal_conversion_comment": "positive rankic with robust-selected portfolio metrics; validation only for selection",
            }
        )
        conversion.append(
            {
                "split": "test",
                "score_name": row["hybrid_score_name"],
                "top_k": row["top_k"],
                "rankic": row["test_rankic"],
                "annualized_return": row["test_annualized_return"],
                "alpha": row["test_alpha"],
                "sharpe_ratio": row["test_sharpe"],
                "information_ratio": row["test_information_ratio"],
                "max_drawdown": row["test_max_drawdown"],
                "turnover": row["test_turnover"],
                "excess_win_rate": row["test_excess_win_rate"],
                "signal_conversion_comment": "holdout diagnostics only; not used for parameter choice",
            }
        )
    conversion_df = pd.DataFrame(conversion)
    return comp, alpha_comp, conversion_df


def promethee_preference_sensitivity(panel: pd.DataFrame, pred: pd.DataFrame, selected: pd.DataFrame, base_prom: opt.PromConfig, top_k: int) -> pd.DataFrame:
    rows = []
    preferences = [
        ("usual", np.nan, np.nan, np.nan),
        ("u_shape", 0.05, np.nan, np.nan),
        ("v_shape", 0.0, 0.25, np.nan),
        ("level", 0.05, 0.25, np.nan),
        ("linear", 0.05, 0.25, np.nan),
        ("gaussian", np.nan, np.nan, 0.20),
    ]
    for pref, q, p, sigma in preferences:
        prom = opt.PromConfig(
            promethee_top_n=base_prom.promethee_top_n,
            entropy_weight_method=base_prom.entropy_weight_method,
            preference_function=pref,
            min_selected_count=base_prom.min_selected_count,
            sign_stability_threshold=base_prom.sign_stability_threshold,
        )
        pro, _ = opt.build_promethee_scores(panel, pred, selected, prom)
        hp = opt.build_hybrid_panel(pred, pro)
        for k in [10, 20, 30]:
            _, _, metrics = evaluate_topk_rule(hp, "pure_promethee_score", k, "none")
            for _, m in metrics.iterrows():
                rows.append(
                    {
                        "preference_function": pref,
                        "q": q,
                        "p": p,
                        "sigma": sigma,
                        "split": m["split"],
                        "mean_rankic": float(hp.loc[hp["split"].eq(m["split"]), [LABEL, "promethee_net_flow"]].corr(method="spearman").iloc[0, 1])
                        if m["split"] != "all" and hp.loc[hp["split"].eq(m["split"])].shape[0] > 2
                        else np.nan,
                        f"top{k}_annualized_return": m["annualized_return"],
                        f"top{k}_sharpe": m["sharpe_ratio"],
                        "alpha": m["alpha"],
                        "beta": m["beta"],
                        "max_drawdown": m["max_drawdown"],
                        "turnover": m["average_turnover"],
                        "rating_separation": np.nan,
                        "avg_net_flow_std": float(pro.groupby("quarter_idx")["promethee_net_flow"].std().mean()),
                        "avg_top20_overlap_with_selected": np.nan,
                        "selected_as_final": pref == "linear",
                        "reason": "linear is final robust PROMETHEE preference function; test metrics are diagnostics only"
                        if pref == "linear"
                        else "sensitivity comparison only; not selected by test performance",
                    }
                )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    keys = ["preference_function", "q", "p", "sigma", "split"]
    metric_cols = [c for c in raw.columns if c not in keys and c not in ["selected_as_final", "reason"]]
    agg = raw.groupby(keys, dropna=False)[metric_cols].first().reset_index()
    flags = raw.groupby(keys, dropna=False).agg(selected_as_final=("selected_as_final", "max"), reason=("reason", "first")).reset_index()
    return agg.merge(flags, on=keys, how="left")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_opt_globals()
    verify_python()
    panel, composite, features = opt.load_model_panel()
    configs = robust_stagewise_configs()
    stagewise_results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, opt.StagewiseConfig]] = {}
    rows: list[dict] = []
    prelim_cache: dict[str, dict[tuple[str, int, str, str], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = {}
    for cfg in configs:
        print(f"[stagewise] {cfg.experiment_id} max={cfg.max_selected_features} min_imp={cfg.min_improvement} refit={cfg.refit_method} alpha={cfg.ridge_alpha}")
        pred, selected, rankic, summary = opt.run_stagewise_one(panel, features, cfg)
        stagewise_results[cfg.experiment_id] = (pred, selected, rankic, summary, cfg)
        hp = pred.copy()
        hp["pure_stagewise_score"] = hp.groupby("quarter_idx")["stagewise_pred"].rank(method="average", pct=True)
        cache = evaluate_panel_variants(hp, rankic, cfg, None, ["pure_stagewise_score"], rows)
        prelim_cache[cfg.experiment_id] = cache
    prelim_log = pd.DataFrame(rows)
    prelim_log["robust_valid_score"] = robust_valid_score(prelim_log)
    top_stagewise = prelim_log.sort_values("robust_valid_score", ascending=False)["experiment_id"].drop_duplicates().head(8).tolist()
    prom = opt.PromConfig(
        promethee_top_n=20,
        entropy_weight_method="smooth_80_20",
        preference_function="linear",
        min_selected_count=2,
        sign_stability_threshold=0.55,
    )
    hybrid_cache: dict[str, dict[tuple[str, int, str, str], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = {}
    for exp_id in top_stagewise:
        pred, selected, rankic, _, cfg = stagewise_results[exp_id]
        print(f"[promethee/hybrid] {exp_id}")
        pro, weights = opt.build_promethee_scores(panel, pred, selected, prom)
        hp = opt.build_hybrid_panel(pred, pro)
        hybrid_cache[exp_id] = evaluate_panel_variants(hp, rankic, cfg, prom, HYBRIDS, rows)
    log = pd.DataFrame(rows).drop_duplicates(
        subset=["experiment_id", "hybrid_score_name", "top_k", "rank_smoothing", "buffer_rule", "promethee_top_n", "entropy_weight_method", "preference_function"],
        keep="last",
    )
    final_row, scored_log = select_final(log)
    final_exp = str(final_row["experiment_id"])
    final_cfg = stagewise_results[final_exp][4]
    base_pred, base_selected, base_rankic, _, _ = stagewise_results[final_exp]
    stability = feature_stability(base_selected.loc[base_selected["quarter_idx"].between(13, 25)], composite)
    core, core_rule = stable_core_features(stability)
    stable_features = core["feature"].astype(str).tolist()
    pred, selected, rankic, summary = final_stagewise_with_stable_test(panel, features, final_cfg, stable_features)
    stability_final = feature_stability(selected, composite)
    core_final, core_rule_final = stable_core_features(stability_final.loc[stability_final["feature"].isin(stable_features)] if stable_features else stability_final)
    pro, weights = opt.build_promethee_scores(panel, pred, selected, prom)
    hp = opt.build_hybrid_panel(pred, pro)
    final_score = str(final_row["hybrid_score_name"])
    final_topk = int(final_row["top_k"])
    final_smoothing = str(final_row["rank_smoothing"])
    final_buffer = str(final_row["buffer_rule"])
    final_score_col = f"robust__{final_score}__{final_smoothing}"
    final_panel = add_rank_smoothing(hp, final_score, final_smoothing, final_score_col)
    holdings, returns, metrics = evaluate_topk_rule(final_panel, final_score_col, final_topk, final_buffer)
    rating, rating_perf = opt.rating_outputs(final_panel.rename(columns={final_score_col: "final_robust_score"}), "final_robust_score")
    scored_log = scored_log.copy()
    scored_log["selected_as_final"] = False
    scored_log["reason"] = ""
    mask = (
        scored_log["experiment_id"].eq(final_exp)
        & scored_log["hybrid_score_name"].eq(final_score)
        & scored_log["top_k"].eq(final_topk)
        & scored_log["rank_smoothing"].eq(final_smoothing)
        & scored_log["buffer_rule"].eq(final_buffer)
        & scored_log["preference_function"].eq(prom.preference_function)
        & scored_log["entropy_weight_method"].eq(prom.entropy_weight_method)
    )
    if not mask.any():
        mask = (
            scored_log["experiment_id"].eq(final_exp)
            & scored_log["hybrid_score_name"].eq(final_score)
            & scored_log["top_k"].eq(final_topk)
            & scored_log["rank_smoothing"].eq(final_smoothing)
            & scored_log["buffer_rule"].eq(final_buffer)
        )
    selected_idx = scored_log.index[mask][0]
    metric_updates = {
        "train_rankic": float(rankic.loc[rankic["split"].eq("train"), "rankic"].mean()),
        "valid_rankic": float(rankic.loc[rankic["split"].eq("valid"), "rankic"].mean()),
        "test_rankic": float(rankic.loc[rankic["split"].eq("test"), "rankic"].mean()),
        "valid_icir": rankic_icir(rankic, "valid"),
        "test_icir": rankic_icir(rankic, "test"),
    }
    for split in ["valid", "test"]:
        mrow = metrics.loc[metrics["split"].eq(split)].iloc[0]
        metric_updates[f"{split}_annualized_return"] = mrow["annualized_return"]
        metric_updates[f"{split}_sharpe"] = mrow["sharpe_ratio"]
        metric_updates[f"{split}_alpha"] = mrow["alpha"]
        metric_updates[f"{split}_information_ratio"] = mrow["information_ratio"]
        metric_updates[f"{split}_max_drawdown"] = mrow["max_drawdown"]
        metric_updates[f"{split}_turnover"] = mrow["average_turnover"]
        metric_updates[f"{split}_excess_win_rate"] = mrow["excess_win_rate"]
    for col, val in metric_updates.items():
        scored_log.loc[selected_idx, col] = val
    scored_log["robust_valid_score"] = robust_valid_score(scored_log)
    scored_log.loc[selected_idx, "selected_as_final"] = True
    scored_log.loc[selected_idx, "reason"] = (
        "Selected by validation-only near-best robust_valid_score, then lower validation turnover; "
        "final metrics refreshed after stable-core and buffer evaluation."
    )
    comp, alpha_comp, conversion = comparison_tables(metrics, rankic, returns, scored_log)
    pref_comp = promethee_preference_sensitivity(panel, pred, selected, prom, final_topk)

    config = {
        **asdict(final_cfg),
        **asdict(prom),
        "hybrid_score_name": final_score,
        "top_k": final_topk,
        "rank_smoothing": final_smoothing,
        "buffer_rule": final_buffer,
        "stable_core_rule": core_rule,
        "stable_core_feature_count": len(stable_features),
        "final_score_column": final_score_col,
        "robust_selection_basis": "validation-only near-best robust_valid_score, then lower validation turnover; test metrics not used for selection",
    }

    scored_log.to_csv(OUT / "robust_experiment_log.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "final_robust_algorithm_config.json", config)
    comp.to_csv(OUT / "robust_vs_original_comparison.csv", index=False, encoding="utf-8-sig")
    stability_final.to_csv(OUT / "stagewise_feature_stability.csv", index=False, encoding="utf-8-sig")
    core_final.to_csv(OUT / "stable_core_features.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "stagewise_selected_features_by_quarter_robust.csv", index=False, encoding="utf-8-sig")
    opt.feature_frequency(selected, composite).to_csv(OUT / "stagewise_feature_frequency_robust.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(OUT / "stagewise_prediction_panel_robust.csv", index=False, encoding="utf-8-sig")
    rankic.to_csv(OUT / "stagewise_quarterly_rankic_robust.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "stagewise_model_summary_robust.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "topk_performance_metrics_robust.csv", index=False, encoding="utf-8-sig")
    returns.to_csv(OUT / "topk_quarterly_returns_robust.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(OUT / "topk_holdings_robust.csv", index=False, encoding="utf-8-sig")
    rating.to_csv(OUT / "rating_panel_robust.csv", index=False, encoding="utf-8-sig")
    rating_perf.to_csv(OUT / "rating_performance_robust.csv", index=False, encoding="utf-8-sig")
    alpha_stability(metrics).to_csv(OUT / "alpha_stability_metrics.csv", index=False, encoding="utf-8-sig")
    returns.to_csv(OUT / "excess_return_by_quarter.csv", index=False, encoding="utf-8-sig")
    alpha_comp.to_csv(OUT / "robust_vs_original_alpha_comparison.csv", index=False, encoding="utf-8-sig")
    conversion.to_csv(OUT / "signal_to_portfolio_conversion.csv", index=False, encoding="utf-8-sig")
    pref_comp.to_csv(OUT / "promethee_preference_function_comparison.csv", index=False, encoding="utf-8-sig")
    # Extra outputs useful for figures and review.
    weights.to_csv(OUT / "entropy_weights_robust.csv", index=False, encoding="utf-8-sig")
    pro.to_csv(OUT / "promethee_scores_robust.csv", index=False, encoding="utf-8-sig")
    final_panel.to_csv(OUT / "hybrid_scores_robust.csv", index=False, encoding="utf-8-sig")

    subprocess.run([str(PYTHON), str(ROOT / "scripts" / "05_generate_robust_figures.py")], cwd=str(ROOT), check=True)
    subprocess.run([str(PYTHON), str(ROOT / "scripts" / "06_update_model_explanation_md.py")], cwd=str(ROOT), check=True)
    print("[done] robust_config=" + json.dumps(config, ensure_ascii=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
