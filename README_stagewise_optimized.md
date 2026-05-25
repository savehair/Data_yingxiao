# Stagewise-PROMETHEE Optimized Pipeline

This branch adds a validation-selected optimization pipeline for the course project route:

1. audited 231 macro-micro composite factors;
2. rolling-window forward stepwise regression;
3. entropy-weighted PROMETHEE using stagewise-selected factors only;
4. Hybrid Score construction;
5. Top-K backtest and 13-grade rating.

The script writes new results to `outputs_optimized/` and `figures_optimized/` so the previous `outputs/` and `figures/` results are not overwritten.

## Run

Use the project Conda interpreter explicitly:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe run_stagewise_optimization.py
```

The script verifies that interpreter before running. Parameter selection uses only train and validation metrics. Test metrics are computed after final selection as holdout diagnostics.

## Main Outputs

- `outputs_optimized/optimization_experiment_log.csv`
- `outputs_optimized/final_selected_algorithm_config.json`
- `outputs_optimized/final_algorithm_comparison_summary.csv`
- `outputs_optimized/stagewise_selected_features_by_quarter.csv`
- `outputs_optimized/stagewise_feature_frequency.csv`
- `outputs_optimized/stagewise_prediction_panel.csv`
- `outputs_optimized/stagewise_quarterly_rankic.csv`
- `outputs_optimized/stagewise_model_summary.csv`
- `outputs_optimized/entropy_weights_stagewise.csv`
- `outputs_optimized/promethee_scores_stagewise.csv`
- `outputs_optimized/hybrid_scores_stagewise.csv`
- `outputs_optimized/topk_quarterly_returns_stagewise.csv`
- `outputs_optimized/topk_performance_metrics_stagewise.csv`
- `outputs_optimized/topk_holdings_stagewise.csv`
- `outputs_optimized/rating_panel_stagewise.csv`
- `outputs_optimized/rating_performance_stagewise.csv`
- `final_stagewise_report_optimized.ipynb`

## Additional Visual Diagnostics

After the main optimization output exists, generate the extended diagnostic figures with:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe scripts\06_generate_additional_visualizations.py
```

This writes an index file to `outputs_optimized/additional_visualization_manifest.csv` and adds figures that explain the result from several angles: parameter comparisons, direct score-scheme comparisons, RankIC by split, Top20 return versus benchmark, cumulative equity and drawdown, score decile returns, rating group separation, selected factor heatmap, PROMETHEE preference-function sensitivity, valid/test stability, and turnover/holding count.

## Leakage Controls

- `future_return_1q` is used only as the label and for ex-post evaluation.
- Rolling stepwise training for quarter `q` uses only quarters before `q`.
- Imputation, winsorization, scaling, and correlation filtering are fit within each rolling training window.
- PROMETHEE feature direction and feature pool are derived from stagewise-selected factors only.
- All ranking, rating, PROMETHEE, and Top-K operations are performed within each quarter.
- Final configuration is selected by validation score only; test performance is not used for tuning.
