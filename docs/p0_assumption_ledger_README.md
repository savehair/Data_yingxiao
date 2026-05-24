# P0 Assumption Ledger

维护项目中未明确指定的建模、标签、评级、回测和数据对齐假设。

## 输入

- `outputs/assumption_ledger_seed.csv`: 初始假设台账，通常由 `scripts/p0_inventory_and_requirements.py` 生成。
- `configs/project_assumptions.yaml`: 用户覆盖配置。只需编辑 `assumptions:` 下的键值。

## 运行

```powershell
python scripts/p0_assumption_ledger.py
```

显式指定路径：

```powershell
python scripts/p0_assumption_ledger.py `
  --assumption-file "outputs/assumption_ledger_seed.csv" `
  --override-yaml "configs/project_assumptions.yaml" `
  --output-dir "outputs"
```

## 输出

- `outputs/assumption_ledger_final.csv`: 标准化后的最终假设台账。
- `configs/project_assumptions.yaml`: 本次运行实际采用的配置，会被写回。
- `outputs/assumption_config_summary.json`: 本次运行采用配置和高风险项摘要。

## 规则

每个必备未指定项必须包含 `value`、`default_assumption`、`alternatives`、`risk_note`。终端会打印“本次运行实际采用的配置摘要”和“高风险未指定项”。
