# Weekly Comparison Metrics Design and Implementation Plan

Last updated: 2026-05-31

This document outlines the design and implementation plan for the **Weekly Comparison Metrics** (Task 2 in the Near-Term Paper Comparison Foundation). The goal is to generate durable weekly metrics artifacts (`logs/comparison_metrics_YYYY-MM-DD.json` and `logs/comparison_metrics_latest.csv`) that evaluate all paper trading accounts and benchmarks side-by-side.

## Requirements

We need to compute and store the following metrics for each account and benchmark:

- **Returns**: Cumulative return, weekly return
- **Risk**: Volatility (annualized), Sharpe ratio (annualized, assuming risk-free rate = 2.0%), max drawdown
- **Trading Activity**: Turnover, cash exposure (average cash weight)
- **Hit Rate**: Percentage of positive weekly returns
- **Relationship to Benchmarks (SPY and QQQ)**:
  - Beta ($\beta$) to SPY and QQQ
  - Tracking error vs SPY and QQQ
- **Strategy Diagnostics**:
  - Weeks in fallback (percentage/count of runs where fallback was active)
  - Realized vs target weight drift (average absolute difference between actual weight and target weight across all symbols)

## Mathematical Formulas

Let $R_{i,t}$ be the weekly return of account/benchmark $i$ at week $t$, and $R_{b,t}$ be the weekly return of benchmark $b$ (SPY or QQQ) at week $t$.

1. **Cumulative Return**:
   $$CR_i = \prod_{t=1}^T (1 + R_{i,t}) - 1$$

2. **Annualized Volatility**:
   $$\sigma_i = \text{Std}(R_i) \times \sqrt{52}$$

3. **Annualized Sharpe Ratio** (assuming risk-free rate $R_f = 0.02$ annualized, or $R_{f,\text{weekly}} = 0.02 / 52$):
   $$\text{Sharpe}_i = \frac{\text{Mean}(R_i - R_{f,\text{weekly}})}{\text{Std}(R_i)} \times \sqrt{52}$$

4. **Max Drawdown**:
   Let $V_{i,t}$ be the cumulative portfolio value (growth of \$1) at week $t$.
   $$\text{Drawdown}_{i,t} = \frac{V_{i,t} - \max_{\tau \le t} V_{i,\tau}}{\max_{\tau \le t} V_{i,\tau}}$$
   $$\text{Max Drawdown}_i = \min_t \text{Drawdown}_{i,t}$$

5. **Turnover**:
   Let $w_{i,j,t}$ be the weight of asset $j$ in portfolio $i$ at week $t$.
   $$\text{Turnover}_{i,t} = \sum_j |w_{i,j,t} - w_{i,j,t-1}|$$

6. **Beta ($\beta$) to Benchmark $b$**:
   $$\beta_i = \frac{\text{Cov}(R_i, R_b)}{\text{Var}(R_b)}$$

7. **Tracking Error vs Benchmark $b$**:
   $$\text{Tracking Error}_i = \text{Std}(R_i - R_b) \times \sqrt{52}$$

8. **Realized vs Target Weight Drift**:
   Let $w_{i,j,t}^{\text{actual}}$ be the actual weight and $w_{i,j,t}^{\text{target}}$ be the target weight of asset $j$ at week $t$.
   $$\text{Drift}_{i,t} = \sum_j |w_{i,j,t}^{\text{actual}} - w_{i,j,t}^{\text{target}}|$$

## Storage Formats

### 1. JSON Artifact (`logs/comparison_metrics_YYYY-MM-DD.json`)

Contains the complete, nested metrics dataset for the current week:

```json
{
  "date": "2026-05-29",
  "accounts": {
    "FinRL": {
      "weekly_return": 0.0325,
      "cumulative_return": 0.0446,
      "volatility": 0.082,
      "sharpe_ratio": 1.25,
      "max_drawdown": 0.0,
      "turnover": 0.66,
      "cash_exposure": 0.0,
      "hit_rate": 0.75,
      "beta_spy": 0.85,
      "beta_qqq": 0.92,
      "tracking_error_spy": 0.054,
      "tracking_error_qqq": 0.042,
      "weeks_in_fallback": 0,
      "weight_drift": 0.012
    },
    "AR": { ... }
  },
  "benchmarks": {
    "SPY": { ... },
    "QQQ": { ... }
  }
}
```

### 2. CSV Artifact (`logs/comparison_metrics_latest.csv`)

A flat table containing the most recent metrics for side-by-side comparison:

```csv
account,weekly_return,cumulative_return,volatility,sharpe_ratio,max_drawdown,turnover,cash_exposure,hit_rate,beta_spy,beta_qqq,tracking_error_spy,tracking_error_qqq,weeks_in_fallback,weight_drift
FinRL,0.0325,0.0446,0.082,1.25,0.0,0.66,0.0,0.75,0.85,0.92,0.054,0.042,0,0.012
AR,0.0056,0.0108,0.054,0.62,-0.0041,0.0,0.0,0.50,1.0,1.15,0.0,0.031,4,0.0
SPY,0.0055,0.0230,0.048,0.95,0.0,N/A,N/A,1.0,1.0,0.82,0.0,0.024,N/A,N/A
QQQ,0.0213,0.0343,0.072,1.10,0.0,N/A,N/A,0.75,1.21,1.0,0.024,0.0,N/A,N/A
```

## Implementation Plan

1. **Calculate Metrics Function**: Implement `calculate_comparison_metrics(conn, run_date)` inside `track_metrics.py`. This function will load historical weekly return data from `weekly_snapshot` and `weekly_weights` tables and compute all annualized and relative metrics.
2. **Retrieve Realized and Target Weights**: Query the `weekly_weights` table to calculate turnover and realized vs target weight drift.
3. **Write Outputs**: Integrate this function at the end of `track_metrics.py` to write both JSON and CSV files.
4. **Unit Testing**: Add focused tests under `tests/test_weekly_workflow.py` to verify that the mathematical formulas are calculated correctly and outputs are generated exactly as designed.
