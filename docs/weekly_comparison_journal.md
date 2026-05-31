# Weekly Comparison Journal

Use this document as a running template for weekly FinRL vs AR vs benchmark
reviews. Copy the template section for each new run.

## Weekly Review Template

### Week Of YYYY-MM-DD

Run status:

- Price refresh:
- FinRL account execution:
- AR account execution:
- Metrics tracker:
- Dashboard regenerated:
- Any failed or skipped step:

Market context:

- SPY weekly return:
- QQQ weekly return:
- Notable market regime:
- Notable volatility or drawdown events:

Strategy results:

| Account | Weekly Return | Cumulative Return | Cash | Turnover | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FinRL | | | | | |
| AR | | | | | |
| SPY | | | | | |
| QQQ | | | | | |

AR diagnostics:

- Active groups:
- Ranked groups:
- Fallback used:
- Fallback reason:
- Group strength notes:

FinRL diagnostics:

- Active symbols:
- Largest position:
- Largest drift:
- ML bucket or symbol changes:

Execution quality:

- Orders submitted:
- Orders filled:
- Orders rejected:
- Estimated slippage:
- Post-trade drift:
- Reconciliation issues:

Data quality:

- Stale symbols:
- Missing prices:
- Benchmark data issues:
- Dashboard or metrics issues:

Decision notes:

- Keep running unchanged:
- Change needed before next run:
- Investigation needed:
- Production-readiness lesson:
- RL-readiness lesson:

Artifacts:

- Execution log:
- Dashboard:
- Metrics artifact:
- Reconciliation report:
- Strategy decision records:

## Running Watch List

- Confirm SPY dashboard stays aligned to weekly snapshot dates.
- Track how often AR enters fallback and why.
- Track whether QQQ is too demanding as the group-strength benchmark.
- Track live target weights vs filled actual weights.
- Track whether weekly execution remains idempotent and recoverable.
- Track whether FinRL outperformance persists after accounting for turnover and
  concentration.

