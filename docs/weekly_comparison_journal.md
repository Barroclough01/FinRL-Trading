# Weekly Comparison Journal

Use this document as a running template for weekly FinRL vs AR vs benchmark
reviews. Copy the template section for each new run.

## Past Weekly Review Entries

### Week Of 2026-05-10

Run status:

- Price refresh: Success
- FinRL account execution: Success (submitted as OPG)
- AR account execution: Success (submitted as OPG)
- Metrics tracker: Success
- Dashboard regenerated: Success
- Any failed or skipped step: None (Initial attempts failed due to missing AXP/CAT data and Alpaca 404, but resolved on final retry)

Market context:

- SPY weekly return: +0.83%
- QQQ weekly return: +0.00% (Baseline week)
- Notable market regime: `risk_on`
- Notable volatility or drawdown events: Low volatility, steady upward trend

Strategy results:

| Account | Weekly Return | Cumulative Return | Cash | Turnover | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FinRL | +0.00% | +0.00% | ~0.0% | N/A | Initialized portfolio with ORCL, SATS, MCHP |
| AR | +0.00% | +0.00% | ~0.0% | N/A | Initialized portfolio with SPY, QQQ, IAU, XLU, XLV |
| SPY | +0.83% | +0.00% | N/A | N/A | Benchmark |
| QQQ | +0.00% | +0.00% | N/A | N/A | Benchmark |

AR diagnostics:

- Active groups: `['FALLBACK']`
- Ranked groups: `['group_a_growth_tech', 'group_b_cyclical', 'group_d_defensive', 'group_c_real_assets']`
- Fallback used: Yes
- Fallback reason: `all_groups_negative_excess_return`
- Group strength notes: All mega-cap groups failed to exceed the QQQ positive excess return threshold.

FinRL diagnostics:

- Active symbols: `ORCL` (33.33%), `SATS` (33.33%), `MCHP` (33.33%)
- Largest position: `ORCL` (33.33%)
- Largest drift: N/A (Initial allocation)
- ML bucket or symbol changes: Initialized positions in tech bucket.

Execution quality:

- Orders submitted: 11 (FinRL: 3 sells, 3 buys; AR: 0 sells, 5 buys)
- Orders filled: 11
- Orders rejected: 0
- Estimated slippage: Low
- Post-trade drift: Low
- Reconciliation issues: None

Data quality:

- Stale symbols: None
- Missing prices: Fixed missing AXP/CAT data
- Benchmark data issues: None
- Dashboard or metrics issues: None

Decision notes:

- Keep running unchanged: Yes, proceed with shadow comparisons.
- Change needed before next run: None.
- Investigation needed: Investigate why Alpaca returned 404 during earlier run.
- Production-readiness lesson: Pre-trade data completeness checks are critical.
- RL-readiness lesson: Standardizing target weights allows seamless execution across accounts.

Artifacts:

- Execution log: `logs/execution_2026-05-10.json`
- Dashboard: `logs/dashboard.html`
- Metrics artifact: `data/finrl_trading.db`
- Reconciliation report: None
- Strategy decision records: `src/strategies/output/audit/adaptive_rotation/audit_2026-05-10.json`

---

### Week Of 2026-05-15

Run status:

- Price refresh: Success
- FinRL account execution: Success (submitted as OPG)
- AR account execution: Success (submitted as OPG)
- Metrics tracker: Success
- Dashboard regenerated: Success
- Any failed or skipped step: None

Market context:

- SPY weekly return: +0.79%
- QQQ weekly return: +1.20%
- Notable market regime: `risk_on`
- Notable volatility or drawdown events: Tech sector strength continues to lead the market.

Strategy results:

| Account | Weekly Return | Cumulative Return | Cash | Turnover | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FinRL | +0.83% | +0.83% | ~0.0% | High | Rebalanced to TXN, ON, SATS |
| AR | -0.41% | -0.41% | ~0.0% | Low | Maintained fallback allocation |
| SPY | +0.79% | +1.43% | N/A | N/A | Benchmark |
| QQQ | +1.20% | +1.20% | N/A | N/A | Benchmark |

AR diagnostics:

- Active groups: `['FALLBACK']`
- Ranked groups: `['group_a_growth_tech', 'group_b_cyclical', 'group_c_real_assets', 'group_d_defensive']`
- Fallback used: Yes
- Fallback reason: `all_groups_negative_excess_return`
- Group strength notes: Mega-cap groups continue to underperform QQQ.

FinRL diagnostics:

- Active symbols: `TXN` (33.33%), `ON` (33.33%), `SATS` (33.33%)
- Largest position: `TXN` (33.33%)
- Largest drift: Fully exited `MCHP` and `ORCL`, entered `TXN` and `ON`.
- ML bucket or symbol changes: Sells: MCHP, ORCL, SATS (partial); Buys: TXN, ON.

Execution quality:

- Orders submitted: 9 (FinRL: 3 sells, 2 buys; AR: 3 sells, 1 buy)
- Orders filled: 9
- Orders rejected: 0
- Estimated slippage: Low
- Post-trade drift: Low
- Reconciliation issues: None

Data quality:

- Stale symbols: None
- Missing prices: None
- Benchmark data issues: None
- Dashboard or metrics issues: None

Decision notes:

- Keep running unchanged: Yes, shadow comparison proceeds cleanly.
- Change needed before next run: None.
- Investigation needed: None.
- Production-readiness lesson: Fractional order execution needs careful handling with time-in-force limits.
- RL-readiness lesson: Rebalance phase transitions (sells then buys) prevent margin issues.

Artifacts:

- Execution log: `logs/execution_2026-05-15.json`
- Dashboard: `logs/dashboard.html`
- Metrics artifact: `data/finrl_trading.db`
- Reconciliation report: None
- Strategy decision records: `src/strategies/output/audit/adaptive_rotation/audit_2026-05-15.json`

---

### Week Of 2026-05-22

Run status:

- Price refresh: Success
- FinRL account execution: Success (submitted live during market hours)
- AR account execution: Success (submitted live during market hours)
- Metrics tracker: Success
- Dashboard regenerated: Success
- Any failed or skipped step: None

Market context:

- SPY weekly return: +0.61%
- QQQ weekly return: +0.07%
- Notable market regime: `risk_on`
- Notable volatility or drawdown events: Market consolidation, low volume week.

Strategy results:

| Account | Weekly Return | Cumulative Return | Cash | Turnover | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FinRL | +0.34% | +1.17% | ~0.0% | Moderate | Rebalanced to MCHP, TXN, ON |
| AR | +0.94% | +0.52% | ~0.0% | Low | Maintained fallback allocation |
| SPY | +0.61% | +1.31% | N/A | N/A | Benchmark |
| QQQ | +0.07% | +1.27% | N/A | N/A | Benchmark |

AR diagnostics:

- Active groups: `['FALLBACK']`
- Ranked groups: `['group_a_growth_tech', 'group_c_real_assets', 'group_b_cyclical', 'group_d_defensive']`
- Fallback used: Yes
- Fallback reason: `all_groups_negative_excess_return`
- Group strength notes: Mega-caps underperforming QQQ trend filter.

FinRL diagnostics:

- Active symbols: `MCHP` (33.33%), `TXN` (33.33%), `ON` (33.33%)
- Largest position: `MCHP` (33.33%)
- Largest drift: Exited `SATS`, entered `MCHP`.
- ML bucket or symbol changes: Sells: SATS, TXN (partial); Buys: MCHP, ON (partial).

Execution quality:

- Orders submitted: 7 (FinRL: 2 sells, 2 buys; AR: 2 sells, 1 buy)
- Orders filled: 7
- Orders rejected: 0
- Estimated slippage: Low (Smooth live execution)
- Post-trade drift: Low
- Reconciliation issues: None

Data quality:

- Stale symbols: None
- Missing prices: None
- Benchmark data issues: None
- Dashboard or metrics issues: None

Decision notes:

- Keep running unchanged: Yes, shadow comparison active.
- Change needed before next run: None.
- Investigation needed: None.
- Production-readiness lesson: Live market orders result in fast execution but can suffer slightly from bid-ask spread.
- RL-readiness lesson: Mid-caps exhibit higher idiosyncratic drift than mega-caps.

Artifacts:

- Execution log: `logs/execution_2026-05-22.json`
- Dashboard: `logs/dashboard.html`
- Metrics artifact: `data/finrl_trading.db`
- Reconciliation report: None
- Strategy decision records: `src/strategies/output/audit/adaptive_rotation/audit_2026-05-22.json`

---

### Week Of 2026-05-29

Run status:

- Price refresh: Success
- FinRL account execution: Success (submitted as OPG)
- AR account execution: Success (submitted as OPG)
- Metrics tracker: Success
- Dashboard regenerated: Success
- Any failed or skipped step: None

Market context:

- SPY weekly return: +0.55%
- QQQ weekly return: +2.13%
- Notable market regime: `risk_on`
- Notable volatility or drawdown events: Tech sector experienced a massive surge, driving QQQ up over 2.1% in a single week.

Strategy results:

| Account | Weekly Return | Cumulative Return | Cash | Turnover | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FinRL | +3.25% | +4.46% | ~0.0% | High | Rebalanced to ORCL, SATS, MCHP |
| AR | +0.56% | +1.08% | ~0.0% | Low | Maintained fallback allocation |
| SPY | +0.55% | +2.30% | N/A | N/A | Benchmark |
| QQQ | +2.13% | +3.43% | N/A | N/A | Benchmark |

AR diagnostics:

- Active groups: `['FALLBACK']`
- Ranked groups: `['group_a_growth_tech', 'group_b_cyclical', 'group_d_defensive', 'group_c_real_assets']`
- Fallback used: Yes
- Fallback reason: `all_groups_negative_excess_return`
- Group strength notes: Mega-cap groups continue to underperform QQQ despite broad tech rally.

FinRL diagnostics:

- Active symbols: `ORCL` (33.33%), `SATS` (33.33%), `MCHP` (33.33%)
- Largest position: `ORCL` (33.33%)
- Largest drift: Exited `TXN` and `ON`, re-entered `ORCL` and `SATS`.
- ML bucket or symbol changes: Sells: TXN, ON; Buys: ORCL, SATS, MCHP (partial).

Execution quality:

- Orders submitted: 7 (FinRL: 2 sells, 3 buys; AR: 1 sell, 1 buy)
- Orders filled: 7
- Orders rejected: 0
- Estimated slippage: Low
- Post-trade drift: Low
- Reconciliation issues: None

Data quality:

- Stale symbols: None
- Missing prices: None
- Benchmark data issues: None
- Dashboard or metrics issues: None

Decision notes:

- Keep running unchanged: Yes, shadow comparison active.
- Change needed before next run: Implement the Pre-Trade Validation gate to secure future runs against missing data.
- Investigation needed: Analyze why mega-caps in AR baseline underperformed QQQ so persistently.
- Production-readiness lesson: Strategy decision records and audit trails are invaluable for retrospective analysis.
- RL-readiness lesson: The high-turnover FinRL model captured the ORCL/SATS surge beautifully, but transaction costs must be modeled.

Artifacts:

- Execution log: `logs/execution_2026-05-29.json`
- Dashboard: `logs/dashboard.html`
- Metrics artifact: `data/finrl_trading.db`
- Reconciliation report: None
- Strategy decision records: `src/strategies/output/audit/adaptive_rotation/audit_2026-05-29.json`

## Running Watch List

- Confirm SPY dashboard stays aligned to weekly snapshot dates.
- Track how often AR enters fallback and why.
- Track whether QQQ is too demanding as the group-strength benchmark.
- Track live target weights vs filled actual weights.
- Track whether weekly execution remains idempotent and recoverable.
- Track whether FinRL outperformance persists after accounting for turnover and
  concentration.
