# Post-Trade Reconciliation Design and Implementation Plan

Last updated: 2026-06-12
Status: **IMPLEMENTED IN THE CURRENT WORKFLOW (2026-06-12)**

This document records the implemented post-trade reconciliation flow for Task 5 in the roadmap. The current weekly run now writes `logs/reconciliation_YYYY-MM-DD.json` and compares target weights, actual positions, cash/equity, and order outcomes after execution.

## Requirements

The reconciliation process must evaluate and compare:

- **Target vs Realized Weights**: Any absolute drift between the strategy's target weights and the actual position weights after trading.
- **Order Statuses**: Compare open orders, filled orders, and any unfilled, cancelled, or rejected orders.
- **Account State**: Verify that final cash and equity are correctly accounted for and match expected levels based on order values.
- **Discrepancy Flags**: Raise alerts if:
  - Any target asset is missing from the final portfolio.
  - Any non-target asset is present in the final portfolio (unexpected holding).
  - The absolute weight drift of any symbol exceeds a tolerance threshold (e.g., 2.0%).
  - Any order failed or was rejected.

## Reconciliation Report Schema (`logs/reconciliation_YYYY-MM-DD.json`)

The output of the reconciliation process will be a structured JSON file keyed by account name:

```json
{
  "date": "2026-05-29",
  "accounts": {
    "FinRL": {
      "reconciled_successfully": true,
      "discrepancies_found": false,
      "alerts": [],
      "target_vs_actual_weights": [
        {
          "symbol": "ORCL",
          "target_weight": 0.3333,
          "actual_weight": 0.3312,
          "drift": 0.0021
        },
        { ... }
      ],
      "orders_summary": {
        "submitted": 5,
        "filled": 5,
        "open": 0,
        "failed_or_rejected": 0
      },
      "cash": 12500.42,
      "equity": 1139861.42
    },
    "AR": { ... }
  }
}
```

## Implementation Plan

1. **Reconciliation Function**: Implement `reconcile_post_trade(run_date, account_name, record)` inside `run_paper_trading.py`. This function will load the compiled decision record for the account, compare target weights and actual positions, check order statuses, and flag any discrepancies.
2. **Combine and Save**: Implement `save_reconciliation_report(run_date, account_name, recon_result)` to maintain a single consolidated reconciliation report `logs/reconciliation_YYYY-MM-DD.json` containing results for all accounts run on that date.
3. **Trigger Post-Trade**: Run the reconciliation process at the very end of `run_account` in `run_paper_trading.py` (after rebalancing and fetching post-trade positions).
4. **Unit Testing**: Add focused tests under `tests/test_weekly_workflow.py` to verify that the reconciliation checks accurately identify weight drift, unexpected holdings, missing assets, and failed orders.
