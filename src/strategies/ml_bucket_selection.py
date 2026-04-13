#!/usr/bin/env python3
"""Per-bucket ML stock selection: train on all tickers <=2025, predict Q1 2026 (or latest quarter with data)."""

import argparse
import os
import sqlite3
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Allow running as standalone script: python3 src/strategies/ml_bucket_selection.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Sector -> Bucket mapping (synced with group_selection_by_gics.py v1.2.2)
# ---------------------------------------------------------------------------
SECTOR_TO_BUCKET = {
    "information technology": "growth_tech",
    "technology": "growth_tech",
    "communication services": "growth_tech",
    "consumer discretionary": "cyclical",
    "consumer cyclical": "cyclical",
    "financials": "cyclical",
    "financial services": "cyclical",
    "industrials": "cyclical",
    "energy": "real_assets",
    "materials": "real_assets",
    "basic materials": "real_assets",
    "real estate": "real_assets",
    "health care": "defensive",
    "healthcare": "defensive",
    "consumer staples": "defensive",
    "consumer defensive": "defensive",
    "utilities": "defensive",
}

FEATURE_COLS = [
    # Original 14
    "EPS", "BPS", "DPS", "cur_ratio", "quick_ratio", "cash_ratio",
    "acc_rec_turnover", "debt_ratio", "debt_to_equity",
    "pe", "ps", "pb", "roe", "net_income_ratio",
    # Profitability
    "gross_margin", "operating_margin", "ebitda_margin", "pretax_margin",
    "effective_tax_rate", "ebt_per_ebit",
    # Efficiency
    "asset_turnover", "fixed_asset_turnover", "inventory_turnover",
    "payables_turnover", "wc_turnover",
    # Leverage
    "debt_to_assets", "debt_to_capital", "lt_debt_to_capital",
    "interest_coverage", "debt_service_coverage", "debt_to_mktcap",
    # Cash Flow
    "fcf_per_share", "ocf_per_share", "cash_per_share", "capex_per_share",
    "fcf_to_ocf", "ocf_ratio", "ocf_to_sales", "ocf_coverage",
    "st_ocf_coverage", "capex_coverage",
    # Per-Share
    "revenue_per_share", "tangible_bvps", "interest_debt_per_share",
    # Valuation
    "peg", "price_to_fcf", "price_to_ocf", "ev_multiple",
    # Dividend
    "dividend_payout", "dividend_yield", "div_capex_coverage",
    # Solvency
    "solvency_ratio",
]


def build_models():
    models = {
        "RF": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "XGB": None,
        "LGBM": None,
        "HistGBM": HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "Ridge": Ridge(alpha=1.0),
    }
    try:
        from xgboost import XGBRegressor
        models["XGB"] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    except ImportError:
        del models["XGB"]

    try:
        from lightgbm import LGBMRegressor
        models["LGBM"] = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
    except ImportError:
        del models["LGBM"]

    return models


def run_bucket(bucket, bdf, feature_cols, val_cutoff="2025-12-31"):
    """Train models for one bucket, return (predictions_df, model_results_list)."""

    train_b = bdf[(bdf["datadate"] < val_cutoff) & (bdf["y_return"].notna())]
    val_b = bdf[(bdf["datadate"] == val_cutoff) & (bdf["y_return"].notna())]
    # Infer on the latest quarter after val_cutoff
    infer_dates = sorted(bdf[bdf["datadate"] > val_cutoff]["datadate"].unique())
    if infer_dates:
        infer_date = infer_dates[-1]
        infer_b = bdf[bdf["datadate"] == infer_date]
    else:
        infer_b = pd.DataFrame()

    print(f"\n{'=' * 60}")
    print(f"  Bucket: {bucket.upper()}")
    print(f"  Train: {len(train_b)} | Val: {len(val_b)} | Infer: {len(infer_b)}")
    if len(infer_b) > 0:
        print(f"  Infer date: {infer_date}")
    print(f"{'=' * 60}")

    if len(train_b) < 20 or len(infer_b) == 0:
        print("  SKIP: insufficient data")
        return pd.DataFrame(), [], []

    X_train, y_train = train_b[feature_cols].values, train_b["y_return"].values
    X_val, y_val = val_b[feature_cols].values, val_b["y_return"].values
    X_infer = infer_b[feature_cols].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if len(val_b) > 0 else None
    X_infer_s = scaler.transform(X_infer)

    models = build_models()
    fitted = {}
    model_results = []
    best_name, best_mse, best_model = None, float("inf"), None

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        fitted[name] = model
        if X_val_s is not None and len(X_val_s) > 0:
            mse = mean_squared_error(y_val, model.predict(X_val_s))
        else:
            mse = float("inf")
        model_results.append({
            "bucket": bucket, "model": name, "val_mse": round(mse, 6),
            "train_size": len(train_b), "val_size": len(val_b), "infer_size": len(infer_b),
        })
        print(f"  {name:12s}: MSE = {mse:.6f}")
        if mse < best_mse:
            best_name, best_mse, best_model = name, mse, model

    # Stacking top 3
    if X_val_s is not None and len(X_val_s) > 0:
        sorted_m = sorted(
            [(n, mean_squared_error(y_val, fitted[n].predict(X_val_s))) for n in fitted],
            key=lambda x: x[1],
        )
    else:
        sorted_m = [(n, 0) for n in fitted]
    top3 = [n for n, _ in sorted_m[:3]]
    stacking = StackingRegressor(
        estimators=[(n, fitted[n]) for n in top3],
        final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1,
    )
    stacking.fit(X_train_s, y_train)
    fitted["Stacking"] = stacking
    if X_val_s is not None and len(X_val_s) > 0:
        stack_mse = mean_squared_error(y_val, stacking.predict(X_val_s))
    else:
        stack_mse = float("inf")
    model_results.append({
        "bucket": bucket, "model": "Stacking", "val_mse": round(stack_mse, 6),
        "train_size": len(train_b), "val_size": len(val_b), "infer_size": len(infer_b),
    })
    print(f"  {'Stacking':12s}: MSE = {stack_mse:.6f}  (base: {top3})")

    if stack_mse < best_mse:
        best_name, best_mse, best_model = "Stacking", stack_mse, stacking

    print(f"  >> Best: {best_name} (MSE={best_mse:.6f})")

    # Predict
    infer_b = infer_b.copy()
    infer_b["predicted_return"] = best_model.predict(X_infer_s)
    infer_b["best_model"] = best_name
    for n, m in fitted.items():
        infer_b[f"pred_{n}"] = m.predict(X_infer_s)

    # Inverse-MSE weighted ensemble
    mse_map = {r["model"]: r["val_mse"] for r in model_results}
    pred_model_cols = [c for c in infer_b.columns if c.startswith("pred_") and c != "pred_ensemble_avg"]
    weights = {}
    for col in pred_model_cols:
        name = col.replace("pred_", "")
        mse = mse_map.get(name, None)
        weights[col] = (1.0 / mse) if mse and mse > 0 else 0
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    infer_b["pred_ensemble_avg"] = sum(infer_b[col] * w for col, w in weights.items())

    infer_b = infer_b.sort_values("predicted_return", ascending=False)

    # Print ranking
    print(f"\n  Ranking:")
    for i, (_, r) in enumerate(infer_b.iterrows()):
        marker = " ***" if i < 3 else ""
        print(f"    {i + 1:2d}. {r['tic']:6s}  {r['predicted_return'] * 100:+6.1f}%{marker}")

    # Feature importance (collect from all models that expose it)
    importance_records = []
    for name, model in fitted.items():
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            for rank_idx, (feat, val) in enumerate(imp.items(), 1):
                importance_records.append({
                    "bucket": bucket, "model": name,
                    "is_best": name == best_name,
                    "feature": feat, "importance": round(val, 6),
                    "rank": rank_idx,
                })
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            imp = pd.Series(coefs, index=feature_cols).sort_values(ascending=False)
            total = imp.sum()
            for rank_idx, (feat, val) in enumerate(imp.items(), 1):
                importance_records.append({
                    "bucket": bucket, "model": name,
                    "is_best": name == best_name,
                    "feature": feat, "importance": round(val / total if total > 0 else 0, 6),
                    "rank": rank_idx,
                })

    # Print top 5 for best model
    best_imp = [r for r in importance_records if r["model"] == best_name]
    if best_imp:
        best_imp_sorted = sorted(best_imp, key=lambda x: x["importance"], reverse=True)
        print(f"\n  Top 5 Features ({best_name}):")
        for r in best_imp_sorted[:5]:
            print(f"    {r['feature']:20s} {r['importance']:.3f}")

    return infer_b, model_results, importance_records


def main():
    parser = argparse.ArgumentParser(description="Per-bucket ML stock selection")
    parser.add_argument("--db", default=os.path.join(project_root, "data", "finrl_trading.db"))
    parser.add_argument("--val-cutoff", default="2025-12-31", help="Validation cutoff date (train < this, val = this)")
    parser.add_argument("--output-dir", default=os.path.join(project_root, "data"))
    args = parser.parse_args()

    # Load data
    conn = sqlite3.connect(args.db)
    # Build column list dynamically from FEATURE_COLS
    _feat_sql = ", ".join(FEATURE_COLS)
    df = pd.read_sql(
        f"""SELECT ticker as tic, datadate, gsector, adj_close_q,
           {_feat_sql}, y_return
           FROM fundamental_data ORDER BY ticker, datadate""",
        conn,
    )
    conn.close()

    print(f"Loaded {len(df)} records, {df['tic'].nunique()} tickers")
    print(f"Date range: {df['datadate'].min()} ~ {df['datadate'].max()}")
    print(f"Val cutoff: {args.val_cutoff}")

    # Prep features
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    global_medians = df[FEATURE_COLS].median()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(global_medians).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Winsorize: clip at 1st/99th percentile to reduce outlier impact
    for c in FEATURE_COLS:
        p01, p99 = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(lower=p01, upper=p99)

    # Assign buckets
    df["bucket"] = df["gsector"].str.lower().map(SECTOR_TO_BUCKET)
    unmapped = df[df["bucket"].isna()]["gsector"].unique()
    if len(unmapped) > 0:
        print(f"WARNING: unmapped sectors: {unmapped}")
    df = df[df["bucket"].notna()].copy()

    # Run per bucket
    all_preds = []
    all_model_results = []
    all_importances = []

    for bucket in ["growth_tech", "cyclical", "real_assets", "defensive"]:
        bdf = df[df["bucket"] == bucket].copy()
        preds, results, importances = run_bucket(bucket, bdf, FEATURE_COLS, val_cutoff=args.val_cutoff)
        if len(preds) > 0:
            all_preds.append(preds)
        all_model_results.extend(results)
        all_importances.extend(importances)

    if not all_preds:
        print("\nNo predictions generated.")
        return

    pred_all = pd.concat(all_preds, ignore_index=True)

    # Per-bucket ranking
    pred_all["rank_best"] = pred_all.groupby("bucket")["predicted_return"].rank(ascending=False).astype(int)
    pred_all["rank_ensemble"] = pred_all.groupby("bucket")["pred_ensemble_avg"].rank(ascending=False).astype(int)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "ml_bucket_predictions.csv")
    pred_all.to_csv(pred_path, index=False)
    print(f"\nSaved: {pred_path} ({len(pred_all)} stocks)")

    model_path = os.path.join(args.output_dir, "ml_bucket_model_results.csv")
    pd.DataFrame(all_model_results).to_csv(model_path, index=False)

    if all_importances:
        imp_df = pd.DataFrame(all_importances)
        imp_df = imp_df.sort_values(["bucket", "model", "rank"])
        imp_path = os.path.join(args.output_dir, "ml_feature_importance.csv")
        imp_df.to_csv(imp_path, index=False)
        print(f"Saved: {imp_path} ({len(imp_df)} rows)")
    print(f"Saved: {model_path} ({len(all_model_results)} rows)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: Top picks per bucket")
    print(f"{'=' * 60}")
    for bucket in ["growth_tech", "cyclical", "real_assets", "defensive"]:
        bp = pred_all[pred_all["bucket"] == bucket].sort_values("rank_best").head(3)
        if len(bp) == 0:
            print(f"  {bucket:15s}: no data")
            continue
        best_m = bp.iloc[0]["best_model"]
        picks = ", ".join(
            f"{r['tic']}({r['predicted_return'] * 100:+.1f}%)" for _, r in bp.iterrows()
        )
        print(f"  {bucket:15s} [{best_m}]: {picks}")


if __name__ == "__main__":
    main()
