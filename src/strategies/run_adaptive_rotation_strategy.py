"""
Adaptive Rotation Strategy Runner
===================================

Command-line tool for running the Adaptive Multi-Asset Rotation Strategy

Usage:
    # Run for a single date
    python run_adaptive_rotation_strategy.py --config path/to/config.yaml --date 2024-02-01
    
    # Run backtest
    python run_adaptive_rotation_strategy.py --backtest --start 2023-01-01 --end 2024-12-31
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.adaptive_rotation import AdaptiveRotationEngine
from src.strategies.adaptive_rotation.data_preprocessor import DataPreprocessor


def run_single_date(config_path: str, as_of_date: str, data_dir: str = None):
    """
    Run strategy for a single date
    
    Args:
        config_path: Path to configuration file
        as_of_date: Decision date (format: YYYY-MM-DD)
        data_dir: Optional data directory path
    """
    print(f"\n{'='*60}")
    print(f"Running Adaptive Rotation Strategy")
    print(f"Config: {config_path}")
    print(f"Date: {as_of_date}")
    print(f"{'='*60}\n")
    
    # 1. Load data first
    print("1. Loading and preprocessing data...")
    from src.strategies.adaptive_rotation.config_loader import load_config
    config = load_config(config_path)
    preprocessor = DataPreprocessor(config)
    preprocessor.load_and_prepare(data_dir=data_dir)
    
    # 2. Initialize engine with preprocessor
    print("2. Initializing strategy engine...")
    engine = AdaptiveRotationEngine(config=config_path, data_preprocessor=preprocessor)
    config = engine.get_config()
    
    # Get data as of the decision date
    price_data = preprocessor.get_as_of_date(as_of_date)
    
    print(f"   - Loaded {len(price_data)} assets")
    print(f"   - Data range: {min(s.index.min() for s in price_data.values())} to {as_of_date}")
    
    # 3. Run strategy
    print(f"\n3. Running strategy to generate target weights...")
    weights, audit_log = engine.run(
        price_data=price_data,
        as_of_date=as_of_date
    )
    
    # 4. Display results
    print(f"\n{'='*60}")
    print("Strategy Output")
    print(f"{'='*60}")
    
    print(f"\nMarket Regime: {weights.regime_state}")
    print(f"Total Invested: {weights.get_invested_weight():.2%}")
    print(f"Cash Position: {weights.cash_weight:.2%}")
    
    print(f"\nTarget Portfolio ({len(weights.weights)} assets):")
    print("-" * 40)
    for symbol, weight in sorted(weights.weights.items(), 
                                  key=lambda x: x[1], reverse=True):
        print(f"  {symbol:8s}: {weight:7.2%}")
    
    # 5. Save audit log (using config path)
    output_dir = Path(config.paths.audit_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audit_file = output_dir / f"audit_{as_of_date}.json"
    audit_log.to_json(str(audit_file))
    print(f"\nAudit log saved to: {audit_file}")
    
    return weights, audit_log


def run_backtest(config_path: str, start_date: str, end_date: str, 
                 data_dir: str = None, freq: str = "W-FRI", 
                 daily_fast_track: bool = True):
    """
    Run backtest over a date range
    
    Args:
        config_path: Path to configuration file
        start_date: Backtest start date (format: YYYY-MM-DD)
        end_date: Backtest end date (format: YYYY-MM-DD)
        data_dir: Optional data directory path
        freq: Regular rebalance frequency (default: W-FRI)
        daily_fast_track: Enable daily Fast Risk-Off monitoring (default: True)
    """
    print(f"\n{'='*60}")
    print(f"Running Backtest")
    print(f"Config: {config_path}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Rebalance: {freq}")
    print(f"Daily Fast Track: {'Enabled' if daily_fast_track else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # 1. Load data first
    print("1. Loading and preprocessing data...")
    from src.strategies.adaptive_rotation.config_loader import load_config
    config = load_config(config_path)
    preprocessor = DataPreprocessor(config)
    preprocessor.load_and_prepare(data_dir=data_dir)
    
    # 2. Initialize engine with preprocessor
    print("2. Initializing strategy engine...")
    engine = AdaptiveRotationEngine(config=config_path, data_preprocessor=preprocessor)
    config = engine.get_config()
    
    # 3. Generate decision dates
    weekly_dates = pd.date_range(start_date, end_date, freq=freq)
    
    if daily_fast_track:
        # Get all trading days for daily monitoring (Fast Risk-Off + Stop-Loss)
        daily_dates = preprocessor.daily_data['^GSPC'].index
        daily_dates = daily_dates[(daily_dates >= pd.Timestamp(start_date)) & 
                                  (daily_dates <= pd.Timestamp(end_date))]
        print(f"\n3. Running backtest with Daily Monitoring:")
        print(f"   - Weekly rebalance points: {len(weekly_dates)}")
        print(f"   - Daily monitoring: {len(daily_dates)} days")
        print(f"   - Daily checks: Fast Risk-Off + Stop-Loss")
        decision_dates = daily_dates  # Monitor every day
    else:
        decision_dates = weekly_dates
        print(f"\n3. Running {len(decision_dates)} weekly decision points...")
    
    results = []
    current_positions = {}
    fast_track_active_until = None
    
    # Statistics tracking (DDS: only Weekly is rebalance, others are adjustments)
    fast_track_adjustment_count = 0
    stop_loss_adjustment_count = 0
    
    for i, date in enumerate(decision_dates, 1):
        try:
            is_weekly = date in weekly_dates.to_list()
            
            # Get weekly price data for strategy
            price_data = preprocessor.get_data_as_of(date)
            price_data = {symbol: df['close'] for symbol, df in price_data.items()}
            
            # Get daily prices for Fast Risk-Off and Stop-Loss checks
            daily_prices = preprocessor.get_daily_data_as_of(date, symbols=['^GSPC', '^VIX', 'QQQ'])
            
            # === Determine Action Type (Priority: Weekly > Fast Track > Stop-Loss) ===
            # DDS: Only Weekly triggers full Rebalance, others are risk adjustments
            
            decision_type = None
            trigger_reasons = []
            
            # Priority 1: Weekly Rebalance (overrides everything else)
            if is_weekly:
                decision_type = 'weekly'
                trigger_reasons.append('weekly_rebalance')
            
            # Priority 2: Fast Risk-Off (only if not weekly)
            elif daily_fast_track:
                from src.strategies.adaptive_rotation.market_regime import check_fast_risk_off_trigger
                
                should_trigger_fast, signals = check_fast_risk_off_trigger(
                    daily_prices.get('^GSPC'),
                    daily_prices.get('QQQ'),
                    daily_prices.get('^VIX'),
                    date,
                    config
                )
                
                if should_trigger_fast and (fast_track_active_until is None or date > fast_track_active_until):
                    decision_type = 'fast_track_trigger'
                    fast_track_active_until = date + pd.Timedelta(days=config.fast_risk_off.behavior.duration_days)
                    trigger_reasons.append(f"fast_risk_off(Price={signals['price_shock']},Vol={signals['volatility_shock']})")
                    
                    print(f"   [FAST TRACK TRIGGERED] {date.strftime('%Y-%m-%d')}: "
                          f"Price={signals['price_shock']}, Vol={signals['volatility_shock']}")
            
            # Priority 3: Stop-Loss (only if not weekly and no fast track)
            if decision_type != 'weekly' and decision_type != 'fast_track_trigger' and current_positions:
                # Get current prices for all positions
                current_prices = {}
                for symbol in current_positions.keys():
                    daily_symbol_data = preprocessor.get_daily_data_as_of(date, symbols=[symbol])
                    if symbol in daily_symbol_data:
                        symbol_prices = daily_symbol_data[symbol]
                        if len(symbol_prices) > 0:
                            current_prices[symbol] = float(symbol_prices.iloc[-1])
                
                # Check stops
                risk_result = engine.risk_manager.check_stops(
                    current_positions,
                    current_prices,
                    date
                )
                
                if len(risk_result.triggered_stops) > 0:
                    decision_type = 'stop_loss_trigger'
                    trigger_reasons.append(f"stop_loss({len(risk_result.triggered_stops)}_positions)")
                    
                    stopped_symbols = [s.symbol for s in risk_result.triggered_stops]
                    print(f"   [STOP-LOSS TRIGGERED] {date.strftime('%Y-%m-%d')}: {', '.join(stopped_symbols)}")
                    
                    # Update positions after stops
                    current_positions = risk_result.updated_positions
            
            # === Execute Actions Based on Decision Type ===
            # DDS Design: Only Weekly triggers full Rebalance
            # Fast Risk-Off and Stop-Loss only adjust existing positions
            
            if decision_type == 'weekly':
                # === WEEKLY REBALANCE: Full Strategy Execution ===
                # Complete Group Selection + Asset Ranking + Portfolio Construction
                weights, audit_log = engine.run(
                    price_data=price_data,
                    as_of_date=date,
                    current_positions=current_positions,
                    mode='backtest'
                )
                
                # Update current positions based on new weights
                current_positions = {}
                for symbol, weight in weights.weights.items():
                    if weight > 0 and symbol in price_data:
                        price_series = price_data[symbol]
                        current_price = float(price_series.iloc[-1])
                        
                        from src.strategies.adaptive_rotation.risk_manager import PositionState
                        current_positions[symbol] = PositionState(
                            symbol=symbol,
                            entry_date=date,
                            entry_price=current_price,
                            peak_price=current_price,
                            peak_date=date
                        )
                
                results.append({
                    'date': date,
                    'invested': weights.get_invested_weight(),
                    'cash': weights.cash_weight,
                    'regime': weights.regime_state,
                    'num_assets': len(weights.weights),
                    'weights': weights,
                    'audit': audit_log,
                    'decision_type': decision_type,
                    'trigger_reasons': trigger_reasons
                })
            
            elif decision_type == 'fast_track_trigger':
                # === FAST RISK-OFF: Adjust Existing Positions (No Rebalance) ===
                # DDS: "No signal recomputation, no re-ranking"
                # Only tighten risk budget: group_cap=0.3, cash_floor=0.5
                
                # Calculate theoretical exposure scaling (informational)
                fast_config = config.fast_risk_off
                target_invested = 1.0 - fast_config.behavior.cash_floor  # 0.5
                
                # In a real implementation, this would reduce position sizes
                # For backtest, we just log the event and update position tracking
                
                # Update position tracking (peak prices for trailing stops)
                for symbol in list(current_positions.keys()):
                    daily_symbol_data = preprocessor.get_daily_data_as_of(date, symbols=[symbol])
                    if symbol in daily_symbol_data and len(daily_symbol_data[symbol]) > 0:
                        current_price = float(daily_symbol_data[symbol].iloc[-1])
                        current_positions[symbol].update_peak(current_price, date)
                
                # Log as adjustment, not full rebalance
                print(f"   [FAST TRACK ADJUSTMENT] Would reduce exposure to {target_invested:.1%} (event logged)")
                fast_track_adjustment_count += 1
                
                # Note: No audit log generated (this is not a full strategy decision)
                # No results.append (no rebalance occurred)
                # Position sizes would be reduced in live trading
            
            elif decision_type == 'stop_loss_trigger':
                # === STOP-LOSS: Remove Triggered Positions Only (No Rebalance) ===
                # Clear stopped positions, activate cooldown
                # Do NOT call engine.run(), do NOT re-select assets
                
                stopped_symbols = [s.symbol for s in risk_result.triggered_stops]
                
                # Log stop-loss exits with cooldown info
                cooldown_days = config.cooldown.after_stop_days
                print(f"   [STOP-LOSS ADJUSTMENT] Exited {len(stopped_symbols)} positions: "
                      f"{', '.join(stopped_symbols)}")
                print(f"      Cooldown activated: {cooldown_days} days (blocks re-entry)")
                stop_loss_adjustment_count += 1
                
                # current_positions already updated by risk_result.updated_positions
                # Note: No audit log, no results.append (this is risk management, not rebalance)
                # Released capital stays as cash, not reallocated
            
            # === No Action Needed: Just Update Peak Tracking ===
            if decision_type is None:
                # No weekly rebalance, no risk triggers
                # Update position peaks for next day's stop-loss check
                if current_positions:
                    for symbol in list(current_positions.keys()):
                        daily_symbol_data = preprocessor.get_daily_data_as_of(date, symbols=[symbol])
                        if symbol in daily_symbol_data:
                            symbol_prices = daily_symbol_data[symbol]
                            if len(symbol_prices) > 0:
                                current_price = float(symbol_prices.iloc[-1])
                                current_positions[symbol].update_peak(current_price, date)
            
            # Progress indicator
            if i % 100 == 0 or (decision_type in ['fast_track_trigger', 'stop_loss_trigger']):
                print(f"   Progress: {i}/{len(decision_dates)} days scanned, "
                      f"{len(results)} weekly rebalances, "
                      f"{fast_track_adjustment_count} fast track adj, "
                      f"{stop_loss_adjustment_count} stop-loss adj")
        
        except Exception as e:
            print(f"   Warning: {date.strftime('%Y-%m-%d')} failed - {str(e)}")
            continue
    
    # 4. Summarize results
    print(f"\n{'='*60}")
    print("Backtest Summary")
    print(f"{'='*60}")
    
    # Count different event types
    weekly_count = sum(1 for r in results if r.get('decision_type') == 'weekly')
    
    if daily_fast_track:
        print(f"\nDecision Points (Daily Monitoring Enabled):")
        print(f"  Days monitored: {len(decision_dates)}")
        print(f"  Weekly rebalances: {weekly_count} (Full strategy execution)")
        print(f"\nDaily Risk Adjustments (No Rebalance):")
        print(f"  Fast Track adjustments: {fast_track_adjustment_count} (Reduce exposure)")
        print(f"  Stop-Loss exits: {stop_loss_adjustment_count} (Clear positions)")
    else:
        print(f"\nSuccessful runs: {len(results)} / {len(decision_dates)} decision points")
    
    if results:
        avg_invested = sum(r['invested'] for r in results) / len(results)
        avg_num_assets = sum(r['num_assets'] for r in results) / len(results)
        
        print(f"\nAverage invested: {avg_invested:.2%}")
        print(f"Average # of assets: {avg_num_assets:.1f}")
        
        # Regime distribution
        regime_counts = {}
        for r in results:
            regime = r['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        print(f"\nRegime Distribution:")
        for regime, count in sorted(regime_counts.items()):
            print(f"  {regime:15s}: {count:4d} ({count/len(results)*100:.1f}%)")
        
        # Note: Fast Track and Stop-Loss events are logged inline during execution
        # They are not recorded in results[] as they don't trigger full rebalances
        
        # 5. Save results
        # Save summary CSV (using config weights_dir)
        weights_dir = Path(config.paths.weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        summary_df = pd.DataFrame([{
            'date': r['date'],
            'invested': r['invested'],
            'cash': r['cash'],
            'regime': r['regime'],
            'num_assets': r['num_assets']
        } for r in results])
        
        summary_file = weights_dir / f"backtest_{start_date}_to_{end_date}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nBacktest summary saved to: {summary_file}")
        
        # Save detailed portfolio weights (new feature)
        weights_detail_df = engine.export_weights_to_dataframe(results)
        weights_detail_file = weights_dir / f"ars_portfolio_weights_{start_date}_to_{end_date}.csv"
        weights_detail_df.to_csv(weights_detail_file, index=False)
        print(f"Detailed portfolio weights saved to: {weights_detail_file}")
        
        # Save audit logs (using config audit_dir)
        audit_dir = Path(config.paths.audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        for r in results:
            audit_file = audit_dir / f"audit_{r['date'].strftime('%Y-%m-%d')}.json"
            r['audit'].to_json(str(audit_file))
        
        print(f"Audit logs saved to: {audit_dir}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Adaptive Multi-Asset Rotation Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for a single date
  python run_adaptive_rotation_strategy.py \\
      --config src/strategies/AdaptiveRotationConf_v1.2.1.yaml \\
      --date 2024-02-01

  # Run backtest
  python run_adaptive_rotation_strategy.py \\
      --config src/strategies/AdaptiveRotationConf_v1.2.1.yaml \\
      --backtest \\
      --start 2023-01-01 \\
      --end 2024-12-31
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='src/strategies/AdaptiveRotationConf_v1.2.1.yaml',
        help='Path to configuration file (default: src/strategies/AdaptiveRotationConf_v1.2.1.yaml)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Data directory path (default: use path from config file)'
    )
    
    # Single run mode
    parser.add_argument(
        '--date',
        type=str,
        help='Run for single date (format: YYYY-MM-DD)'
    )
    
    # Backtest mode
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run in backtest mode'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Backtest start date (format: YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='Backtest end date (format: YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--freq',
        type=str,
        default='W-FRI',
        help='Rebalance frequency (default: W-FRI for weekly on Friday)'
    )
    
    parser.add_argument(
        '--no-daily-fast-track',
        action='store_true',
        help='Disable daily Fast Risk-Off monitoring (default: enabled)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.backtest:
            # Backtest mode
            if not args.start or not args.end:
                parser.error("Backtest mode requires --start and --end arguments")
            
            run_backtest(
                config_path=args.config,
                start_date=args.start,
                end_date=args.end,
                data_dir=args.data_dir,
                freq=args.freq,
                daily_fast_track=not args.no_daily_fast_track
            )
        
        elif args.date:
            # Single run mode
            run_single_date(
                config_path=args.config,
                as_of_date=args.date,
                data_dir=args.data_dir
            )
        
        else:
            parser.error("Please specify either --date or --backtest mode")
    
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()