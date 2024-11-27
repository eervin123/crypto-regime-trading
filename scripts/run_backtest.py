"""
Backtest runner for optimized strategies.
Uses the same data pipeline and parameter structure as optimization.
"""

from pathlib import Path
import json
import pandas as pd
import vectorbtpro as vbt
import yaml
import numpy as np
from regimes_multi_strat_pf import calculate_regimes_nb
from optuna_multistrat import (
    run_ma_strategy_with_stops as run_ma_strategy,
    run_macd_divergence_strategy_with_stops as run_macd_strategy,
    run_rsi_divergence_strategy_with_stops as run_rsi_strategy,
    run_bbands_strategy_with_stops as run_bbands_strategy,
    run_psar_strategy_with_stops as run_psar_strategy,
    run_rsi_mean_reversion_strategy,
    mean_reversion_strategy
)

# Define required parameters for each strategy - let's verify these match optuna_multistrat.py
STRATEGY_PARAMS = {
    'Moving Average': {
        'function': run_ma_strategy,
        'required_params': ['fast_window', 'slow_window', 'direction', 'use_sl_tp', 'atr_window', 'atr_multiplier']  # Updated to match optuna version
    },
    'MACD Divergence': {
        'function': run_macd_strategy,
        'required_params': ['fast_window', 'slow_window', 'signal_window', 'direction', 'use_sl_tp', 'atr_window', 'atr_multiplier']
    },
    'RSI Divergence': {
        'function': run_rsi_strategy,
        'required_params': ['rsi_window', 'rsi_threshold', 'lookback_window', 'direction', 'use_sl_tp', 'atr_window', 'atr_multiplier']
    },
    'Bollinger Bands': {
        'function': run_bbands_strategy,
        'required_params': ['bb_window', 'bb_alpha', 'direction', 'use_sl_tp', 'atr_window', 'atr_multiplier']
    },
    'Parabolic SAR': {
        'function': run_psar_strategy,
        'required_params': ['direction', 'use_sl_tp', 'atr_window', 'atr_multiplier', 'af0', 'af_increment', 'max_af']
    },
    'RSI Mean Reversion': {
        'function': run_rsi_mean_reversion_strategy,
        'required_params': ['direction', 'use_sl_tp', 'atr_window', 'atr_multiplier', 'rsi_window', 'rsi_lower', 'rsi_upper']
    },
    'Mean Reversion': {
        'function': mean_reversion_strategy,
        'required_params': ['direction', 'bb_window', 'bb_alpha', 'timeframe_1', 'timeframe_2', 'atr_window', 'atr_multiplier']
    }
}

def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/optuna_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(base_timeframe="1T", analysis_timeframe="30T", regime_timeframe="1D"):
    """Load and prepare data at different timeframes."""
    data = vbt.BinanceData.load("data/m1_data.pkl")

    # Analysis timeframe data
    data_analysis = {
        "BTC": data.resample(analysis_timeframe).data["BTCUSDT"],
        "ETH": data.resample(analysis_timeframe).data["ETHUSDT"]
    }

    # Regime timeframe data with returns
    data_regime = {
        "BTC": data.resample(regime_timeframe).data["BTCUSDT"],
        "ETH": data.resample(regime_timeframe).data["ETHUSDT"]
    }
    
    # Add returns for regime calculation
    for symbol in ["BTC", "ETH"]:
        data_regime[symbol]["Return"] = data_regime[symbol]["Close"].pct_change()

    return data_analysis, data_regime

def calculate_regimes(data_regime, data_analysis, analysis_timeframe):
    """Calculate and align market regimes."""
    RegimeIndicator = vbt.IndicatorFactory(
        class_name="RegimeIndicator",
        input_names=["price", "returns"],
        param_names=["ma_short_window", "ma_long_window", "vol_short_window", "avg_vol_window"],
        output_names=["regimes"]
    ).with_apply_func(calculate_regimes_nb)

    aligned_regime_data = {}
    
    for symbol in ["BTC", "ETH"]:
        regime_indicator = RegimeIndicator.run(
            data_regime[symbol]["Close"],
            data_regime[symbol]["Return"],
            ma_short_window=21,
            ma_long_window=88,
            vol_short_window=21,
            avg_vol_window=365,
        )

        data_regime[symbol]["Market Regime"] = regime_indicator.regimes.values
        regime_data = data_regime[symbol]["Market Regime"]
        analysis_regime_data = regime_data.resample(analysis_timeframe).ffill()
        
        aligned_regime_data[symbol] = analysis_regime_data.reindex(
            data_analysis[symbol].index, method="ffill"
        )

    return aligned_regime_data

def run_full_backtest(symbol, timeframe, target_regimes):
    """Run backtest using optimal parameters."""
    # Load config and setup data pipeline
    config = load_config()
    base_tf = config["timeframes"]["base"]
    analysis_tf = config["timeframes"]["analysis"]
    regime_tf = config["timeframes"]["regime"]
    
    # Load and preprocess data
    data_analysis, data_regime = load_data(
        base_timeframe=base_tf,
        analysis_timeframe=analysis_tf,
        regime_timeframe=regime_tf
    )
    
    # Calculate regimes
    aligned_regime_data = calculate_regimes(
        data_regime, 
        data_analysis,
        analysis_timeframe=analysis_tf
    )
    
    results = []
    equity_curves = {}
    
    # Run each strategy
    for strategy_name, strategy_info in STRATEGY_PARAMS.items():
        try:
            # Load strategy parameters
            strategy_id = strategy_name.lower().replace(' ', '_')
            param_file = Path("results/params") / f"{symbol.lower()}_{strategy_id}_params_regimes_{'_'.join(map(str, target_regimes))}_tf_{timeframe}.json"
            
            print(f"\nLoading parameters for {strategy_name} from: {param_file}")
            
            if not param_file.exists():
                print(f"Parameter file not found for {strategy_name}, skipping...")
                continue
                
            with open(param_file, 'r') as f:
                params_data = json.load(f)
            
            # Try each direction
            for direction in ['long', 'short']:
                if params_data[direction]:
                    strategy_params = params_data[direction]
                    print(f"Loading {strategy_name} {direction} params: {strategy_params}")
                    
                    # Extract only the relevant parameters for this strategy
                    required_params = strategy_info['required_params']
                    filtered_params = {
                        k: int(v) if isinstance(v, (float, np.float64)) and ('window' in k or k.endswith('_ma'))
                        else float(v) if isinstance(v, (float, np.float64))
                        else v
                        for k, v in strategy_params.items()
                        if k in required_params or k == 'direction'
                    }
                    
                    # Add default fees if not present
                    if 'fees' in required_params and 'fees' not in filtered_params:
                        filtered_params['fees'] = 0.001
                    
                    print(f"Running {strategy_name} with params: {filtered_params}")
                    
                    pf = strategy_info['function'](
                        symbol_ohlcv_df=data_analysis[symbol],
                        regime_data=aligned_regime_data[symbol],
                        allowed_regimes=target_regimes,
                        **filtered_params
                    )
                    
                    # Store results
                    equity_curves[f"{symbol}_{strategy_name}_{direction}"] = pf.value
                    results.append({
                        'Symbol': symbol,
                        'Strategy': strategy_name,
                        'Direction': direction,
                        'Total Return': pf.total_return,
                        'Sharpe Ratio': pf.sharpe_ratio,
                        'Max Drawdown': pf.max_drawdown,
                        'Total Trades': pf.trades.count(),
                        'Win Rate': pf.trades.win_rate,
                        'Profit Factor': pf.trades.profit_factor
                    })
                    
        except Exception as e:
            print(f"Error running {strategy_name}: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No strategies produced results")
        
    return pd.DataFrame(results), equity_curves

def ensure_results_dir():
    """Create results directory and subdirectories if they don't exist."""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create backtests subdirectory
    backtests_dir = results_dir / "backtests"
    backtests_dir.mkdir(exist_ok=True)
    
    return backtests_dir

if __name__ == "__main__":
    # Load config for defaults
    config = load_config()
    timeframe = config["timeframes"]["analysis"].replace("T", "m")
    target_regimes = config["regime"]["target_regimes"]
    
    # Ensure results directory exists
    backtests_dir = ensure_results_dir()
    
    results_all = []
    equity_curves_all = {}
    
    # Run backtest for both BTC and ETH
    for symbol in ["BTC", "ETH"]:
        print(f"\nRunning backtest for {symbol} on {timeframe} timeframe for regimes {target_regimes}")
        
        try:
            results_df, equity_curves = run_full_backtest(symbol, timeframe, target_regimes)
            results_all.append(results_df)
            equity_curves_all.update(equity_curves)
            
        except Exception as e:
            print(f"Error running backtest for {symbol}: {str(e)}")
            continue
    
    if results_all:
        # Combine results
        combined_results = pd.concat(results_all, ignore_index=True)
        
        print("\nBacktest Results Summary:")
        print("=" * 80)
        print(combined_results.to_string())
        
        # Create subplots for visualization
        n_strategies = len(STRATEGY_PARAMS)
        fig = vbt.make_subplots(
            rows=n_strategies,
            cols=1,
            subplot_titles=[s for s in STRATEGY_PARAMS.keys()],
            vertical_spacing=0.05
        )
        
        # Plot results by strategy
        for i, strategy in enumerate(STRATEGY_PARAMS.keys(), 1):
            strategy_curves = pd.DataFrame({
                name: curve 
                for name, curve in equity_curves_all.items() 
                if f"_{strategy}_" in name
            })
            
            if not strategy_curves.empty:
                strategy_curves.vbt.plot(
                    add_trace_kwargs=dict(row=i, col=1),
                    fig=fig
                )
        
        fig.update_layout(
            height=300 * n_strategies,
            title="Strategy Performance by Symbol",
            showlegend=True
        )
        fig.show()
        
        # Save results to CSV in backtests directory
        results_file = Path("results/backtests") / f"backtest_results_regimes_{'_'.join(map(str, target_regimes))}_{timeframe}.csv"
        combined_results.to_csv(results_file)
        print(f"\nResults saved to: {results_file}")
        
    else:
        print("No results were generated for any symbol")