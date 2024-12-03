"""
Multi-Strategy Cryptocurrency Trading Optimization Framework

This module implements a comprehensive framework for optimizing multiple trading strategies
across different market regimes using Optuna. It supports both in-sample and out-of-sample
testing for various technical analysis strategies on cryptocurrency data.

Key Features:
- Multiple strategy implementations (MA, MACD, RSI, Bollinger Bands, etc.)
- Market regime detection and filtering
- Parameter optimization using Optuna
- Performance analysis and visualization
- Configurable through YAML files

Usage:
    python optuna_multistrat.py
"""

from strategies import (
    run_ma_strategy_with_stops,
    run_macd_divergence_strategy_with_stops,
    run_rsi_divergence_strategy_with_stops,
    run_bbands_strategy_with_stops,
    run_psar_strategy_with_stops,
    run_rsi_mean_reversion_strategy,
    mean_reversion_strategy,
    calculate_regimes
)

# Import from our new modules
from data_utils import (
    load_data,
    validate_timeframe_params,
)
from config_utils import (
    load_config,
    ensure_results_dir
)
from results_utils import (
    create_stats,
    format_results_table,
    save_optimal_params,
    save_results_to_csv,
    plot_performance_comparison,
    display_results,
)
from db_utils import initialize_optuna_database

import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import vectorbtpro as vbt
from tabulate import tabulate
import plotly.io as pio
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import gc

# Set the default renderer to 'browser' to open plots in your default web browser
pio.renderers.default = "browser"
vbt.settings.set_theme("dark")

# Define parameter ranges for each strategy
config = load_config()
bbands_params = config["strategy_params"]["bbands"]
ma_params = config["strategy_params"]["ma"]
rsi_params = config["strategy_params"]["rsi"]
macd_params = config["strategy_params"]["macd"]
psar_params = config["strategy_params"]["psar"]
rsi_mean_reversion_params = config["strategy_params"]["rsi_mean_reversion"]
mean_reversion_params = config["strategy_params"]["mean_reversion"]


def optimize_strategy(
    strategy_func,
    strategy_params,
    symbol_ohlcv_df,
    regime_data,
    allowed_regimes,
    direction,
    n_trials=None,
    n_best=5,
):
    """
    Optimize strategy parameters using Optuna, returning top N results.

    Args:
        strategy_func (callable): Strategy function to optimize
        strategy_params (dict): Parameter space for optimization
        symbol_ohlcv_df (pd.DataFrame): OHLCV price data
        regime_data (pd.Series): Market regime labels
        allowed_regimes (list): List of regime labels to trade in
        direction (str): Trading direction ('long' or 'short')
        n_trials (int, optional): Number of optimization trials
        n_best (int): Number of best trials to return (default: 5)

    Returns:
        tuple: Tuple containing:
            - Best parameters
            - Portfolio object
            - Objective value
    """
    # Update the n_trials parameter to use config if not specified
    if n_trials is None:
        n_trials = config["optimization"]["n_trials"]

    # Add validation for timeframe parameters if they exist
    if "timeframe_1" in strategy_params and "timeframe_2" in strategy_params:
        validate_timeframe_params(
            strategy_params["timeframe_1"], strategy_params["timeframe_2"]
        )

    def objective(trial):
        params = {}
        for k, v in strategy_params.items():
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], (int, float)):
                if isinstance(v[0], int):
                    params[k] = trial.suggest_int(k, v[0], v[1])
                else:
                    params[k] = trial.suggest_float(k, v[0], v[1])
            elif isinstance(v, list):
                params[k] = trial.suggest_categorical(k, v)
            else:
                params[k] = v

        try:
            pf = strategy_func(
                symbol_ohlcv_df=symbol_ohlcv_df,
                regime_data=regime_data,
                allowed_regimes=allowed_regimes,
                **params,
            )

            if pf is None:
                print(f"Warning: Strategy returned None portfolio for params: {params}")
                return float("-inf")

            # Check minimum trades requirement
            if pf.trades.count() < config["objective"]["min_trades"]:
                print(
                    f"Warning: Not enough trades ({pf.trades.count()}) for params: {params}"
                )
                return float("-inf")

            # Calculate weighted objective based on configuration
            objective_value = 0.0
            metrics = config["objective"]["metrics"]

            # Add metric contributions based on weights
            for metric_name, metric_config in metrics.items():
                weight = metric_config["weight"]
                if weight > 0:
                    if metric_name == "calmar_ratio":
                        value = pf.calmar_ratio
                    elif metric_name == "sharpe_ratio":
                        value = pf.sharpe_ratio
                    elif metric_name == "sortino_ratio":
                        value = pf.sortino_ratio
                    elif metric_name == "omega_ratio":
                        value = pf.omega_ratio
                    elif metric_name == "total_return":
                        value = pf.total_return
                    elif metric_name == "win_rate":
                        value = pf.trades.win_rate
                    elif metric_name == "profit_factor":
                        value = pf.trades.profit_factor

                    if pd.isna(value) or np.isinf(value):
                        print(
                            f"Warning: Invalid {metric_name} ({value}) for params: {params}"
                        )
                        return float("-inf")

                    objective_value += weight * value

            # Add trade count weight if configured
            trade_weight = config["objective"]["trade_weight"]
            if trade_weight > 0:
                objective_value += trade_weight * pf.trades.count()

            return float("-inf") if pd.isna(objective_value) else objective_value

        except Exception as e:
            print(f"Error in objective function: {str(e)}, params: {params}")
            return float("-inf")

    sampler = TPESampler(n_startup_trials=10, seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=25, interval_steps=10)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def early_stopping_callback(study, trial):
        if study.best_trial.number + 200 < trial.number:
            study.stop()

    current_trial = 0

    def progress_callback(study, trial):
        nonlocal current_trial
        current_trial += 1
        if current_trial % 50 == 0:  # Update less frequently
            print(f"\rProgress: {(current_trial/n_trials)*100:.1f}% complete", end="")

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[early_stopping_callback, progress_callback],
        show_progress_bar=False,
    )

    # Get top N trials
    top_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else float("-inf"),
        reverse=True,
    )[:n_best]

    results = []
    for trial in top_trials:
        try:
            if trial.value is not None and trial.value > float("-inf"):
                params = trial.params.copy()
                params["direction"] = direction

                pf = strategy_func(
                    symbol_ohlcv_df=symbol_ohlcv_df,
                    regime_data=regime_data,
                    allowed_regimes=allowed_regimes,
                    **params,
                )

                if pf is not None:
                    results.append((params, pf, trial.value))

                # Clear memory
                del pf
                gc.collect()

        except Exception as e:
            print(f"Error processing trial: {str(e)}")
            continue

    # Return the best result for the main workflow
    best_result = (
        max(results, key=lambda x: x[2]) if results else (None, None, float("-inf"))
    )
    return best_result


def run_optimized_strategies(
    target_regimes,
    in_sample_data,
    out_sample_data,
    in_sample_regimes,
    out_sample_regimes,
):
    """
    Run optimization and backtesting for all strategies.

    Args:
        target_regimes (list): List of regime labels to trade in
        in_sample_data (dict): In-sample price data for each symbol
        out_sample_data (dict): Out-of-sample price data for each symbol
        in_sample_regimes (dict): In-sample regime data for each symbol
        out_sample_regimes (dict): Out-of-sample regime data for each symbol

    Returns:
        tuple: Containing:
            - in_sample_results (pd.DataFrame)
            - out_sample_results (pd.DataFrame)
            - combined_results (pd.DataFrame)
            - equity_curves_df (pd.DataFrame)
    """
    strategies = [
        ("Moving Average", run_ma_strategy_with_stops, ma_params),
        ("MACD Divergence", run_macd_divergence_strategy_with_stops, macd_params),
        ("RSI Divergence", run_rsi_divergence_strategy_with_stops, rsi_params),
        ("Bollinger Bands", run_bbands_strategy_with_stops, bbands_params),
        ("Parabolic SAR", run_psar_strategy_with_stops, psar_params),
        (
            "RSI Mean Reversion",
            run_rsi_mean_reversion_strategy,
            rsi_mean_reversion_params,
        ),
        ("Mean Reversion", mean_reversion_strategy, mean_reversion_params),
    ]

    # Optimize on in-sample data
    in_sample_results = []
    equity_curves = {}  # Initialize equity_curves dictionary

    for symbol in ["BTC", "ETH"]:
        for name, func, params in strategies:
            # Optimize for long
            long_params = params.copy()
            long_params["direction"] = ["long"]
            best_long_params, long_pf, _ = optimize_strategy(
                func,
                long_params,
                in_sample_data[symbol],
                in_sample_regimes[symbol],
                target_regimes,
                "long",
            )

            # Only add if strategy produced valid results
            if long_pf is not None:
                in_sample_results.append(
                    create_stats(name, symbol, "long", long_pf, best_long_params)
                )
                equity_curves[f"{symbol}_{name}_long_in_sample"] = long_pf.value
            else:
                print(f"Warning: {name} long strategy optimization failed for {symbol}")

            # Optimize for short
            short_params = params.copy()
            short_params["direction"] = ["short"]
            best_short_params, short_pf, _ = optimize_strategy(
                func,
                short_params,
                in_sample_data[symbol],
                in_sample_regimes[symbol],
                target_regimes,
                "short",
            )

            # Only add if strategy produced valid results
            if short_pf is not None:
                in_sample_results.append(
                    create_stats(name, symbol, "short", short_pf, best_short_params)
                )
                equity_curves[f"{symbol}_{name}_short_in_sample"] = short_pf.value
            else:
                print(
                    f"Warning: {name} short strategy optimization failed for {symbol}"
                )

    # After optimization but before out-of-sample testing, save the parameters
    timeframe_id = analysis_tf.replace("T", "min")
    save_optimal_params(
        pd.DataFrame(in_sample_results), timeframe_id, target_regimes, config
    )

    # Test optimized parameters on out-of-sample data
    out_sample_results = []
    combined_results = []

    for stat in in_sample_results:
        # Get the optimized parameters (excluding metrics and metadata)
        params = {
            k: v
            for k, v in stat.items()
            if k
            not in [
                "Symbol",
                "Strategy",
                "Direction",
                "Portfolio",
                "Total Return",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Win Rate",
                "Max Drawdown",
                "Calmar Ratio",
                "Omega Ratio",
                "Profit Factor",
                "Expectancy",
                "Total Trades",
            ]
        }

        symbol = stat["Symbol"]
        direction = stat["Direction"]
        strategy_name = stat["Strategy"].split(" (")[0]

        # Find the corresponding strategy function
        strategy_func = next(
            func for name, func, _ in strategies if name == strategy_name
        )

        # Run strategy with optimized params on out-of-sample data
        out_pf = strategy_func(
            symbol_ohlcv_df=out_sample_data[symbol],
            regime_data=out_sample_regimes[symbol],
            allowed_regimes=target_regimes,
            **params,
        )

        # Save out-of-sample equity curve
        equity_curves[f"{symbol}_{strategy_name}_{direction}_out_sample"] = out_pf.value

        # Create stats for out-of-sample
        out_sample_results.append(
            create_stats(strategy_name, symbol, direction, out_pf, params)
        )

        # Run strategy on combined data
        combined_data = pd.concat(
            [in_sample_data[symbol], out_sample_data[symbol]], verify_integrity=False
        )
        combined_regimes = pd.concat(
            [in_sample_regimes[symbol], out_sample_regimes[symbol]],
            verify_integrity=False,
        )

        # Remove duplicates
        combined_data = combined_data[~combined_data.index.duplicated(keep="first")]
        combined_regimes = combined_regimes[
            ~combined_regimes.index.duplicated(keep="first")
        ]

        # Sort index
        combined_data = combined_data.sort_index()
        combined_regimes = combined_regimes.sort_index()

        # Ensure alignment
        common_idx = combined_data.index.intersection(combined_regimes.index)
        combined_data = combined_data.loc[common_idx]
        combined_regimes = combined_regimes.loc[common_idx]

        combined_pf = strategy_func(
            symbol_ohlcv_df=combined_data,
            regime_data=combined_regimes,
            allowed_regimes=target_regimes,
            **params,
        )

        # Save combined equity curve
        equity_curves[f"{symbol}_{strategy_name}_{direction}_combined"] = (
            combined_pf.value
        )

        # Create stats for combined sample
        combined_results.append(
            create_stats(strategy_name, symbol, direction, combined_pf, params)
        )

    # Create equity curves DataFrame
    equity_curves_df = pd.DataFrame({k: pd.Series(v) for k, v in equity_curves.items()})

    return (
        pd.DataFrame(in_sample_results),
        pd.DataFrame(out_sample_results),
        pd.DataFrame(combined_results),
        equity_curves_df,
    )


if __name__ == "__main__":
    # Load config
    config = load_config()

    # Get timeframes from config
    base_tf = config["timeframes"]["base"]
    analysis_tf = config["timeframes"]["analysis"]
    regime_tf = config["timeframes"]["regime"]

    # Load data with specified timeframes
    data_analysis, data_regime = load_data(
        base_timeframe=base_tf,
        analysis_timeframe=analysis_tf,
        regime_timeframe=regime_tf,
    )

    # Calculate regimes
    aligned_regime_data = calculate_regimes(
        data_regime, data_analysis, analysis_timeframe=analysis_tf
    )

    # Calculate split index
    in_sample_pct = config["data"]["in_sample_pct"]
    split_idx = {
        symbol: int(len(data_analysis[symbol]) * in_sample_pct)
        for symbol in ["BTC", "ETH"]
    }

    # Split data
    in_sample_data = {
        symbol: data_analysis[symbol].iloc[: split_idx[symbol]]
        for symbol in ["BTC", "ETH"]
    }
    out_sample_data = {
        symbol: data_analysis[symbol].iloc[split_idx[symbol] :]
        for symbol in ["BTC", "ETH"]
    }

    # Get target regimes from config
    target_regimes = config["regime"]["target_regimes"]

    # Initialize database storage
    storage = initialize_optuna_database()

    # Run optimization and get both in-sample and out-of-sample results
    in_sample_results, out_sample_results, combined_results, equity_curves_df = (
        run_optimized_strategies(
            target_regimes,
            in_sample_data,
            out_sample_data,
            aligned_regime_data,
            aligned_regime_data,
        )
    )

    # Create results directory and get subdirectories
    results_dir, params_dir, backtests_dir = ensure_results_dir()

    # Create timeframe identifier for filenames
    timeframe_id = f"{analysis_tf.replace('T', 'min')}"

    # Create results dictionary
    results_dict = {
        "in_sample": in_sample_results,
        "out_sample": out_sample_results,
        "combined": combined_results,
    }

    # Save results
    save_results_to_csv(results_dict, target_regimes, timeframe_id, backtests_dir)

    # Save equity curves
    equity_curves_file = (
        backtests_dir
        / f"equity_curves_regimes_{'_'.join(map(str, target_regimes))}_{timeframe_id}.csv"
    )
    equity_curves_df.to_csv(equity_curves_file)
    print(f"\nEquity curves saved to: {equity_curves_file}")

    # Print date ranges
    print(
        f"In-sample period: {in_sample_data['BTC'].index[0]} to {in_sample_data['BTC'].index[-1]}"
    )
    print(
        f"Out-sample period: {out_sample_data['BTC'].index[0]} to {out_sample_data['BTC'].index[-1]}"
    )

    # Display results using new function with vertical mode
    display_results(results_dict, display_mode="vertical")
