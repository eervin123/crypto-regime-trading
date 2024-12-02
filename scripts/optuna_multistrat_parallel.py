"""
Parallel Implementation of Multi-Strategy Cryptocurrency Trading Optimization

This module is an experimental parallel version of optuna_multistrat.py, designed to
test performance improvements through parallel processing.
"""

from optuna_multistrat import (
    load_config,
    load_data,
    calculate_regimes,
    create_stats,
    ensure_results_dir,
    save_optimal_params,
    format_results_table,
    # Import strategy functions
    run_ma_strategy_with_stops,
    run_macd_divergence_strategy_with_stops,
    run_rsi_divergence_strategy_with_stops,
    run_bbands_strategy_with_stops,
    run_psar_strategy_with_stops,
    run_rsi_mean_reversion_strategy,
    mean_reversion_strategy,
)
from tabulate import tabulate  # Add this import at the top of the file
import optuna
from optuna.samplers import TPESampler
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import logging
import pandas as pd
import numpy as np
import vectorbtpro as vbt
import gc
from pathlib import Path
from optuna.storages import RDBStorage
import plotly.io as pio
from plotly.subplots import make_subplots
import psutil  # Add at the top with other imports
import plotly.graph_objects as go

# Set the default renderer to 'browser' to open plots in your default web browser
pio.renderers.default = "browser"
# dark mode
vbt.settings.set_theme("dark")

# Add this near the top of the file, after the imports
STRATEGY_MAPPING = {
    "Moving Average": ("ma", run_ma_strategy_with_stops),
    "MACD Divergence": ("macd", run_macd_divergence_strategy_with_stops),
    "RSI Divergence": ("rsi", run_rsi_divergence_strategy_with_stops),
    "Bollinger Bands": ("bbands", run_bbands_strategy_with_stops),
    "Parabolic SAR": ("psar", run_psar_strategy_with_stops),
    "RSI Mean Reversion": ("rsi_mean_reversion", run_rsi_mean_reversion_strategy),
    "Mean Reversion": ("mean_reversion", mean_reversion_strategy),
}


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("parallel_optimization.log"),
            logging.StreamHandler(),
        ],
    )


def objective(
    trial,
    strategy_func,
    strategy_params,
    symbol_ohlcv_df,
    regime_data,
    allowed_regimes,
    config,
):
    """Objective function for Optuna optimization."""
    params = {}
    for k, v in strategy_params.items():
        if isinstance(v, list):
            # Handle categorical parameters
            params[k] = trial.suggest_categorical(k, v)
        elif isinstance(v, (list, tuple)) and len(v) == 2:
            # Handle numeric ranges
            if isinstance(v[0], int):
                params[k] = trial.suggest_int(k, v[0], v[1])
            else:
                params[k] = trial.suggest_float(k, v[0], v[1])
        else:
            # Pass fixed parameters as is
            params[k] = v

    try:
        pf = strategy_func(
            symbol_ohlcv_df=symbol_ohlcv_df,
            regime_data=regime_data,
            allowed_regimes=allowed_regimes,
            **params,
        )

        # Check minimum trades requirement
        if pf.trades.count() < config["objective"]["min_trades"]:
            return float("-inf")

        # Get objective type from config
        objective_type = config["objective"]["type"]

        # Calculate objective value based on type
        if objective_type == "weighted_avg":
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
                        return float("-inf")

                    objective_value += weight * value
        else:
            # Use single metric specified by type
            if objective_type == "calmar_ratio":
                objective_value = pf.calmar_ratio
            elif objective_type == "sharpe_ratio":
                objective_value = pf.sharpe_ratio
            elif objective_type == "sortino_ratio":
                objective_value = pf.sortino_ratio
            elif objective_type == "omega_ratio":
                objective_value = pf.omega_ratio
            elif objective_type == "total_return":
                objective_value = pf.total_return
            elif objective_type == "win_rate":
                objective_value = pf.trades.win_rate
            elif objective_type == "profit_factor":
                objective_value = pf.trades.profit_factor
            else:
                raise ValueError(f"Unknown objective type: {objective_type}")

            if pd.isna(objective_value) or np.isinf(objective_value):
                return float("-inf")

        # Add trade count weight if configured (applies to all objective types)
        trade_weight = config["objective"]["trade_weight"]
        if trade_weight > 0:
            objective_value += trade_weight * pf.trades.count()

        return float("-inf") if pd.isna(objective_value) else objective_value

    except Exception as e:
        print(f"Error in objective function: {str(e)}")
        return float("-inf")


def optimize_strategy_parallel(
    strategy_func,
    strategy_params,
    symbol_ohlcv_df,
    regime_data,
    allowed_regimes,
    direction,
    n_trials=None,
):
    """Parallel version of strategy optimization using Optuna."""
    if n_trials is None:
        n_trials = 100  # Default value

    # Add direction to strategy params
    params = strategy_params.copy()
    params["direction"] = direction

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(n_startup_trials=10, seed=42)
    )

    study.optimize(
        lambda trial: objective(
            trial,
            strategy_func,
            params,
            symbol_ohlcv_df,
            regime_data,
            allowed_regimes,
            config,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Get best parameters
    best_params = study.best_params
    best_params["direction"] = direction

    # Run strategy with best parameters
    best_pf = strategy_func(
        symbol_ohlcv_df=symbol_ohlcv_df,
        regime_data=regime_data,
        allowed_regimes=allowed_regimes,
        **best_params,
    )

    return best_params, best_pf, direction


def initialize_optuna_database():
    """Initialize SQLite database for Optuna studies."""
    try:
        # Create results directory for database
        db_dir = Path("results/optuna_db")
        db_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database with error handling
        storage = RDBStorage(
            url=f"sqlite:///{db_dir}/optuna_studies.db",
            heartbeat_interval=60,
            grace_period=120,
        )

        return storage
    except Exception as e:
        logging.error(f"Failed to initialize database: {str(e)}")
        raise


def optimize_single_strategy(args):
    """Optimize a single strategy with given parameters."""
    (
        name,
        func,
        params,
        symbol,
        direction,
        in_sample_data,
        in_sample_regimes,
        target_regimes,
        config,
    ) = args

    try:
        # Create a unique study name
        study_name = f"{symbol}_{name}_{direction}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = study_name.replace(" ", "_").lower()

        # Create new study without loading existing one
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(
                n_startup_trials=config["optimization"]["n_startup_trials"], seed=42
            ),
        )

        def objective_wrapper(trial):
            """Wrapper for the objective function to handle parameter configuration."""
            # Get parameter ranges from config
            strategy_params = {}
            for param_name, param_range in params.items():
                if (
                    param_name == "direction"
                ):  # Skip direction as it's handled separately
                    continue
                elif isinstance(param_range, list):
                    if isinstance(param_range[0], str):  # For timeframe parameters
                        strategy_params[param_name] = trial.suggest_categorical(
                            param_name, param_range
                        )
                    elif isinstance(param_range[0], bool):  # For boolean parameters
                        strategy_params[param_name] = trial.suggest_categorical(
                            param_name, param_range
                        )
                    else:  # For numerical parameters
                        if "window" in param_name or param_name in [
                            "lookback_window",
                            "rsi_window",
                            "atr_window",
                            "bb_window",
                            "fast_ma",
                            "slow_ma",
                            "macd_fast",
                            "macd_slow",
                            "macd_signal",
                        ]:
                            strategy_params[param_name] = trial.suggest_int(
                                param_name, int(param_range[0]), int(param_range[1])
                            )
                        else:
                            strategy_params[param_name] = trial.suggest_float(
                                param_name, param_range[0], param_range[1]
                            )

            strategy_params["direction"] = direction

            try:
                pf = func(
                    symbol_ohlcv_df=in_sample_data[symbol],
                    regime_data=in_sample_regimes[symbol],
                    allowed_regimes=target_regimes,
                    **strategy_params,
                )

                # Extract necessary metrics
                result = objective(
                    trial,
                    func,
                    strategy_params,
                    in_sample_data[symbol],
                    in_sample_regimes[symbol],
                    target_regimes,
                    config,
                )

                # Clear portfolio to free memory
                del pf
                gc.collect()

                return result

            except Exception as e:
                print(f"Error in trial with parameters {strategy_params}: {str(e)}")
                return float("-inf")

        # Get n_trials from config
        n_trials = config["optimization"]["n_trials"]
        print(f"Optimizing {name} {direction} for {symbol} ({n_trials} trials)")
        current_trial = 0
        
        def progress_callback(study, trial):
            nonlocal current_trial
            current_trial += 1
            if current_trial % 50 == 0:  # Update less frequently
                print(f"\r{name} {direction} {symbol}: {(current_trial/n_trials)*100:.1f}% complete", end="")
        
        study.optimize(
            objective_wrapper,
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[progress_callback]
        )

        if study.best_trial is not None:
            best_params = study.best_trial.params
            best_params["direction"] = direction

            # Run backtest with best parameters
            pf = func(
                symbol_ohlcv_df=in_sample_data[symbol],
                regime_data=in_sample_regimes[symbol],
                allowed_regimes=target_regimes,
                **best_params,
            )

            if pf is not None:
                # Create stats
                stats = create_stats(name, symbol, direction, pf, best_params)
                # Add equity curve separately
                stats["equity_curve"] = pf.value.to_numpy()

                # Clear portfolio to free memory
                del pf
                gc.collect()

                return stats

    except Exception as e:
        print(f"Error in optimize_single_strategy: {str(e)}")
        return None


def run_optimized_strategies_parallel(
    target_regimes,
    in_sample_data,
    out_sample_data,
    in_sample_regimes,
    out_sample_regimes,
):
    """Parallel implementation of strategy optimization."""
    try:
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Load config for parameter ranges and optimization settings
        config = load_config()
        strategy_params = config["strategy_params"]

        # Initialize results containers
        equity_curves = {}
        in_sample_results = []
        out_sample_results = []

        # Prepare optimization tasks
        optimization_tasks = []
        for symbol in ["BTC", "ETH"]:
            for strategy_name, (param_key, strategy_func) in STRATEGY_MAPPING.items():
                params = strategy_params[param_key].copy()
                for direction in ["long", "short"]:
                    optimization_tasks.append(
                        (
                            strategy_name,
                            strategy_func,
                            params,
                            symbol,
                            direction,
                            in_sample_data,
                            in_sample_regimes,
                            target_regimes,
                            config,
                        )
                    )

        # Calculate total expected trials
        n_trials = config["optimization"]["n_trials"]
        n_strategies = len(STRATEGY_MAPPING)
        total_trials = n_trials * n_strategies * 2 * 2  # 2 directions, 2 symbols
        completed_strategies = 0
        total_strategies = len(optimization_tasks)

        print(f"\nStarting optimization with {total_trials} total trials across {total_strategies} strategy combinations...")

        # Run optimizations in parallel with error handling
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = list(executor.map(optimize_single_strategy, optimization_tasks))

        # Process results
        for result in futures:
            if result is not None:
                try:
                    strategy_name = (
                        result["Strategy"]
                        .replace("(", "")
                        .replace(")", "")
                        .replace(" ", "_")
                    )
                    key = f"{result['Symbol']}_{strategy_name}_{result['Direction']}_in-sample"
                    equity_curves[key] = result.pop("equity_curve")
                    in_sample_results.append(result)
                except Exception as e:
                    logging.error(f"Error processing result: {str(e)}")
                    continue

        # Run out-of-sample tests
        for result in in_sample_results:
            symbol = result["Symbol"]
            strategy_name = result["Strategy"].split(" (")[0]
            direction = result["Direction"]
            strategy_func = STRATEGY_MAPPING[strategy_name][1]
            param_key = STRATEGY_MAPPING[strategy_name][0]

            # Get parameters (excluding metrics)
            strategy_specific_params = strategy_params[param_key]
            params = {k: result[k] for k in strategy_specific_params.keys()}
            params["direction"] = direction

            # Run out-of-sample backtest
            out_pf = strategy_func(
                symbol_ohlcv_df=out_sample_data[symbol],
                regime_data=out_sample_regimes[symbol],
                allowed_regimes=target_regimes,
                **params,
            )

            # Create out-of-sample stats
            out_result = create_stats(strategy_name, symbol, direction, out_pf, params)
            key = f"{symbol}_{strategy_name}_{direction}_out-of-sample"
            equity_curves[key] = out_pf.value.to_numpy()
            out_sample_results.append(out_result)

            # Clear out-of-sample portfolio
            del out_pf
            gc.collect()

        # Create DataFrames for results
        in_sample_df = pd.DataFrame(in_sample_results)
        out_sample_df = pd.DataFrame(out_sample_results)

        # Create equity curves DataFrame
        equity_curves_df = pd.DataFrame(
            {k: pd.Series(v) for k, v in equity_curves.items()}
        )

        # Save optimal parameters
        timeframe_id = config["timeframes"]["analysis"].replace("T", "min")
        save_optimal_params(in_sample_df, timeframe_id, target_regimes, config)

        # Create results directory and get subdirectories
        results_dir, params_dir, backtests_dir = ensure_results_dir()

        # Save results for each period
        for period, results_df in [
            ("in_sample", in_sample_df),
            ("out_sample", out_sample_df),
        ]:
            if results_df is not None and not results_df.empty:
                # Save to CSV with strategies as columns
                csv_df = results_df.drop(
                    "Portfolio", axis=1, errors="ignore"
                ).set_index(["Symbol", "Strategy", "Direction"])
                csv_df = csv_df.unstack(["Strategy", "Direction"])

                # Create filepath with timeframe
                filename = f"{period}_results_parallel_regimes_{'_'.join(map(str, target_regimes))}_{timeframe_id}.csv"
                filepath = backtests_dir / filename

                # Save the CSV
                csv_df.to_csv(filepath)
                print(f"Saved {period} results to {filepath}")

        # Save equity curves
        if equity_curves_df is not None and not equity_curves_df.empty:
            equity_curves_file = (
                backtests_dir
                / f"equity_curves_parallel_regimes_{'_'.join(map(str, target_regimes))}_{timeframe_id}.csv"
            )
            equity_curves_df.to_csv(equity_curves_file)
            print(f"\nEquity curves saved to: {equity_curves_file}")

        # Print the date ranges for reference
        print(
            f"In-sample period: {in_sample_data['BTC'].index[0]} to {in_sample_data['BTC'].index[-1]}"
        )
        print(
            f"Out-sample period: {out_sample_data['BTC'].index[0]} to {out_sample_data['BTC'].index[-1]}"
        )

        # Display tabulated results
        for symbol in ["BTC", "ETH"]:
            print(f"\n{symbol} In-Sample Results:")
            print(
                tabulate(
                    format_results_table(
                        in_sample_df[in_sample_df["Symbol"] == symbol]
                    ),
                    headers="keys",
                    tablefmt="pipe",
                    floatfmt=".4f",
                )
            )

            # Add browser table display
            in_sample_table = create_results_table(
                format_results_table(in_sample_df[in_sample_df["Symbol"] == symbol]),
                f"{symbol} In-Sample Results"
            )
            in_sample_table.show()

            print(f"\n{symbol} Out-of-Sample Results:")
            print(
                tabulate(
                    format_results_table(
                        out_sample_df[out_sample_df["Symbol"] == symbol]
                    ),
                    headers="keys",
                    tablefmt="pipe",
                    floatfmt=".4f",
                )
            )

            # Add browser table display
            out_sample_table = create_results_table(
                format_results_table(out_sample_df[out_sample_df["Symbol"] == symbol]),
                f"{symbol} Out-of-Sample Results"
            )
            out_sample_table.show()

            print("\n" + "=" * 80)

        return in_sample_df, out_sample_df, equity_curves_df

    except Exception as e:
        logging.error(f"Error in parallel optimization: {str(e)}")
        raise


def run_full_backtest(
    strategy_mapping,
    in_sample_df,
    data_analysis,
    aligned_regime_data,
    target_regimes,
    config,
):
    """Run full backtest using optimized parameters."""
    full_results = []
    equity_curves = {}

    # For each successful optimization result
    for _, row in in_sample_df.iterrows():
        symbol = row["Symbol"]
        strategy_name = row["Strategy"].split(" (")[0]
        direction = row["Direction"]

        # Get strategy function and parameters
        strategy_func = strategy_mapping[strategy_name][1]
        param_key = strategy_mapping[strategy_name][0]
        strategy_specific_params = config["strategy_params"][param_key]

        # Extract optimized parameters and ensure correct types
        params = {}
        for k in strategy_specific_params.keys():
            if k in row:
                # Convert to int if it's a window parameter
                if any(
                    window_name in k.lower()
                    for window_name in ["window", "ma", "period"]
                ):
                    params[k] = int(row[k])
                else:
                    params[k] = row[k]
        params["direction"] = direction

        try:
            # Run full backtest
            full_pf = strategy_func(
                symbol_ohlcv_df=data_analysis[symbol],
                regime_data=aligned_regime_data[symbol],
                allowed_regimes=target_regimes,
                **params,
            )

            # Create stats
            full_result = create_stats(
                strategy_name, symbol, direction, full_pf, params
            )
            key = f"{symbol}_{strategy_name}_{direction}_full"
            equity_curves[key] = full_pf.value.to_numpy()
            full_results.append(full_result)

            # Clear portfolio
            del full_pf
            gc.collect()

        except Exception as e:
            print(
                f"Error running full backtest for {symbol} {strategy_name} {direction}: {str(e)}"
            )
            continue

    # Create DataFrame
    full_df = pd.DataFrame(full_results) if full_results else pd.DataFrame()

    return full_df, equity_curves


def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")


def create_results_table(df, title):
    """Create a Plotly table figure from DataFrame."""
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='darkslategray',
            align='left',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='black',
            align='left',
            font=dict(color='white', size=11)
        )
    )])
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=400 * (len(df) // 5 + 1),  # Adjust height based on number of rows
    )
    
    return fig


if __name__ == "__main__":
    # Setup logging
    setup_logging()
    log_memory_usage()  # Initial memory usage

    # Load config
    config = load_config()
    log_memory_usage()  # After config load

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
    log_memory_usage()  # After data loading

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

    # Run optimization and get results
    log_memory_usage()  # Before optimization
    in_sample_df, out_sample_df, equity_curves_df = run_optimized_strategies_parallel(
        target_regimes,
        in_sample_data,
        out_sample_data,
        aligned_regime_data,
        aligned_regime_data,
    )
    log_memory_usage()  # After optimization

    # Create results directory and get subdirectories
    results_dir, params_dir, backtests_dir = ensure_results_dir()

    # Create timeframe identifier for filenames
    timeframe_id = f"{analysis_tf.replace('T', 'min')}"

    # Save results for each period
    for period, results_df in [
        ("in_sample", in_sample_df),
        ("out_sample", out_sample_df),
    ]:
        if results_df is not None and not results_df.empty:
            # Save to CSV with strategies as columns
            csv_df = results_df.drop("Portfolio", axis=1, errors="ignore").set_index(
                ["Symbol", "Strategy", "Direction"]
            )
            csv_df = csv_df.unstack(["Strategy", "Direction"])

            # Create filepath with timeframe
            filename = f"{period}_results_parallel_regimes_{'_'.join(map(str, target_regimes))}_{timeframe_id}.csv"
            filepath = backtests_dir / filename

            # Save the CSV
            csv_df.to_csv(filepath)
            print(f"Saved {period} results to {filepath}")

    # Save equity curves
    if equity_curves_df is not None and not equity_curves_df.empty:
        equity_curves_file = (
            backtests_dir
            / f"equity_curves_parallel_regimes_{'_'.join(map(str, target_regimes))}_{timeframe_id}.csv"
        )
        equity_curves_df.to_csv(equity_curves_file)
        print(f"\nEquity curves saved to: {equity_curves_file}")

    # Run full backtest with optimized parameters
    log_memory_usage()  # Before full backtest
    full_df, full_equity_curves = run_full_backtest(
        STRATEGY_MAPPING,
        in_sample_df,
        data_analysis,
        aligned_regime_data,
        target_regimes,
        config,
    )
    log_memory_usage()  # After full backtest

    # Now create visualizations after we have all the data
    log_memory_usage()  # Before plotting
    print("\nCreating performance visualizations...")
    for symbol in ["BTC", "ETH"]:
        # Create 3x1 subplots
        fig = vbt.make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[
                f"{symbol} In-Sample Performance",
                f"{symbol} Out-of-Sample Performance",
                f"{symbol} Full Backtest Performance",
            ],
            vertical_spacing=0.1,
        )

        # Get unique strategies for consistent colors
        strategies = in_sample_df[in_sample_df["Symbol"] == symbol]["Strategy"].unique()

        # Plot in-sample results
        for strategy in strategies:
            strategy_results = in_sample_df[
                (in_sample_df["Symbol"] == symbol)
                & (in_sample_df["Strategy"] == strategy)
            ]
            for _, result in strategy_results.iterrows():
                pf_fig = result["Portfolio"].plot_cum_returns()
                for trace in pf_fig.data:
                    trace.name = f"{strategy} ({result['Direction']})"
                    fig.add_trace(trace, row=1, col=1)

        # Plot out-of-sample results
        for strategy in strategies:
            strategy_results = out_sample_df[
                (out_sample_df["Symbol"] == symbol)
                & (out_sample_df["Strategy"] == strategy)
            ]
            for _, result in strategy_results.iterrows():
                pf_fig = result["Portfolio"].plot_cum_returns()
                for trace in pf_fig.data:
                    trace.name = f"{strategy} ({result['Direction']})"
                    fig.add_trace(trace, row=2, col=1)

        # Plot full backtest results
        for strategy in strategies:
            strategy_results = full_df[
                (full_df["Symbol"] == symbol) & (full_df["Strategy"] == strategy)
            ]
            for _, result in strategy_results.iterrows():
                pf_fig = result["Portfolio"].plot_cum_returns()
                for trace in pf_fig.data:
                    trace.name = f"{strategy} ({result['Direction']})"
                    fig.add_trace(trace, row=3, col=1)

        # Update layout
        fig.update_layout(
            height=1200,
            width=1200,
            title=f"{symbol} Strategy Performance Comparison",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
        )

        # Show the figure
        fig.show()
    log_memory_usage()  # After plotting
