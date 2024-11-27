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

Dependencies:
    - vectorbtpro
    - optuna
    - pandas
    - numpy
    - plotly
    - yaml
"""

from regimes_multi_strat_pf import (
    calculate_regimes_nb,
    psar_nb_with_next,
)

import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import vectorbtpro as vbt
from tabulate import tabulate
import plotly.io as pio
import numpy as np
import yaml
from pathlib import Path
import json
from datetime import datetime

# Set the default renderer to 'browser' to open plots in your default web browser
pio.renderers.default = "browser"

vbt.settings.set_theme("dark")


def load_config():
    """
    Load configuration settings from YAML file.

    Returns:
        dict: Configuration dictionary containing:
            - optimization parameters
            - strategy parameters
            - data split settings
            - regime settings
    """
    config_path = Path("config/optuna_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(base_timeframe="1T", analysis_timeframe="30T", regime_timeframe="1D"):
    """
    Load and prepare BTC and ETH data at different timeframes.
    
    Args:
        base_timeframe (str): Base timeframe of raw data (e.g., "1T" for 1-minute)
        analysis_timeframe (str): Main analysis timeframe (e.g., "30T", "1H", "4H")
        regime_timeframe (str): Timeframe for regime calculation (e.g., "1D")
    """
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
    """
    Calculate and align market regimes for BTC and ETH across timeframes.

    Args:
        data_regime (dict): Dictionary containing regime timeframe data for each symbol
        data_analysis (dict): Dictionary containing analysis timeframe data for each symbol
        analysis_timeframe (str): Timeframe for analysis (e.g., "30T", "1H")
    """
    # Create regime indicators
    RegimeIndicator = vbt.IndicatorFactory(
        class_name="RegimeIndicator",
        input_names=["price", "returns"],
        param_names=["ma_short_window", "ma_long_window", "vol_short_window", "avg_vol_window"],
        output_names=["regimes"]
    ).with_apply_func(calculate_regimes_nb)

    aligned_regime_data = {}
    
    for symbol in ["BTC", "ETH"]:
        # Calculate regime indicators
        regime_indicator = RegimeIndicator.run(
            data_regime[symbol]["Close"],
            data_regime[symbol]["Return"],
            ma_short_window=21,
            ma_long_window=88,
            vol_short_window=21,
            avg_vol_window=365,
        )

        # Add regimes to regime timeframe data
        data_regime[symbol]["Market Regime"] = regime_indicator.regimes.values

        # Resample and align regime data to analysis timeframe
        regime_data = data_regime[symbol]["Market Regime"]
        analysis_regime_data = regime_data.resample(analysis_timeframe).ffill()
        
        # Align with analysis timeframe data
        aligned_regime_data[symbol] = analysis_regime_data.reindex(
            data_analysis[symbol].index, method="ffill"
        )

    return aligned_regime_data


def validate_timeframe_params(tf1_list, tf2_list):
    """Validate that timeframe lists have no overlap and tf2 values are larger than tf1."""

    # Convert timeframe strings to hours
    def to_hours(tf):
        if isinstance(tf, (int, float)):
            return tf
        num = int("".join(filter(str.isdigit, tf)))
        unit = "".join(filter(str.isalpha, tf)).upper()
        if unit == "H":
            return num
        elif unit == "D":
            return num * 24
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    # Get the maximum value from tf1_list and minimum value from tf2_list
    max_tf1 = max(to_hours(tf) for tf in tf1_list)
    min_tf2 = min(to_hours(tf) for tf in tf2_list)

    if min_tf2 <= max_tf1:
        raise ValueError(
            "All timeframe_2 values must be larger than timeframe_1 values"
        )


def optimize_strategy(strategy_func, strategy_params, symbol_ohlcv_df, regime_data, allowed_regimes, n_trials=None):
    """
    Optimize strategy parameters using Optuna.

    Args:
        strategy_func (callable): Strategy function to optimize
        strategy_params (dict): Parameter space for optimization
        symbol_ohlcv_df (pd.DataFrame): OHLCV price data
        regime_data (pd.Series): Market regime labels
        allowed_regimes (list): List of regime labels to trade in
        n_trials (int, optional): Number of optimization trials

    Returns:
        tuple: (best_params, best_portfolio, best_direction)
            Optimized parameters and resulting portfolio
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

        pf = strategy_func(
            symbol_ohlcv_df=symbol_ohlcv_df,
            regime_data=regime_data,
            allowed_regimes=allowed_regimes,
            **params,
        )

        # Objective function options to maximize:

        # Current objective: Balance between trade frequency and returns
        # - Weights number of trades (20%) to avoid overfitting on few trades
        objective = (pf.trades.count() * 0.20) * pf.total_return

        # Alternative objectives to consider:
        # pf.total_return              # Simple returns - good for pure performance
        # pf.sharpe_ratio             # Returns/volatility - good for risk-adjusted performance
        # pf.sortino_ratio            # Similar to Sharpe but only penalizes downside volatility
        # pf.omega_ratio              # Probability weighted ratio of gains vs losses
        # pf.trades.win_rate          # Pure win rate - but beware of small gains vs large losses
        # pf.calmar_ratio             # Returns/max drawdown - good for drawdown-sensitive strategies
        # pf.trades.profit_factor     # Gross profits/gross losses - good for consistent profitability

        return (
            float("-inf") if pd.isna(objective) else objective
        )  # Return -inf for invalid strategies

    sampler = TPESampler(n_startup_trials=10, seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=25, interval_steps=10)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def early_stopping_callback(study, trial):
        if study.best_trial.number + 200 < trial.number:
            study.stop()

    study.optimize(objective, n_trials=n_trials, callbacks=[early_stopping_callback])

    best_params = study.best_params
    best_direction = best_params["direction"]

    print(
        f"Best parameters for {strategy_func.__name__} ({best_direction}): ",
        best_params,
    )
    print("Best value: ", study.best_value)

    # Backtest with the best parameters
    best_pf = strategy_func(
        symbol_ohlcv_df=symbol_ohlcv_df,
        regime_data=regime_data,
        allowed_regimes=allowed_regimes,
        **best_params,
    )

    return best_params, best_pf, best_direction


# Define parameter ranges for each strategy
config = load_config()
bbands_params = config["strategy_params"]["bbands"]
ma_params = config["strategy_params"]["ma"]
rsi_params = config["strategy_params"]["rsi"]
macd_params = config["strategy_params"]["macd"]
psar_params = config["strategy_params"]["psar"]
rsi_mean_reversion_params = config["strategy_params"]["rsi_mean_reversion"]
mean_reversion_params = config["strategy_params"]["mean_reversion"]


def run_rsi_mean_reversion_strategy(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    direction: str = "long",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
    rsi_window: int = 14,
    rsi_lower: int = 30,
    rsi_upper: int = 70,
):
    """
    Implements an RSI mean reversion strategy with regime filtering and optional stop-loss/take-profit.
    
    Args:
        symbol_ohlcv_df (pd.DataFrame): OHLCV price data
        regime_data (pd.Series): Market regime labels
        allowed_regimes (list): List of regime labels to trade in
        direction (str, optional): Trading direction ('long' or 'short'). Defaults to "long".
        use_sl_tp (bool, optional): Whether to use stop-loss/take-profit. Defaults to True.
        atr_window (int, optional): ATR calculation window. Defaults to 14.
        atr_multiplier (float, optional): Multiplier for ATR-based stops. Defaults to 2.0.
        fees (float, optional): Trading fees as decimal. Defaults to 0.001.
        rsi_window (int, optional): RSI calculation window. Defaults to 14.
        rsi_lower (int, optional): RSI oversold threshold. Defaults to 30.
        rsi_upper (int, optional): RSI overbought threshold. Defaults to 70.

    Returns:
        vbt.Portfolio: Portfolio object containing strategy results
    """
    rsi = vbt.RSI.run(close=symbol_ohlcv_df["Close"], window=rsi_window)

    # Determine entries
    long_entries = (rsi.rsi < rsi_lower) & (regime_data.isin(allowed_regimes))
    short_entries = (rsi.rsi > rsi_upper) & (regime_data.isin(allowed_regimes))
    regime_exits = ~regime_data.isin(allowed_regimes)

    pf_kwargs = {
        "close": symbol_ohlcv_df["Close"],
        "open": symbol_ohlcv_df["Open"],
        "high": symbol_ohlcv_df["High"],
        "low": symbol_ohlcv_df["Low"],
        "fees": fees,
    }

    if use_sl_tp:
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df["High"],
            low=symbol_ohlcv_df["Low"],
            close=symbol_ohlcv_df["Close"],
            window=atr_window,
        ).atr

        if direction == "long":
            pf_kwargs.update(
                {
                    "sl_stop": symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "sl_stop": symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )

    if direction == "long":
        pf_kwargs.update({"entries": long_entries, "exits": regime_exits})
    else:
        pf_kwargs.update({"short_entries": short_entries, "short_exits": regime_exits})

    return vbt.PF.from_signals(**pf_kwargs)


def mean_reversion_strategy(
    symbol_ohlcv_df,
    regime_data,
    allowed_regimes,
    direction="long",
    bb_window=21,
    bb_alpha=2.0,
    timeframe_1="4H",
    timeframe_2="24H",
    use_sl_tp: bool = True,
    atr_window=14,
    atr_multiplier=3.0,
    fees: float = 0.001,
    **kwargs,
):
    """
    Implements a dual timeframe Bollinger Bands mean reversion strategy with regime filtering.
    
    Uses VBT's optimized Bollinger Bands implementation to identify mean reversion opportunities
    across two different timeframes. Includes optional stop-loss/take-profit based on ATR.

    Args:
        symbol_ohlcv_df (pd.DataFrame): OHLCV price data
        regime_data (pd.Series): Market regime labels
        allowed_regimes (list): List of regime labels to trade in
        direction (str, optional): Trading direction ('long' or 'short'). Defaults to "long".
        bb_window (int, optional): Bollinger Bands calculation window. Defaults to 21.
        bb_alpha (float, optional): Number of standard deviations for bands. Defaults to 2.0.
        timeframe_1 (str, optional): First timeframe for analysis. Defaults to "4H".
        timeframe_2 (str, optional): Second timeframe for analysis. Defaults to "24H".
        use_sl_tp (bool, optional): Whether to use stop-loss/take-profit. Defaults to True.
        atr_window (int, optional): ATR calculation window. Defaults to 14.
        atr_multiplier (float, optional): Multiplier for ATR-based stops. Defaults to 3.0.
        fees (float, optional): Trading fees as decimal. Defaults to 0.001.
        **kwargs: Additional keyword arguments

    Returns:
        vbt.Portfolio: Portfolio object containing strategy results
    """

    data = vbt.BinanceData.from_data(symbol_ohlcv_df)

    bbands_tf1 = vbt.talib("BBANDS").run(
        data.close,
        timeperiod=bb_window,
        nbdevup=bb_alpha,
        nbdevdn=bb_alpha,
        timeframe=timeframe_1,
    )
    bbands_tf2 = vbt.talib("BBANDS").run(
        data.close,
        timeperiod=bb_window,
        nbdevup=bb_alpha,
        nbdevdn=bb_alpha,
        timeframe=timeframe_2,
    )

    # Generate entry conditions
    long_entries = (
        (data.close < bbands_tf2.middleband) & (data.close < bbands_tf1.lowerband)
    ) | ((data.close > bbands_tf2.lowerband) & (data.close < bbands_tf1.lowerband))

    short_entries = (
        (data.close > bbands_tf2.middleband) & (data.close > bbands_tf1.upperband)
    ) | ((data.close < bbands_tf2.upperband) & (data.close > bbands_tf1.upperband))

    allowed_regime_mask = regime_data.isin(allowed_regimes)
    long_entries = long_entries & allowed_regime_mask
    short_entries = short_entries & allowed_regime_mask
    regime_change_exits = allowed_regime_mask.shift(1) & ~allowed_regime_mask

    pf_kwargs = {
        "close": data.close,
        "open": data.open,
        "high": data.high,
        "low": data.low,
        "fees": fees,
    }

    if use_sl_tp:
        atr = vbt.ATR.run(data.high, data.low, data.close, window=atr_window).atr

        if direction == "long":
            pf_kwargs.update(
                {
                    "sl_stop": data.close - atr_multiplier * atr,
                    "tp_stop": data.close + atr_multiplier * atr,
                }
            )
        else:
            pf_kwargs.update(
                {
                    "sl_stop": data.close + atr_multiplier * atr,
                    "tp_stop": data.close - atr_multiplier * atr,
                }
            )

    if direction == "long":
        pf_kwargs.update(
            {"entries": long_entries, "exits": regime_change_exits | short_entries}
        )
    else:
        pf_kwargs.update(
            {
                "short_entries": short_entries,
                "short_exits": regime_change_exits | long_entries,
            }
        )

    return vbt.Portfolio.from_signals(**pf_kwargs)


def run_ma_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    fast_ma: int = 21,
    slow_ma: int = 55,
    direction: str = "long",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
):
    """
    Implements a dual moving average crossover strategy with regime filtering and optional stops.

    Args:
        symbol_ohlcv_df (pd.DataFrame): OHLCV price data
        regime_data (pd.Series): Market regime labels
        allowed_regimes (list): List of regime labels to trade in
        fast_ma (int, optional): Fast moving average period. Defaults to 21.
        slow_ma (int, optional): Slow moving average period. Defaults to 55.
        direction (str, optional): Trading direction ('long' or 'short'). Defaults to "long".
        use_sl_tp (bool, optional): Whether to use stop-loss/take-profit. Defaults to True.
        atr_window (int, optional): ATR calculation window. Defaults to 14.
        atr_multiplier (float, optional): Multiplier for ATR-based stops. Defaults to 2.0.
        fees (float, optional): Trading fees as decimal. Defaults to 0.001.

    Returns:
        vbt.Portfolio: Portfolio object containing strategy results
    """
    fast_ma = vbt.MA.run(symbol_ohlcv_df.Close, window=fast_ma).ma
    slow_ma = vbt.MA.run(symbol_ohlcv_df.Close, window=slow_ma).ma

    long_entries = fast_ma > slow_ma
    long_exits = fast_ma < slow_ma
    short_entries = fast_ma < slow_ma
    short_exits = fast_ma > slow_ma

    # Add regime filter
    long_entries = long_entries & regime_data.isin(allowed_regimes)
    short_entries = short_entries & regime_data.isin(allowed_regimes)
    long_regime_exits = ~regime_data.isin(allowed_regimes)
    short_regime_exits = ~regime_data.isin(allowed_regimes)

    # Combine regime exits with other exit conditions
    long_exits = long_exits | long_regime_exits
    short_exits = short_exits | short_regime_exits

    pf_kwargs = {
        "close": symbol_ohlcv_df.Close,
        "open": symbol_ohlcv_df.Open,
        "high": symbol_ohlcv_df.High,
        "low": symbol_ohlcv_df.Low,
        "fees": fees,
    }

    if use_sl_tp:
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df.High,
            low=symbol_ohlcv_df.Low,
            close=symbol_ohlcv_df.Close,
            window=atr_window,
        ).atr

        if direction == "long":
            pf_kwargs.update(
                {
                    "entries": long_entries,
                    "exits": long_exits,
                    "sl_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "short_entries": short_entries,
                    "short_exits": short_exits,
                    "sl_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
    else:
        if direction == "long":
            pf_kwargs.update({"entries": long_entries, "exits": long_exits})
        else:
            pf_kwargs.update(
                {"short_entries": short_entries, "short_exits": short_exits}
            )

    return vbt.PF.from_signals(**pf_kwargs)


def run_macd_divergence_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
    direction: str = "long",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
):
    """
    Implements a MACD crossover strategy with regime filtering and optional stops.

    Args:
        symbol_ohlcv_df (pd.DataFrame): OHLCV price data
        regime_data (pd.Series): Market regime labels
        allowed_regimes (list): List of regime labels to trade in
        fast_window (int, optional): Fast EMA period for MACD. Defaults to 12.
        slow_window (int, optional): Slow EMA period for MACD. Defaults to 26.
        signal_window (int, optional): Signal line period. Defaults to 9.
        direction (str, optional): Trading direction ('long' or 'short'). Defaults to "long".
        use_sl_tp (bool, optional): Whether to use stop-loss/take-profit. Defaults to True.
        atr_window (int, optional): ATR calculation window. Defaults to 14.
        atr_multiplier (float, optional): Multiplier for ATR-based stops. Defaults to 2.0.
        fees (float, optional): Trading fees as decimal. Defaults to 0.001.

    Returns:
        vbt.Portfolio: Portfolio object containing strategy results
    """
    # Calculate MACD
    macd = vbt.MACD.run(
        symbol_ohlcv_df["Close"],
        fast_window=fast_window,
        slow_window=slow_window,
        signal_window=signal_window,
    )

    # Generate entry signals
    if direction == "long":
        entries = (macd.macd > macd.signal) & (
            macd.macd.shift(1) <= macd.signal.shift(1)
        )
    else:  # short
        entries = (macd.macd < macd.signal) & (
            macd.macd.shift(1) >= macd.signal.shift(1)
        )

    # Apply regime filter
    entries = entries & regime_data.isin(allowed_regimes)

    pf_kwargs = {
        "close": symbol_ohlcv_df["Close"],
        "open": symbol_ohlcv_df["Open"],
        "high": symbol_ohlcv_df["High"],
        "low": symbol_ohlcv_df["Low"],
        "fees": fees,
    }

    if use_sl_tp:
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df["High"],
            low=symbol_ohlcv_df["Low"],
            close=symbol_ohlcv_df["Close"],
            window=atr_window,
        ).atr

        if direction == "long":
            pf_kwargs.update(
                {
                    "sl_stop": symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "sl_stop": symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )

    if direction == "long":
        pf_kwargs.update(
            {"entries": entries, "exits": ~regime_data.isin(allowed_regimes)}
        )
    else:
        pf_kwargs.update(
            {
                "short_entries": entries,
                "short_exits": ~regime_data.isin(allowed_regimes),
            }
        )

    return vbt.PF.from_signals(**pf_kwargs)


def run_rsi_divergence_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    rsi_window: int = 14,
    rsi_threshold: int = 30,
    lookback_window: int = 25,
    direction: str = "long",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
):
    # Calculate RSI
    rsi = vbt.RSI.run(symbol_ohlcv_df["Close"], window=rsi_window).rsi

    # Calculate rolling minimum for price and RSI
    price_min = symbol_ohlcv_df["Close"].rolling(window=lookback_window).min()
    rsi_min = rsi.rolling(window=lookback_window).min()

    # Generate entry signals
    if direction == "long":
        entries = (
            (symbol_ohlcv_df["Close"] == price_min)
            & (rsi < rsi_threshold)
            & (rsi > rsi_min)
            & (regime_data.isin(allowed_regimes))
        )
    else:  # short
        entries = (
            (
                symbol_ohlcv_df["Close"]
                == symbol_ohlcv_df["Close"].rolling(window=lookback_window).max()
            )
            & (rsi > 100 - rsi_threshold)
            & (rsi < rsi.rolling(window=lookback_window).max())
            & (regime_data.isin(allowed_regimes))
        )

    pf_kwargs = {
        "close": symbol_ohlcv_df["Close"],
        "open": symbol_ohlcv_df["Open"],
        "high": symbol_ohlcv_df["High"],
        "low": symbol_ohlcv_df["Low"],
        "fees": fees,
    }

    if use_sl_tp:
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df["High"],
            low=symbol_ohlcv_df["Low"],
            close=symbol_ohlcv_df["Close"],
            window=atr_window,
        ).atr

        if direction == "long":
            pf_kwargs.update(
                {
                    "sl_stop": symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "sl_stop": symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )

    if direction == "long":
        pf_kwargs.update(
            {"entries": entries, "exits": ~regime_data.isin(allowed_regimes)}
        )
    else:
        pf_kwargs.update(
            {
                "short_entries": entries,
                "short_exits": ~regime_data.isin(allowed_regimes),
            }
        )

    return vbt.PF.from_signals(**pf_kwargs)


def run_psar_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    af0: float = 0.02,
    af_increment: float = 0.02,
    max_af: float = 0.2,
    direction: str = "long",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
):
    """
    Implements a Parabolic SAR strategy with regime filtering and optional stops.

    Args:
        symbol_ohlcv_df (pd.DataFrame): OHLCV price data
        regime_data (pd.Series): Market regime labels
        allowed_regimes (list): List of regime labels to trade in
        af0 (float, optional): Initial acceleration factor. Defaults to 0.02.
        af_increment (float, optional): Acceleration factor increment. Defaults to 0.02.
        max_af (float, optional): Maximum acceleration factor. Defaults to 0.2.
        direction (str, optional): Trading direction ('long' or 'short'). Defaults to "long".
        use_sl_tp (bool, optional): Whether to use stop-loss/take-profit. Defaults to True.
        atr_window (int, optional): ATR calculation window. Defaults to 14.
        atr_multiplier (float, optional): Multiplier for ATR-based stops. Defaults to 2.0.
        fees (float, optional): Trading fees as decimal. Defaults to 0.001.

    Returns:
        vbt.Portfolio: Portfolio object containing strategy results
    """
    # Calculate PSAR
    long, short, _, _, _, _ = psar_nb_with_next(
        symbol_ohlcv_df["High"].values,
        symbol_ohlcv_df["Low"].values,
        symbol_ohlcv_df["Close"].values,
        af0=af0,
        af_increment=af_increment,
        max_af=max_af,
    )

    # Generate entry signals
    if direction == "long":
        entries = pd.Series(
            long < symbol_ohlcv_df["Low"].values, index=symbol_ohlcv_df.index
        )
    else:  # short
        entries = pd.Series(
            short > symbol_ohlcv_df["High"].values, index=symbol_ohlcv_df.index
        )

    # Apply regime filter
    entries = entries & regime_data.isin(allowed_regimes)

    pf_kwargs = {
        "close": symbol_ohlcv_df["Close"],
        "open": symbol_ohlcv_df["Open"],
        "high": symbol_ohlcv_df["High"],
        "low": symbol_ohlcv_df["Low"],
        "fees": fees,
    }

    if use_sl_tp:
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df["High"],
            low=symbol_ohlcv_df["Low"],
            close=symbol_ohlcv_df["Close"],
            window=atr_window,
        ).atr

        if direction == "long":
            pf_kwargs.update(
                {
                    "sl_stop": symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "sl_stop": symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )

    if direction == "long":
        pf_kwargs.update(
            {"entries": entries, "exits": ~regime_data.isin(allowed_regimes)}
        )
    else:
        pf_kwargs.update(
            {
                "short_entries": entries,
                "short_exits": ~regime_data.isin(allowed_regimes),
            }
        )

    return vbt.PF.from_signals(**pf_kwargs)


def run_bbands_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    direction: str = "long",
    fees: float = 0.001,
    bb_window: int = 14,
    bb_alpha: float = 2,
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: int = 5,
    **kwargs,
):
    # Calculate Bollinger Bands
    bbands_run = vbt.BBANDS.run(
        close=symbol_ohlcv_df["Close"], window=bb_window, alpha=bb_alpha
    )

    # Determine entries
    long_entries = (symbol_ohlcv_df["Close"] < bbands_run.lower) & (
        regime_data.isin(allowed_regimes)
    )
    short_entries = (symbol_ohlcv_df["Close"] > bbands_run.upper) & (
        regime_data.isin(allowed_regimes)
    )

    # Create exit signals when leaving allowed regimes
    regime_exits = ~regime_data.isin(allowed_regimes)

    # Common portfolio parameters
    pf_kwargs = {
        "close": symbol_ohlcv_df["Close"],
        "open": symbol_ohlcv_df["Open"],
        "high": symbol_ohlcv_df["High"],
        "low": symbol_ohlcv_df["Low"],
        "fees": fees,
    }

    if use_sl_tp:
        # Calculate ATR and stops
        atr = vbt.ATR.run(
            high=symbol_ohlcv_df["High"],
            low=symbol_ohlcv_df["Low"],
            close=symbol_ohlcv_df["Close"],
            window=atr_window,
        ).atr

        if direction == "long":
            pf_kwargs.update(
                {
                    "entries": long_entries,
                    "exits": regime_exits,
                    "sl_stop": symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "short_entries": short_entries,
                    "short_exits": regime_exits,
                    "sl_stop": symbol_ohlcv_df["Close"] + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df["Close"] - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
    else:
        if direction == "long":
            pf_kwargs.update(
                {
                    "entries": long_entries,
                    "exits": regime_exits,
                }
            )
        else:
            pf_kwargs.update(
                {
                    "short_entries": short_entries,
                    "short_exits": regime_exits,
                }
            )

    return vbt.PF.from_signals(**pf_kwargs)


def create_stats(name, symbol, direction, pf, params):
    """
    Create a dictionary of strategy performance statistics and parameters.

    Args:
        name (str): Strategy name
        symbol (str): Trading symbol (e.g., 'BTC', 'ETH')
        direction (str): Trading direction ('long' or 'short')
        pf (vbt.Portfolio): Portfolio object with strategy results
        params (dict): Strategy parameters used

    Returns:
        dict: Dictionary containing:
            - Symbol: Trading symbol
            - Strategy: Strategy name with direction
            - Direction: Trading direction
            - Total Return: Overall strategy return
            - Sharpe Ratio: Risk-adjusted return metric
            - Sortino Ratio: Downside risk-adjusted return
            - Win Rate: Percentage of winning trades
            - Max Drawdown: Maximum peak to trough decline
            - Calmar Ratio: Return to max drawdown ratio
            - Omega Ratio: Probability weighted ratio of gains vs losses
            - Profit Factor: Gross profits divided by gross losses
            - Expectancy: Average profit per trade
            - Total Trades: Number of completed trades
            - Portfolio: VBT Portfolio object
            - Strategy parameters used
    """
    return {
        "Symbol": symbol,
        "Strategy": f"{name} ({direction.capitalize()})",
        "Direction": direction,
        "Total Return": pf.total_return,
        "Sharpe Ratio": pf.sharpe_ratio,
        "Sortino Ratio": pf.sortino_ratio,
        "Win Rate": pf.trades.win_rate,
        "Max Drawdown": pf.max_drawdown,
        "Calmar Ratio": pf.calmar_ratio,
        "Omega Ratio": pf.omega_ratio,
        "Profit Factor": pf.trades.profit_factor,
        "Expectancy": pf.trades.expectancy,
        "Total Trades": pf.trades.count(),
        "Portfolio": pf,  # Store the Portfolio object
        **params,
    }


def ensure_results_dir():
    """Create results directory and subdirectories if they don't exist."""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create params subdirectory
    params_dir = results_dir / "params"
    params_dir.mkdir(exist_ok=True)
    
    # Create backtests subdirectory
    backtests_dir = results_dir / "backtests"
    backtests_dir.mkdir(exist_ok=True)
    
    return results_dir, params_dir, backtests_dir

def save_optimal_params(in_sample_results, timeframe_id, target_regimes, config):
    """
    Save optimal parameters for each strategy to JSON files with config metadata.
    """
    # Get the params directory
    results_dir, params_dir, _ = ensure_results_dir()  # Now properly unpacking the tuple
    
    # Strategy name mapping between results and config
    strategy_config_mapping = {
        'Bollinger Bands (Long)': 'bbands',
        'Bollinger Bands (Short)': 'bbands',
        'Moving Average (Long)': 'ma',
        'Moving Average (Short)': 'ma',
        'MACD Divergence (Long)': 'macd',
        'MACD Divergence (Short)': 'macd',
        'RSI Divergence (Long)': 'rsi',
        'RSI Divergence (Short)': 'rsi',
        'Parabolic SAR (Long)': 'psar',
        'Parabolic SAR (Short)': 'psar',
        'RSI Mean Reversion (Long)': 'rsi_mean_reversion',
        'RSI Mean Reversion (Short)': 'rsi_mean_reversion',
        'Mean Reversion (Long)': 'mean_reversion',
        'Mean Reversion (Short)': 'mean_reversion'
    }
    
    # Create metadata about this optimization run
    metadata = {
        "optimization_date": datetime.now().strftime("%Y-%m-%d"),
        "optimization_settings": config["optimization"],
        "timeframes": config["timeframes"],
        "regime_settings": {
            "ma_short_window": config["regime"]["ma_short_window"],
            "ma_long_window": config["regime"]["ma_long_window"],
            "vol_short_window": config["regime"]["vol_short_window"],
            "avg_vol_window": config["regime"]["avg_vol_window"],
            "target_regimes": target_regimes
        },
        "data_settings": config["data"],
        "analysis_timeframe": timeframe_id
    }
    
    # Group results by Symbol and Strategy (without direction)
    # First, clean strategy names to remove direction
    in_sample_results['Base Strategy'] = in_sample_results['Strategy'].apply(
        lambda x: x.split(' (')[0]
    )
    
    grouped = in_sample_results.groupby(['Symbol', 'Base Strategy'])
    
    for (symbol, strategy), group in grouped:
        # Get the correct strategy name for config lookup
        strategy_config_name = strategy_config_mapping.get(f"{strategy} (Long)")  # Use either Long or Short version
        if strategy_config_name is None:
            print(f"Warning: No config mapping found for strategy {strategy}")
            continue
            
        # Get parameter columns (exclude metrics and metadata)
        param_cols = [col for col in group.columns if col not in [
            'Symbol', 'Strategy', 'Base Strategy', 'Direction', 'Portfolio',
            'Total Return', 'Sharpe Ratio', 'Sortino Ratio',
            'Win Rate', 'Max Drawdown', 'Calmar Ratio',
            'Omega Ratio', 'Profit Factor', 'Expectancy',
            'Total Trades'
        ]]
        
        params_dict = {
            "long": None,
            "short": None,
            "metadata": metadata,
            "strategy_params": config["strategy_params"][strategy_config_name]
        }
        
        # Check for long direction results
        long_results = group[group['Direction'] == 'long']
        if not long_results.empty and long_results['Total Return'].iloc[0] > 0:
            params_dict['long'] = long_results[param_cols].to_dict('records')[0]
        else:
            print(f"Warning: No valid long results for {symbol} {strategy}")
            
        # Check for short direction results
        short_results = group[group['Direction'] == 'short']
        if not short_results.empty and short_results['Total Return'].iloc[0] > 0:
            params_dict['short'] = short_results[param_cols].to_dict('records')[0]
        else:
            print(f"Warning: No valid short results for {symbol} {strategy}")
        
        # Only save if we have at least one valid direction
        if params_dict['long'] is not None or params_dict['short'] is not None:
            # Create filename with regime info
            strategy_name = strategy.lower().replace(' ', '_')
            regime_str = '_'.join(map(str, target_regimes))
            filename = (f"{symbol.lower()}_{strategy_name}_params"
                      f"_regimes_{regime_str}"
                      f"_tf_{timeframe_id}.json")
            
            # Save to JSON
            with open(params_dir / filename, 'w') as f:
                json.dump(params_dict, indent=4, default=str, fp=f)
            
            print(f"Saved optimal parameters for {symbol} {strategy} to {filename}")
        else:
            print(f"Skipping {symbol} {strategy} - no valid results for either direction")

def run_optimized_strategies(
    target_regimes,
    in_sample_data,
    out_sample_data,
    in_sample_regimes,
    out_sample_regimes,
):
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
            )

            # Only add if strategy produced valid results
            if long_pf.total_return > 0:
                in_sample_results.append(
                    create_stats(name, symbol, "long", long_pf, best_long_params)
                )

            # Optimize for short
            short_params = params.copy()
            short_params["direction"] = ["short"]
            best_short_params, short_pf, _ = optimize_strategy(
                func,
                short_params,
                in_sample_data[symbol],
                in_sample_regimes[symbol],
                target_regimes,
            )

            # Only add if strategy produced valid results
            if short_pf.total_return > 0:
                in_sample_results.append(
                    create_stats(name, symbol, "short", short_pf, best_short_params)
                )

    # After optimization but before out-of-sample testing, save the parameters
    timeframe_id = analysis_tf.replace('T', 'm')
    save_optimal_params(pd.DataFrame(in_sample_results), timeframe_id, target_regimes, config)

    # Test optimized parameters on out-of-sample data
    out_sample_results = []
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
        pf = strategy_func(
            symbol_ohlcv_df=out_sample_data[symbol],
            regime_data=out_sample_regimes[symbol],
            allowed_regimes=target_regimes,
            **params,
        )

        # Create stats
        out_sample_results.append(
            create_stats(strategy_name, symbol, direction, pf, params)
        )

    return pd.DataFrame(in_sample_results), pd.DataFrame(out_sample_results)


def format_results_table(results_df):
    """
    Format results dataframe for display with strategies as columns.

    Args:
        results_df (pd.DataFrame): DataFrame containing strategy results

    Returns:
        pd.DataFrame: Formatted DataFrame with:
            - Metrics as rows (Total Return, Sharpe Ratio, etc.)
            - Strategies as columns
            - Formatted values (percentages, decimals)
    """
    # Basic metrics we always want to show
    metric_rows = [
        "Total Return",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Win Rate",
        "Max Drawdown",
        "Calmar Ratio",
        "Total Trades",
    ]

    # Get all parameter columns (they vary by strategy)
    param_rows = [
        col
        for col in results_df.columns
        if col not in metric_rows + ["Strategy", "Direction", "Symbol", "Portfolio"]
    ]

    # Create new DataFrame with strategies as columns
    formatted_df = pd.DataFrame()
    
    for _, row in results_df.iterrows():
        strategy_name = f"{row['Strategy']} ({row['Direction']})"
        
        # Create a series for this strategy's data
        strategy_data = pd.Series(dtype='object')
        
        # Add metrics with corrected Total Return formatting
        strategy_data["Total Return"] = f"{row['Total Return'] * 100:.2f}%"  # Convert decimal to percentage
        strategy_data["Win Rate"] = f"{row['Win Rate']:.2%}"
        strategy_data["Max Drawdown"] = f"{row['Max Drawdown']:.2%}"
        strategy_data["Sharpe Ratio"] = f"{row['Sharpe Ratio']:.2f}"
        strategy_data["Sortino Ratio"] = f"{row['Sortino Ratio']:.2f}"
        strategy_data["Calmar Ratio"] = f"{row['Calmar Ratio']:.2f}"
        strategy_data["Total Trades"] = f"{row['Total Trades']:.0f}"
        
        # Add parameters
        for param in param_rows:
            value = row[param]
            if isinstance(value, (float, np.float32, np.float64)):
                strategy_data[f"param_{param}"] = f"{value:.3f}"
            elif isinstance(value, (int, np.int32, np.int64)):
                strategy_data[f"param_{param}"] = f"{value:.0f}"
            else:
                strategy_data[f"param_{param}"] = str(value)
        
        # Add this strategy as a new column
        formatted_df[strategy_name] = strategy_data

    return formatted_df


if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # Get timeframes from config
    base_tf = config["timeframes"]["base"]  # e.g., "1T"
    analysis_tf = config["timeframes"]["analysis"]  # e.g., "30T"
    regime_tf = config["timeframes"]["regime"]  # e.g., "1D"

    # Load data with specified timeframes
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

    # Calculate split index
    in_sample_pct = config["data"]["in_sample_pct"]
    split_idx = {
        symbol: int(len(data_analysis[symbol]) * in_sample_pct)
        for symbol in ["BTC", "ETH"]
    }

    # Split data
    in_sample_data = {
        symbol: data_analysis[symbol].iloc[:split_idx[symbol]]
        for symbol in ["BTC", "ETH"]
    }
    out_sample_data = {
        symbol: data_analysis[symbol].iloc[split_idx[symbol]:]
        for symbol in ["BTC", "ETH"]
    }

    # Get target regimes from config
    target_regimes = config["regime"]["target_regimes"]

    # Run optimization and get both in-sample and out-of-sample results
    in_sample_results, out_sample_results = run_optimized_strategies(
        target_regimes,
        in_sample_data,
        out_sample_data,
        aligned_regime_data,
        aligned_regime_data,
    )

    # Create results directory and get subdirectories
    results_dir, params_dir, backtests_dir = ensure_results_dir()
    
    # Create timeframe identifier for filenames
    timeframe_id = f"{analysis_tf.replace('T', 'm')}"
    
    # Save results
    for period, results in [('in_sample', in_sample_results), ('out_sample', out_sample_results)]:
        # Save to CSV with strategies as columns
        csv_df = results.drop('Portfolio', axis=1).set_index(['Symbol', 'Strategy', 'Direction'])
        csv_df = csv_df.unstack(['Strategy', 'Direction'])
        
        # Create filepath in backtests directory with timeframe
        filename = f"{period}_results_regimes_{'_'.join(map(str, target_regimes))}_{timeframe_id}.csv"
        filepath = backtests_dir / filename
        
        # Save the CSV
        csv_df.to_csv(filepath)
        print(f"Saved {period} results to {filepath}")
    # Print the date ranges for reference

    print(
        f"In-sample period: {in_sample_data['BTC'].index[0]} to {in_sample_data['BTC'].index[-1]}"
    )
    print(
        f"Out-sample period: {out_sample_data['BTC'].index[0]} to {out_sample_data['BTC'].index[-1]}"
    )

    # Display tabulated results first
    for symbol in ["BTC", "ETH"]:
        print(f"\n{symbol} In-Sample Results:")
        print(
            tabulate(
                format_results_table(
                    in_sample_results[in_sample_results["Symbol"] == symbol]
                ),
                headers="keys",
                tablefmt="pipe",
                floatfmt=".4f",
            )
        )

        print(f"\n{symbol} Out-of-Sample Results:")
        print(
            tabulate(
                format_results_table(
                    out_sample_results[out_sample_results["Symbol"] == symbol]
                ),
                headers="keys",
                tablefmt="pipe",
                floatfmt=".4f",
            )
        )
        print("\n" + "=" * 80)

    # Create subplot figures for each symbol and period
    for symbol in ["BTC", "ETH"]:
        # Get unique strategies
        strategies = in_sample_results["Strategy"].unique()
        n_strategies = len(strategies)

        # Calculate grid dimensions (trying to make it roughly square)
        n_cols = int(np.ceil(np.sqrt(n_strategies)))
        n_rows = int(np.ceil(n_strategies / n_cols))

        # Create in-sample figure
        fig_in = vbt.make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"{strat}" for strat in strategies],
            vertical_spacing=0.1,
        )

        # Plot in-sample results
        for idx, strategy in enumerate(strategies):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            strategy_results = in_sample_results[
                (in_sample_results["Symbol"] == symbol)
                & (in_sample_results["Strategy"] == strategy)
            ]

            for _, result in strategy_results.iterrows():
                pf_fig = result["Portfolio"].plot_cum_returns()
                for trace in pf_fig.data:
                    trace.name = f"{result['Direction']}"
                    fig_in.add_trace(trace, row=row, col=col)

        fig_in.update_layout(
            height=300 * n_rows,
            width=1200,
            title=f"{symbol} In-Sample Performance by Strategy",
            showlegend=True,
        )
        fig_in.show()

        # Create out-of-sample figure
        fig_out = vbt.make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"{strat}" for strat in strategies],
            vertical_spacing=0.1,
        )

        # Plot out-of-sample results
        for idx, strategy in enumerate(strategies):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            strategy_results = out_sample_results[
                (out_sample_results["Symbol"] == symbol)
                & (out_sample_results["Strategy"] == strategy)
            ]

            for _, result in strategy_results.iterrows():
                pf_fig = result["Portfolio"].plot_cum_returns()
                for trace in pf_fig.data:
                    trace.name = f"{result['Direction']}"
                    fig_out.add_trace(trace, row=row, col=col)

        fig_out.update_layout(
            height=300 * n_rows,
            width=1200,
            title=f"{symbol} Out-of-Sample Performance by Strategy",
            showlegend=True,
        )
        fig_out.show()

