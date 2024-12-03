"""
Trading Strategies Module

This module contains implementations of various trading strategies used in the
cryptocurrency trading optimization framework.

Strategies:
- Moving Average Crossover
- MACD Divergence
- RSI Divergence
- Bollinger Bands
- Parabolic SAR
- RSI Mean Reversion
- Mean Reversion
"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np
from numba import njit

# Strategy Regimes
SIMPLE_MA_LONG_ONLY_BTC = [1, 2]
SIMPLE_MA_LONG_ONLY_ETH = [1, 2]
SIMPLE_MA_SHORT_ONLY_BTC = [5, 6]
SIMPLE_MA_SHORT_ONLY_ETH = [5, 6]
SIMPLE_MACD_LONG_ONLY_BTC = [1, 2, 3]
SIMPLE_MACD_LONG_ONLY_ETH = [1, 2]
SIMPLE_MACD_SHORT_ONLY_BTC = [4, 5, 6]
SIMPLE_MACD_SHORT_ONLY_ETH = [5, 6]
SIMPLE_RSI_DIVERGENCE_LONG_ONLY_BTC = [1, 2, 3]
SIMPLE_BBANDS_LIMITS_LONG_ONLY_BTC = [2]
SIMPLE_BBANDS_LIMITS_LONG_ONLY_ETH = [2]
SIMPLE_BBANDS_LIMITS_SHORT_ONLY_BTC = [5, 6]
SIMPLE_BBANDS_LIMITS_SHORT_ONLY_ETH = [5, 6]
SIMPLE_PSAR_LONG_ONLY_BTC = [1, 2]
SIMPLE_PSAR_LONG_ONLY_ETH = [1, 2]
SIMPLE_PSAR_SHORT_ONLY_BTC = [5, 6]
SIMPLE_PSAR_SHORT_ONLY_ETH = [5, 6]


@njit
def rolling_mean_nb(arr, window):
    """
    Calculate the rolling mean of an array with a given window size.
    Uses data only up to yesterday for each calculation.

    Parameters:
    arr (np.ndarray): Input array.
    window (int): Window size for the rolling mean.

    Returns:
    np.ndarray: Array of rolling means.
    """
    out = np.empty_like(arr)
    for i in range(len(arr)):
        if i < window:
            out[i] = np.nan
        else:
            # Use data up to yesterday (i, not i+1)
            out[i] = np.mean(arr[i - window : i])
    return out


@njit
def annualized_volatility_nb(returns, window):
    """
    Calculate the annualized volatility of returns with a given window size.
    Uses data only up to yesterday for each calculation.

    Parameters:
    returns (np.ndarray): Array of returns.
    window (int): Window size for the volatility calculation.

    Returns:
    np.ndarray: Array of annualized volatilities.
    """
    out = np.empty_like(returns)
    for i in range(len(returns)):
        if i < window:
            out[i] = np.nan
        else:
            # Use data up to yesterday (i, not i+1)
            out[i] = np.std(returns[i - window : i]) * np.sqrt(365)
    return out


@njit
def determine_regime_nb(price, ma_short, ma_long, vol_short, avg_vol_threshold):
    """
    Determine the market regime based on yesterday's price and indicators.

    Parameters:
    price (np.ndarray): Array of prices.
    ma_short (np.ndarray): Array of short moving averages.
    ma_long (np.ndarray): Array of long moving averages.
    vol_short (np.ndarray): Array of short volatilities.
    avg_vol_threshold (float): Threshold for average volatility.

    Returns:
    np.ndarray: Array of market regimes.
    """
    regimes = np.empty_like(price, dtype=np.int32)
    for i in range(len(price)):
        if (
            i == 0
            or np.isnan(ma_short[i])
            or np.isnan(ma_long[i])
            or np.isnan(vol_short[i])
        ):
            regimes[i] = -1  # Unknown
        else:
            # Use yesterday's price for comparison
            prev_price = price[i - 1]
            if prev_price > ma_short[i] and prev_price > ma_long[i]:
                if vol_short[i] > avg_vol_threshold:
                    regimes[i] = 1  # Above Avg Vol Bull Trend
                else:
                    regimes[i] = 2  # Below Avg Vol Bull Trend
            elif prev_price < ma_short[i] and prev_price < ma_long[i]:
                if vol_short[i] > avg_vol_threshold:
                    regimes[i] = 5  # Above Avg Vol Bear Trend
                else:
                    regimes[i] = 6  # Below Avg Vol Bear Trend
            else:
                if vol_short[i] > avg_vol_threshold:
                    regimes[i] = 3  # Above Avg Vol Sideways
                else:
                    regimes[i] = 4  # Below Avg Vol Sideways
    return regimes


@njit
def calculate_regimes_nb(
    price, returns, ma_short_window, ma_long_window, vol_short_window, avg_vol_window
):
    """
    Calculate market regimes based on historical data only.

    Parameters:
    price (np.ndarray): Array of prices.
    returns (np.ndarray): Array of returns.
    ma_short_window (int): Window size for the short moving average.
    ma_long_window (int): Window size for the long moving average.
    vol_short_window (int): Window size for the short volatility calculation.
    avg_vol_window (int): Window size for the average volatility calculation.

    Returns:
    np.ndarray: Array of market regimes.
    """
    # Calculate indicators using data up to yesterday
    ma_short = rolling_mean_nb(price, ma_short_window)
    ma_long = rolling_mean_nb(price, ma_long_window)
    vol_short = annualized_volatility_nb(returns, vol_short_window)

    # Calculate average volatility threshold using historical data
    vol_long = annualized_volatility_nb(returns, avg_vol_window)
    avg_vol_threshold = np.nanmean(vol_long)

    # Determine regimes using yesterday's price for comparison
    regimes = determine_regime_nb(
        price, ma_short, ma_long, vol_short, avg_vol_threshold
    )
    return regimes


def calculate_regimes(data_regime, data_analysis, analysis_timeframe):
    """
    Calculate and align market regimes for BTC and ETH across timeframes.

    Args:
        data_regime (dict): Dictionary containing regime timeframe data for each symbol
        data_analysis (dict): Dictionary containing analysis timeframe data for each symbol
        analysis_timeframe (str): Timeframe for analysis (e.g., "30T", "1H")

    Returns:
        dict: Dictionary containing aligned regime data for each symbol
    """
    # Create regime indicators
    RegimeIndicator = vbt.IndicatorFactory(
        class_name="RegimeIndicator",
        input_names=["price", "returns"],
        param_names=[
            "ma_short_window",
            "ma_long_window",
            "vol_short_window",
            "avg_vol_window",
        ],
        output_names=["regimes"],
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


@njit
def psar_nb_with_next(high, low, close, af0=0.02, af_increment=0.02, max_af=0.2):
    """
    Calculate Parabolic SAR with next period's values.

    Parameters:
    high (np.ndarray): Array of high prices
    low (np.ndarray): Array of low prices
    close (np.ndarray): Array of closing prices
    af0 (float): Initial acceleration factor
    af_increment (float): Acceleration factor increment
    max_af (float): Maximum acceleration factor

    Returns:
    tuple: (long, short, af, reversal, next_long, next_short) arrays
    """
    length = len(high)
    long = np.full(length, np.nan)
    short = np.full(length, np.nan)
    af = np.full(length, np.nan)
    reversal = np.zeros(length, dtype=np.int_)
    next_long = np.full(length, np.nan)
    next_short = np.full(length, np.nan)

    # Find first non-NaN index
    start_idx = 0
    while start_idx < length and (
        np.isnan(high[start_idx])
        or np.isnan(low[start_idx])
        or np.isnan(close[start_idx])
    ):
        start_idx += 1

    if start_idx >= length:
        return long, short, af, reversal, next_long, next_short

    # ... rest of existing psar_nb_with_next implementation ...


@njit
def apply_cooldown(entries, exits, cooldown_period):
    cooldown_mask = np.zeros_like(entries, dtype=np.bool_)
    last_exit_index = -cooldown_period  # Initialize to allow first entry

    for i in range(len(entries)):
        if exits[i]:
            last_exit_index = i
        # Apply cooldown mask based on last exit
        if i < last_exit_index + cooldown_period:
            cooldown_mask[i] = True

    return entries & ~cooldown_mask


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
    cooldown_period: int = 24,
):
    """
    Implements a MACD crossover strategy with regime filtering and optional stops.
    """
    # Ensure the indices are aligned
    symbol_ohlcv_df, regime_data = symbol_ohlcv_df.align(regime_data, axis=0, join='inner')
    
    print(f"After alignment - DataFrame shape: {len(symbol_ohlcv_df)}, Regime shape: {len(regime_data)}")
    
    # Calculate MACD
    macd = vbt.MACD.run(
        symbol_ohlcv_df.Close,
        fast_window=fast_window,
        slow_window=slow_window,
        signal_window=signal_window,
    )

    # Generate entry/exit signals
    if direction == "long":
        entries = (macd.macd > macd.signal) & (
            macd.macd.shift(1) <= macd.signal.shift(1)
        )
        exits = (macd.macd < macd.signal) & (
            macd.macd.shift(1) >= macd.signal.shift(1)
        )
    else:  # short
        entries = (macd.macd < macd.signal) & (
            macd.macd.shift(1) >= macd.signal.shift(1)
        )
        exits = (macd.macd > macd.signal) & (
            macd.macd.shift(1) <= macd.signal.shift(1)
        )

    # Fill NaN values with False
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    print(f"After signal generation - Entries: {len(entries)}, Exits: {len(exits)}")

    # Apply regime filter
    entries = entries & regime_data.isin(allowed_regimes)
    regime_exits = ~regime_data.isin(allowed_regimes)
    exits = exits | regime_exits

    print(f"After regime filter - Entries: {len(entries)}, Exits: {len(exits)}")

    # Ensure the length matches before conversion
    assert len(entries) == len(symbol_ohlcv_df), f"Entries length mismatch: {len(entries)} vs {len(symbol_ohlcv_df)}"
    assert len(exits) == len(symbol_ohlcv_df), f"Exits length mismatch: {len(exits)} vs {len(symbol_ohlcv_df)}"

    # Convert to numpy arrays for cooldown
    entries_np = entries.to_numpy()
    exits_np = exits.to_numpy()

    print(f"Before cooldown - Entries: {len(entries_np)}, Exits: {len(exits_np)}")

    # Apply cooldown
    entries_np = apply_cooldown(entries_np, exits_np, cooldown_period)

    print(f"After cooldown - Entries: {len(entries_np)}")

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
                    "entries": entries_np,
                    "exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "short_entries": entries_np,
                    "short_exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
    else:
        if direction == "long":
            pf_kwargs.update({"entries": entries_np, "exits": exits_np})
        else:
            pf_kwargs.update({"short_entries": entries_np, "short_exits": exits_np})

    return vbt.PF.from_signals(**pf_kwargs)


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
    cooldown_period: int = 1440,  # Default 1 day for minute data
):
    """
    Implements a dual moving average crossover strategy with regime filtering,
    optional stops, and cooldown period after exits.
    """
    print(f"Original DataFrame shape: {len(symbol_ohlcv_df)}")
    
    # Calculate MAs
    fast_ma = vbt.MA.run(symbol_ohlcv_df.Close, window=fast_ma).ma
    slow_ma = vbt.MA.run(symbol_ohlcv_df.Close, window=slow_ma).ma
    
    print(f"Fast MA shape: {len(fast_ma)}")
    print(f"Slow MA shape: {len(slow_ma)}")

    # Generate raw entry/exit signals
    long_entries = fast_ma > slow_ma
    long_exits = fast_ma < slow_ma
    short_entries = fast_ma < slow_ma
    short_exits = fast_ma > slow_ma

    print(f"Long entries shape before regime: {len(long_entries)}")
    print(f"Long exits shape before regime: {len(long_exits)}")

    # Add regime filter
    long_entries = long_entries & regime_data.isin(allowed_regimes)
    short_entries = short_entries & regime_data.isin(allowed_regimes)
    regime_exits = ~regime_data.isin(allowed_regimes)

    print(f"Long entries shape after regime: {len(long_entries)}")
    print(f"Regime exits shape: {len(regime_exits)}")

    # Combine all exit conditions
    long_exits = long_exits | regime_exits
    short_exits = short_exits | regime_exits

    # Convert to numpy arrays for cooldown application
    if direction == "long":
        entries_np = long_entries.to_numpy()
        exits_np = long_exits.to_numpy()
    else:
        entries_np = short_entries.to_numpy()
        exits_np = short_exits.to_numpy()

    print(f"Entries shape before cooldown: {len(entries_np)}")
    print(f"Exits shape before cooldown: {len(exits_np)}")

    # Assert shapes before cooldown
    assert len(entries_np) == len(symbol_ohlcv_df), f"Entries shape mismatch before cooldown: {len(entries_np)} vs {len(symbol_ohlcv_df)}"
    assert len(exits_np) == len(symbol_ohlcv_df), f"Exits shape mismatch before cooldown: {len(exits_np)} vs {len(symbol_ohlcv_df)}"

    # Apply cooldown
    entries_np = apply_cooldown(entries_np, exits_np, cooldown_period)

    print(f"Entries shape after cooldown: {len(entries_np)}")

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
                    "entries": entries_np,
                    "exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "short_entries": entries_np,
                    "short_exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
    else:
        if direction == "long":
            pf_kwargs.update({"entries": entries_np, "exits": exits_np})
        else:
            pf_kwargs.update({"short_entries": entries_np, "short_exits": exits_np})

    return vbt.PF.from_signals(**pf_kwargs)


def run_rsi_divergence_strategy_with_stops(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    direction: str = "long",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
    rsi_window: int = 14,
    rsi_threshold: int = 30,
    lookback_window: int = 20,
    cooldown_period: int = 24,
):
    """
    Implements an RSI divergence strategy with regime filtering and optional stops.
    """
    # Ensure the indices are aligned
    symbol_ohlcv_df, regime_data = symbol_ohlcv_df.align(regime_data, axis=0, join='inner')
    
    print(f"After alignment - DataFrame shape: {len(symbol_ohlcv_df)}, Regime shape: {len(regime_data)}")
    
    # Calculate RSI
    rsi = vbt.RSI.run(close=symbol_ohlcv_df.Close, window=rsi_window)

    # Generate entry signals
    if direction == "long":
        entries = (
            (symbol_ohlcv_df.Close == symbol_ohlcv_df.Close.rolling(window=lookback_window).min())
            & (rsi.rsi < rsi_threshold)
            & (rsi.rsi > rsi.rsi.rolling(window=lookback_window).min())
        )
    else:  # short
        entries = (
            (symbol_ohlcv_df.Close == symbol_ohlcv_df.Close.rolling(window=lookback_window).max())
            & (rsi.rsi > 100 - rsi_threshold)
            & (rsi.rsi < rsi.rsi.rolling(window=lookback_window).max())
        )

    # Fill NaN values with False
    entries = entries.fillna(False)
    exits = pd.Series(False, index=symbol_ohlcv_df.index)

    print(f"After signal generation - Entries: {len(entries)}, Exits: {len(exits)}")

    # Apply regime filter
    entries = entries & regime_data.isin(allowed_regimes)
    exits = ~regime_data.isin(allowed_regimes)

    print(f"After regime filter - Entries: {len(entries)}, Exits: {len(exits)}")

    # Ensure the length matches before conversion
    assert len(entries) == len(symbol_ohlcv_df), f"Entries length mismatch: {len(entries)} vs {len(symbol_ohlcv_df)}"
    assert len(exits) == len(symbol_ohlcv_df), f"Exits length mismatch: {len(exits)} vs {len(symbol_ohlcv_df)}"

    # Convert to numpy arrays for cooldown
    entries_np = entries.to_numpy()
    exits_np = exits.to_numpy()

    print(f"Before cooldown - Entries: {len(entries_np)}, Exits: {len(exits_np)}")

    # Apply cooldown
    entries_np = apply_cooldown(entries_np, exits_np, cooldown_period)

    print(f"After cooldown - Entries: {len(entries_np)}")

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
                    "entries": entries_np,
                    "exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "short_entries": entries_np,
                    "short_exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
    else:
        if direction == "long":
            pf_kwargs.update({"entries": entries_np, "exits": exits_np})
        else:
            pf_kwargs.update({"short_entries": entries_np, "short_exits": exits_np})

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
    cooldown_period: int = 24,
):
    """
    Implements a Bollinger Bands strategy with regime filtering and optional stops.
    """
    # Ensure the indices are aligned
    symbol_ohlcv_df, regime_data = symbol_ohlcv_df.align(regime_data, axis=0, join='inner')
    
    print(f"After alignment - DataFrame shape: {len(symbol_ohlcv_df)}, Regime shape: {len(regime_data)}")
    
    # Calculate Bollinger Bands
    bbands = vbt.BBANDS.run(
        close=symbol_ohlcv_df.Close, 
        window=bb_window, 
        alpha=bb_alpha
    )

    # Generate entry signals
    if direction == "long":
        entries = (symbol_ohlcv_df.Close < bbands.lower)
    else:  # short
        entries = (symbol_ohlcv_df.Close > bbands.upper)

    # Fill NaN values with False
    entries = entries.fillna(False)
    exits = pd.Series(False, index=symbol_ohlcv_df.index)

    print(f"After signal generation - Entries: {len(entries)}, Exits: {len(exits)}")

    # Apply regime filter
    entries = entries & regime_data.isin(allowed_regimes)
    exits = ~regime_data.isin(allowed_regimes)

    print(f"After regime filter - Entries: {len(entries)}, Exits: {len(exits)}")

    # Ensure the length matches before conversion
    assert len(entries) == len(symbol_ohlcv_df), f"Entries length mismatch: {len(entries)} vs {len(symbol_ohlcv_df)}"
    assert len(exits) == len(symbol_ohlcv_df), f"Exits length mismatch: {len(exits)} vs {len(symbol_ohlcv_df)}"

    # Convert to numpy arrays for cooldown
    entries_np = entries.to_numpy()
    exits_np = exits.to_numpy()

    print(f"Before cooldown - Entries: {len(entries_np)}, Exits: {len(exits_np)}")

    # Apply cooldown
    entries_np = apply_cooldown(entries_np, exits_np, cooldown_period)

    print(f"After cooldown - Entries: {len(entries_np)}")

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
                    "entries": entries_np,
                    "exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "short_entries": entries_np,
                    "short_exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
    else:
        if direction == "long":
            pf_kwargs.update({"entries": entries_np, "exits": exits_np})
        else:
            pf_kwargs.update({"short_entries": entries_np, "short_exits": exits_np})

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
    cooldown_period: int = 24,
):
    """
    Implements a Parabolic SAR strategy with regime filtering and optional stops.
    """
    # Ensure the indices are aligned
    symbol_ohlcv_df, regime_data = symbol_ohlcv_df.align(regime_data, axis=0, join='inner')
    
    print(f"After alignment - DataFrame shape: {len(symbol_ohlcv_df)}, Regime shape: {len(regime_data)}")
    
    # Calculate PSAR
    long, short, _, _, _, _ = psar_nb_with_next(
        symbol_ohlcv_df.High.values,
        symbol_ohlcv_df.Low.values,
        symbol_ohlcv_df.Close.values,
        af0=af0,
        af_increment=af_increment,
        max_af=max_af,
    )

    # Generate entry signals
    if direction == "long":
        entries = pd.Series(
            long < symbol_ohlcv_df.Low.values, 
            index=symbol_ohlcv_df.index
        )
    else:  # short
        entries = pd.Series(
            short > symbol_ohlcv_df.High.values, 
            index=symbol_ohlcv_df.index
        )

    # Fill NaN values with False
    entries = entries.fillna(False)
    exits = pd.Series(False, index=symbol_ohlcv_df.index)

    print(f"After signal generation - Entries: {len(entries)}, Exits: {len(exits)}")

    # Apply regime filter
    entries = entries & regime_data.isin(allowed_regimes)
    exits = ~regime_data.isin(allowed_regimes)

    print(f"After regime filter - Entries: {len(entries)}, Exits: {len(exits)}")

    # Ensure the length matches before conversion
    assert len(entries) == len(symbol_ohlcv_df), f"Entries length mismatch: {len(entries)} vs {len(symbol_ohlcv_df)}"
    assert len(exits) == len(symbol_ohlcv_df), f"Exits length mismatch: {len(exits)} vs {len(symbol_ohlcv_df)}"

    # Convert to numpy arrays for cooldown
    entries_np = entries.to_numpy()
    exits_np = exits.to_numpy()

    print(f"Before cooldown - Entries: {len(entries_np)}, Exits: {len(exits_np)}")

    # Apply cooldown
    entries_np = apply_cooldown(entries_np, exits_np, cooldown_period)

    print(f"After cooldown - Entries: {len(entries_np)}")

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
                    "entries": entries_np,
                    "exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "short_entries": entries_np,
                    "short_exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
    else:
        if direction == "long":
            pf_kwargs.update({"entries": entries_np, "exits": exits_np})
        else:
            pf_kwargs.update({"short_entries": entries_np, "short_exits": exits_np})

    return vbt.PF.from_signals(**pf_kwargs)


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
    cooldown_period: int = 24,
):
    """
    Implements an RSI mean reversion strategy with regime filtering and optional stops.
    """
    # Ensure the indices are aligned
    symbol_ohlcv_df, regime_data = symbol_ohlcv_df.align(regime_data, axis=0, join='inner')
    
    print(f"After alignment - DataFrame shape: {len(symbol_ohlcv_df)}, Regime shape: {len(regime_data)}")
    
    rsi = vbt.RSI.run(close=symbol_ohlcv_df.Close, window=rsi_window)

    # Generate entry signals
    if direction == "long":
        entries = (rsi.rsi < rsi_lower)
    else:  # short
        entries = (rsi.rsi > rsi_upper)

    # Fill NaN values with False
    entries = entries.fillna(False)
    exits = pd.Series(False, index=symbol_ohlcv_df.index)

    print(f"After signal generation - Entries: {len(entries)}, Exits: {len(exits)}")

    # Apply regime filter
    entries = entries & regime_data.isin(allowed_regimes)
    exits = ~regime_data.isin(allowed_regimes)

    print(f"After regime filter - Entries: {len(entries)}, Exits: {len(exits)}")

    # Ensure the length matches before conversion
    assert len(entries) == len(symbol_ohlcv_df), f"Entries length mismatch: {len(entries)} vs {len(symbol_ohlcv_df)}"
    assert len(exits) == len(symbol_ohlcv_df), f"Exits length mismatch: {len(exits)} vs {len(symbol_ohlcv_df)}"

    # Convert to numpy arrays for cooldown
    entries_np = entries.to_numpy()
    exits_np = exits.to_numpy()

    print(f"Before cooldown - Entries: {len(entries_np)}, Exits: {len(exits_np)}")

    # Apply cooldown
    entries_np = apply_cooldown(entries_np, exits_np, cooldown_period)

    print(f"After cooldown - Entries: {len(entries_np)}")

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
                    "entries": entries_np,
                    "exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "short_entries": entries_np,
                    "short_exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
    else:
        if direction == "long":
            pf_kwargs.update({"entries": entries_np, "exits": exits_np})
        else:
            pf_kwargs.update({"short_entries": entries_np, "short_exits": exits_np})

    return vbt.PF.from_signals(**pf_kwargs)


def mean_reversion_strategy(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: list,
    direction: str = "long",
    bb_window: int = 21,
    bb_alpha: float = 2.0,
    timeframe_1: str = "4H",
    timeframe_2: str = "24H",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 3.0,
    fees: float = 0.001,
    cooldown_period: int = 24,
):
    """
    Implements a dual timeframe Bollinger Bands mean reversion strategy with regime filtering and optional stops.
    """
    # Ensure the indices are aligned
    symbol_ohlcv_df, regime_data = symbol_ohlcv_df.align(regime_data, axis=0, join='inner')
    
    print(f"After alignment - DataFrame shape: {len(symbol_ohlcv_df)}, Regime shape: {len(regime_data)}")
    
    # Calculate Bollinger Bands for both timeframes
    bbands_tf1 = vbt.BBANDS.run(
        symbol_ohlcv_df.Close,
        window=bb_window,
        alpha=bb_alpha,
    )
    bbands_tf2 = vbt.BBANDS.run(
        symbol_ohlcv_df.Close,
        window=bb_window * 6,  # Longer timeframe
        alpha=bb_alpha,
    )

    # Generate entry conditions
    if direction == "long":
        entries = (
            (symbol_ohlcv_df.Close < bbands_tf2.middle) & 
            (symbol_ohlcv_df.Close < bbands_tf1.lower)
        ) | (
            (symbol_ohlcv_df.Close > bbands_tf2.lower) & 
            (symbol_ohlcv_df.Close < bbands_tf1.lower)
        )
    else:  # short
        entries = (
            (symbol_ohlcv_df.Close > bbands_tf2.middle) & 
            (symbol_ohlcv_df.Close > bbands_tf1.upper)
        ) | (
            (symbol_ohlcv_df.Close < bbands_tf2.upper) & 
            (symbol_ohlcv_df.Close > bbands_tf1.upper)
        )

    # Fill NaN values with False
    entries = entries.fillna(False)
    exits = pd.Series(False, index=symbol_ohlcv_df.index)

    print(f"After signal generation - Entries: {len(entries)}, Exits: {len(exits)}")

    # Apply regime filter
    entries = entries & regime_data.isin(allowed_regimes)
    exits = ~regime_data.isin(allowed_regimes)

    print(f"After regime filter - Entries: {len(entries)}, Exits: {len(exits)}")

    # Ensure the length matches before conversion
    assert len(entries) == len(symbol_ohlcv_df), f"Entries length mismatch: {len(entries)} vs {len(symbol_ohlcv_df)}"
    assert len(exits) == len(symbol_ohlcv_df), f"Exits length mismatch: {len(exits)} vs {len(symbol_ohlcv_df)}"

    # Convert to numpy arrays for cooldown
    entries_np = entries.to_numpy()
    exits_np = exits.to_numpy()

    print(f"Before cooldown - Entries: {len(entries_np)}, Exits: {len(exits_np)}")

    # Apply cooldown
    entries_np = apply_cooldown(entries_np, exits_np, cooldown_period)

    print(f"After cooldown - Entries: {len(entries_np)}")

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
                    "entries": entries_np,
                    "exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
        else:
            pf_kwargs.update(
                {
                    "short_entries": entries_np,
                    "short_exits": exits_np,
                    "sl_stop": symbol_ohlcv_df.Close + atr_multiplier * atr,
                    "tp_stop": symbol_ohlcv_df.Close - atr_multiplier * atr,
                    "delta_format": "target",
                }
            )
    else:
        if direction == "long":
            pf_kwargs.update({"entries": entries_np, "exits": exits_np})
        else:
            pf_kwargs.update({"short_entries": entries_np, "short_exits": exits_np})

    return vbt.PF.from_signals(**pf_kwargs)


def create_test_data(size: int = 5000) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Create test data for strategy testing.
    """
    # Create random OHLCV data
    close = np.random.random(size)
    high = close + np.random.random(size) * 0.1
    low = close - np.random.random(size) * 0.1
    open_prices = close + np.random.random(size) * 0.05 - 0.025

    ohlcv_data = pd.DataFrame({
        'Close': close,
        'High': high,
        'Low': low,
        'Open': open_prices,
    })

    # Create regime data (1,2 for bull, 5,6 for bear)
    regime_data = pd.Series(np.random.choice([1, 2, 5, 6], size=size))
    allowed_regimes = [1, 2]  # Bull market regimes

    return ohlcv_data, regime_data, allowed_regimes


def test_strategy(strategy_func, **kwargs):
    """
    Test a strategy with sample data and print results.
    """
    # Create test data
    ohlcv_data, regime_data, allowed_regimes = create_test_data()
    
    try:
        # Run strategy
        portfolio = strategy_func(
            symbol_ohlcv_df=ohlcv_data,
            regime_data=regime_data,
            allowed_regimes=allowed_regimes,
            **kwargs
        )
        
        # Print basic stats
        print(f"\nTesting {strategy_func.__name__}:")
        print(f"Data shape: {ohlcv_data.shape}")
        print("\nStrategy Stats:")
        print(portfolio.stats())
        
    except Exception as e:
        print(f"Error testing {strategy_func.__name__}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    ohlcv_data, regime_data, allowed_regimes = create_test_data()
    
    # Test MA strategy
    print("\nTesting MA Strategy:")
    portfolio = run_ma_strategy_with_stops(ohlcv_data, regime_data, allowed_regimes)
    print(portfolio.stats())
    macd_strategy_with_stops = run_macd_divergence_strategy_with_stops(ohlcv_data, regime_data, allowed_regimes)
    print(macd_strategy_with_stops.stats())
    
