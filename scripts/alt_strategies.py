"""
Alternative trading strategies focused on non-trending market conditions.
Complements the strategies in optuna_multistrat.py.

Strategies:
- Volatility breakout
- Volume profile range trading
"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np
from typing import List


def volatility_breakout(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: List[int] = [1, 2],  # Default to trending regimes
    lookback_window: int = 20,
    vol_multiplier: float = 1.5,
    direction: str = "both",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
) -> vbt.Portfolio:
    """
    Trades breakouts from low volatility periods.
    
    Args:
        symbol_ohlcv_df (pd.DataFrame): OHLCV price data
        regime_data (pd.Series): Market regime labels
        allowed_regimes (List[int]): List of regime labels to trade in
        lookback_window (int): Window for volatility calculation
        vol_multiplier (float): Multiplier for volatility threshold
        direction (str): Trading direction ('long', 'short', or 'both')
        use_sl_tp (bool): Whether to use stop-loss/take-profit
        atr_window (int): ATR calculation window
        atr_multiplier (float): Multiplier for ATR-based stops
        fees (float): Trading fees
    """
    # Calculate True Range using ATR
    atr = vbt.ATR.run(
        symbol_ohlcv_df['High'],
        symbol_ohlcv_df['Low'],
        symbol_ohlcv_df['Close'],
        window=1  # Use a window of 1 to get the true range
    )
    tr = atr.atr  # This gives us the true range
    
    avg_tr = tr.rolling(window=lookback_window).mean()
    current_tr = tr / avg_tr
    
    # Generate signals
    vol_expansion = current_tr > vol_multiplier
    
    # Portfolio arguments
    pf_kwargs = {
        "close": symbol_ohlcv_df['Close'],
        "open": symbol_ohlcv_df['Open'],
        "high": symbol_ohlcv_df['High'],
        "low": symbol_ohlcv_df['Low'],
        "fees": fees,
    }
    
    if use_sl_tp:
        atr = vbt.ATR.run(
            symbol_ohlcv_df['High'],
            symbol_ohlcv_df['Low'],
            symbol_ohlcv_df['Close'],
            window=atr_window
        ).atr
        
        if direction in ["long", "both"]:
            pf_kwargs.update({
                "sl_stop": symbol_ohlcv_df['Close'] - atr_multiplier * atr,
                "tp_stop": symbol_ohlcv_df['Close'] + atr_multiplier * atr,
                "delta_format": "target",
            })
        if direction in ["short", "both"]:
            pf_kwargs.update({
                "sl_stop": symbol_ohlcv_df['Close'] + atr_multiplier * atr,
                "tp_stop": symbol_ohlcv_df['Close'] - atr_multiplier * atr,
                "delta_format": "target",
            })
        
    # Add regime filtering
    regime_mask = regime_data.isin(allowed_regimes)
    vol_expansion = vol_expansion & regime_mask
        
    if direction == "long":
        pf_kwargs["entries"] = vol_expansion
    elif direction == "short":
        pf_kwargs["short_entries"] = vol_expansion
    else:  # both
        pf_kwargs.update({
            "entries": vol_expansion,
            "short_entries": vol_expansion
        })
    
    return vbt.Portfolio.from_signals(**pf_kwargs)

def volume_profile_range(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    profile_period: str = "1D",
    value_area_pct: float = 0.70,
    direction: str = "both",
    use_sl_tp: bool = True,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001,
    allowed_regimes: List[int] = [3, 4],  # Default to ranging regimes
    min_volume_threshold: float = 0.1,  # Minimum volume percentile to consider
) -> vbt.Portfolio:
    """
    Implements a volume profile range trading strategy.
    
    Args:
        symbol_ohlcv_df (pd.DataFrame): OHLCV price data
        regime_data (pd.Series): Market regime labels
        profile_period (str): Timeframe for volume profile calculation
        value_area_pct (float): Percentage of volume to consider for value area
        direction (str): Trading direction ('long', 'short', or 'both')
        use_sl_tp (bool): Whether to use stop-loss/take-profit
        atr_window (int): ATR calculation window
        atr_multiplier (float): Multiplier for ATR-based stops
        fees (float): Trading fees
        allowed_regimes (List[int]): List of regime labels to trade in
        min_volume_threshold (float): Minimum volume percentile to consider
    """
    # Calculate price levels for volume profile
    price_levels = np.linspace(
        symbol_ohlcv_df['Low'].min(),
        symbol_ohlcv_df['High'].max(),
        100  # Number of price levels
    )
    
    volume_profile = pd.DataFrame(index=symbol_ohlcv_df.index, columns=['POC', 'VAH', 'VAL'])
    
    # Calculate volume profile metrics for each period
    for period_start in pd.date_range(
        start=symbol_ohlcv_df.index[0],
        end=symbol_ohlcv_df.index[-1],
        freq=profile_period
    ):
        period_end = period_start + pd.Timedelta(profile_period)
        mask = (symbol_ohlcv_df.index >= period_start) & (symbol_ohlcv_df.index < period_end)
        period_data = symbol_ohlcv_df[mask]
        
        if len(period_data) == 0:
            continue
            
        # Calculate volume for each price level
        volumes = []
        for price in price_levels:
            vol = period_data[
                (period_data['Low'] <= price) & 
                (period_data['High'] >= price)
            ]['Volume'].sum()
            volumes.append(vol)
            
        volumes = np.array(volumes)
        total_volume = volumes.sum()
        
        if total_volume == 0:
            continue
            
        # Calculate POC (Point of Control)
        poc_idx = np.argmax(volumes)
        poc = price_levels[poc_idx]
        
        # Calculate Value Area
        sorted_indices = np.argsort(volumes)[::-1]  # Sort volume indices in descending order
        cumsum_volume = np.cumsum(volumes[sorted_indices])
        value_area_threshold = total_volume * value_area_pct
        
        # Find price levels within value area
        value_area_indices = sorted_indices[cumsum_volume <= value_area_threshold]
        value_area_prices = price_levels[value_area_indices]
        
        vah = np.max(value_area_prices)  # Value Area High
        val = np.min(value_area_prices)  # Value Area Low
        
        # Assign values to the period
        volume_profile.loc[mask, 'POC'] = poc
        volume_profile.loc[mask, 'VAH'] = vah
        volume_profile.loc[mask, 'VAL'] = val
    
    # Forward fill values
    volume_profile = volume_profile.ffill()
    
    # Generate trading signals
    long_entries = (
        (symbol_ohlcv_df['Close'] <= volume_profile['VAL']) & 
        (symbol_ohlcv_df['Volume'] >= symbol_ohlcv_df['Volume'].rolling(20).quantile(min_volume_threshold))
    )
    
    short_entries = (
        (symbol_ohlcv_df['Close'] >= volume_profile['VAH']) &
        (symbol_ohlcv_df['Volume'] >= symbol_ohlcv_df['Volume'].rolling(20).quantile(min_volume_threshold))
    )
    
    # Apply regime filter
    regime_mask = regime_data.isin(allowed_regimes)
    long_entries = long_entries & regime_mask
    short_entries = short_entries & regime_mask
    
    # Portfolio arguments
    pf_kwargs = {
        "close": symbol_ohlcv_df['Close'],
        "open": symbol_ohlcv_df['Open'],
        "high": symbol_ohlcv_df['High'],
        "low": symbol_ohlcv_df['Low'],
        "fees": fees,
    }
    
    if use_sl_tp:
        atr = vbt.ATR.run(
            symbol_ohlcv_df['High'],
            symbol_ohlcv_df['Low'],
            symbol_ohlcv_df['Close'],
            window=atr_window
        ).atr
        
        if direction in ["long", "both"]:
            pf_kwargs.update({
                "sl_stop": symbol_ohlcv_df['Close'] - atr_multiplier * atr,
                "tp_stop": volume_profile['POC'],  # Target POC for mean reversion
                "delta_format": "target",
            })
        if direction in ["short", "both"]:
            pf_kwargs.update({
                "sl_stop": symbol_ohlcv_df['Close'] + atr_multiplier * atr,
                "tp_stop": volume_profile['POC'],  # Target POC for mean reversion
                "delta_format": "target",
            })
    
    # Set entries based on direction
    if direction == "long":
        pf_kwargs["entries"] = long_entries
    elif direction == "short":
        pf_kwargs["short_entries"] = short_entries
    else:  # both
        pf_kwargs.update({
            "entries": long_entries,
            "short_entries": short_entries
        })
    
    return vbt.Portfolio.from_signals(**pf_kwargs)
