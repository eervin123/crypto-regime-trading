"""
Simple tester for alternative trading strategies.
Uses the same data pipeline as recent_strategy_analysis.py but focused on testing
the new strategies from alt_strategies.py
"""

# Data analysis libraries
import pandas as pd
import vectorbtpro as vbt

# Local imports
# from alt_strategies import volatility_breakout, volume_profile_range
from scripts.regimes_multi_strat_pf import calculate_regimes_nb


from typing import List
import numpy as np

def volatility_breakout(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    allowed_regimes: List[int] = [1, 2],
    ma_window: int = 24,  # 24-hour MA
    lookback_window: int = 48,  # 2-day volatility lookback
    vol_multiplier: float = 1.5,
    direction: str = "both",
    fees: float = 0.001,
    use_sl_tp: bool = False,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
) -> vbt.Portfolio:
    """
    Trades breakouts based on moving average trends and volatility expansion.
    """
    # Calculate moving average and its slope
    ma = symbol_ohlcv_df['Close'].rolling(window=ma_window).mean()
    ma_slope = ma.diff(lookback_window)  #uptrend or downtrend
    
    # Calculate True Range and volatility
    atr = vbt.ATR.run(
        symbol_ohlcv_df['High'],
        symbol_ohlcv_df['Low'],
        symbol_ohlcv_df['Close'],
        window=lookback_window
    ).atr
    
    avg_vol = atr.rolling(window=lookback_window).mean()
    vol_expansion = atr > (avg_vol * vol_multiplier)
    
    # Generate directional signals
    long_signals = (
        vol_expansion &  # Volatility expansion
        (ma_slope > 0) &  # Upward trend
        regime_data.isin(allowed_regimes)  # Correct regime
    )
    
    short_signals = (
        vol_expansion &  # Volatility expansion
        (ma_slope < 0) &  # Downward trend
        regime_data.isin(allowed_regimes)  # Correct regime
    )
    
    # Portfolio arguments
    pf_kwargs = {
        "close": symbol_ohlcv_df['Close'],
        "open": symbol_ohlcv_df['Open'],
        "high": symbol_ohlcv_df['High'],
        "low": symbol_ohlcv_df['Low'],
        "fees": fees,
        "size": 1.0,
        "init_cash": 100.0,
        "freq": "1h"
    }
    
    if direction == "long":
        pf_kwargs["entries"] = long_signals
        pf_kwargs["exits"] = short_signals  # Exit longs on short signals
    elif direction == "short":
        pf_kwargs["short_entries"] = short_signals
        pf_kwargs["short_exits"] = long_signals  # Exit shorts on long signals
    else:  # both
        pf_kwargs.update({
            "entries": long_signals,
            "exits": short_signals,  # Long positions reverse on short signals
            "short_entries": short_signals,
            "short_exits": long_signals  # Short positions reverse on long signals
        })
    
    if use_sl_tp:
        sl_stop = atr_multiplier * atr
        tp_stop = 2 * atr_multiplier * atr
        pf_kwargs.update({
            "sl_stop": sl_stop,
            "tp_stop": tp_stop,
            "delta_format": "absolute"
        })
    else:
        # Use td_stop for time-based exits
        pf_kwargs["td_stop"] = 168  # Exit after 24 periods (hours)
        pf_kwargs["time_delta_format"] = "rows"  # Specify that td_stop is in terms of rows
    
    return vbt.Portfolio.from_signals(**pf_kwargs)

def volume_profile_range(
    symbol_ohlcv_df: pd.DataFrame,
    regime_data: pd.Series,
    profile_period: str = "1D",
    value_area_pct: float = 0.70,
    direction: str = "both",
    use_sl_tp: bool = False,
    fees: float = 0.001,
    allowed_regimes: List[int] = [1, 2],
    min_volume_threshold: float = 0.5,
) -> vbt.Portfolio:
    """
    Implements a volume profile range trading strategy.
    """
    # Calculate price levels for volume profile (fewer levels for speed)
    price_levels = np.linspace(
        symbol_ohlcv_df['Low'].min(),
        symbol_ohlcv_df['High'].max(),
        50  # Reduced from 100 for better performance
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
            
        # Calculate POC and value areas
        poc_idx = np.argmax(volumes)
        poc = price_levels[poc_idx]
        
        sorted_indices = np.argsort(volumes)[::-1]
        cumsum_volume = np.cumsum(volumes[sorted_indices])
        value_area_threshold = total_volume * value_area_pct
        
        value_area_indices = sorted_indices[cumsum_volume <= value_area_threshold]
        value_area_prices = price_levels[value_area_indices]
        
        if len(value_area_prices) > 0:
            vah = np.max(value_area_prices)
            val = np.min(value_area_prices)
        else:
            vah = volume_profile.loc[mask, 'VAH'].iloc[-1] if len(volume_profile.loc[mask, 'VAH']) > 0 else poc
            val = volume_profile.loc[mask, 'VAL'].iloc[-1] if len(volume_profile.loc[mask, 'VAL']) > 0 else poc
        
        volume_profile.loc[mask, 'POC'] = poc
        volume_profile.loc[mask, 'VAH'] = vah
        volume_profile.loc[mask, 'VAL'] = val
    
    # Forward fill values
    volume_profile = volume_profile.ffill()
    
    # Generate trading signals with volume filter
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
        "size": 1.0,
        "init_cash": 100.0,
        "freq": "1h"
    }
    
    if direction == "both":
        pf_kwargs.update({
            "entries": long_signals,
            "exits": short_signals,
            "short_entries": short_signals,
            "short_exits": long_signals
        })
    
    if use_sl_tp:
        atr = vbt.ATR.run(
            symbol_ohlcv_df['High'],
            symbol_ohlcv_df['Low'],
            symbol_ohlcv_df['Close'],
            window=atr_window
        ).atr
        
        sl_stop = atr_multiplier * atr
        tp_stop = 2 * atr_multiplier * atr
        pf_kwargs.update({
            "sl_stop": sl_stop,
            "tp_stop": tp_stop,
            "delta_format": "absolute"
        })
    else:
        pf_kwargs["td_stop"] = 24  # Exit after 24 periods
        pf_kwargs["time_delta_format"] = "rows"
    
    return vbt.Portfolio.from_signals(**pf_kwargs)

def volume_profile_strategy(
    symbol_ohlcv_df: pd.DataFrame,
    profile_period: str = "1D",
    value_area_pct: float = 0.70,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    fees: float = 0.001
) -> vbt.Portfolio:
    """
    Implements a volume profile mean reversion strategy.
    
    Args:
        symbol_ohlcv_df (pd.DataFrame): OHLCV price data
        profile_period (str): Period for volume profile calculation ('1D', '4H', etc.)
        value_area_pct (float): Percentage of volume to include in value area
        atr_window (int): Window for ATR calculation
        atr_multiplier (float): Multiplier for ATR-based stops
        fees (float): Trading fees
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
        
        # Handle empty value areas
        if len(value_area_prices) > 0:
            vah = np.max(value_area_prices)  # Value Area High
            val = np.min(value_area_prices)  # Value Area Low
        else:
            # If no value area prices, use POC as both high and low
            vah = poc
            val = poc
        
        # Assign values to the period
        volume_profile.loc[mask, 'POC'] = poc
        volume_profile.loc[mask, 'VAH'] = vah
        volume_profile.loc[mask, 'VAL'] = val
    
    # Forward fill values
    volume_profile = volume_profile.ffill()
    
    # Generate trading signals
    long_entries = symbol_ohlcv_df['Close'] <= volume_profile['VAL']
    short_entries = symbol_ohlcv_df['Close'] >= volume_profile['VAH']
    
    # Portfolio arguments
    pf_kwargs = {
        "close": symbol_ohlcv_df['Close'],
        "open": symbol_ohlcv_df['Open'],
        "high": symbol_ohlcv_df['High'],
        "low": symbol_ohlcv_df['Low'],
        "fees": fees,
        "entries": long_entries,
        "short_entries": short_entries
    }
    
    # Add stop-loss and take-profit
    atr = vbt.ATR.run(
        symbol_ohlcv_df['High'],
        symbol_ohlcv_df['Low'],
        symbol_ohlcv_df['Close'],
        window=atr_window
    ).atr
    
    pf_kwargs.update({
        "sl_stop": atr_multiplier * atr,
        "tp_stop": volume_profile['POC'],  # Target POC for mean reversion
        "delta_format": "absolute"
    })
    
    # Create portfolio
    pf = vbt.Portfolio.from_signals(**pf_kwargs)
    
    return pf

def load_data(base_timeframe="1T", analysis_timeframe="1H", regime_timeframe="1D"):
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

def test_strategies():
    # Load data at different timeframes
    data_analysis, data_regime = load_data(
        base_timeframe="1T",
        analysis_timeframe="1H",
        regime_timeframe="1D"
    )
    
    # Test volume profile strategy
    print("\nTesting Volume Profile strategy...")
    vol_profile_results = {}
    for symbol in ["BTC", "ETH"]:
        vol_profile_results[symbol] = volume_profile_strategy(
            data_analysis[symbol],
            profile_period="1D",
            value_area_pct=0.70,
            atr_window=14,
            atr_multiplier=2.0,
            fees=0.001
        )
        
        print(f"\n{symbol} Volume Profile Results:")
        print(vol_profile_results[symbol].stats())
        
        fig = vol_profile_results[symbol].plot()
        fig.show()

if __name__ == "__main__":
    test_strategies() 