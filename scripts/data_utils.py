"""
Data Utilities Module

This module handles data loading and preprocessing functions.
"""

import vectorbtpro as vbt

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
        "ETH": data.resample(analysis_timeframe).data["ETHUSDT"],
    }

    # Regime timeframe data with returns
    data_regime = {
        "BTC": data.resample(regime_timeframe).data["BTCUSDT"],
        "ETH": data.resample(regime_timeframe).data["ETHUSDT"],
    }

    # Add returns for regime calculation
    for symbol in ["BTC", "ETH"]:
        data_regime[symbol]["Return"] = data_regime[symbol]["Close"].pct_change()

    return data_analysis, data_regime

def validate_timeframe_params(tf1_list, tf2_list):
    """
    Validate that timeframe lists have no overlap and tf2 values are larger than tf1.
    
    Args:
        tf1_list (list): List of first timeframe values
        tf2_list (list): List of second timeframe values
    
    Raises:
        ValueError: If timeframe validation fails
    """
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

    max_tf1 = max(to_hours(tf) for tf in tf1_list)
    min_tf2 = min(to_hours(tf) for tf in tf2_list)

    if min_tf2 <= max_tf1:
        raise ValueError(
            "All timeframe_2 values must be larger than timeframe_1 values"
        ) 