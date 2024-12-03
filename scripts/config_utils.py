"""
Configuration Utilities Module

This module handles configuration loading and directory management for the trading framework.
"""

import yaml
from pathlib import Path
from strategies import (
    run_ma_strategy_with_stops,
    run_macd_divergence_strategy_with_stops,
    run_rsi_divergence_strategy_with_stops,
    run_bbands_strategy_with_stops,
    run_psar_strategy_with_stops,
    run_rsi_mean_reversion_strategy,
    mean_reversion_strategy,
)

STRATEGY_MAPPING = {
    "Moving Average": ("ma", run_ma_strategy_with_stops),
    "MACD Divergence": ("macd", run_macd_divergence_strategy_with_stops),
    "RSI Divergence": ("rsi", run_rsi_divergence_strategy_with_stops),
    "Bollinger Bands": ("bbands", run_bbands_strategy_with_stops),
    "Parabolic SAR": ("psar", run_psar_strategy_with_stops),
    "RSI Mean Reversion": ("rsi_mean_reversion", run_rsi_mean_reversion_strategy),
    "Mean Reversion": ("mean_reversion", mean_reversion_strategy),
}


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
        config = yaml.safe_load(f)
    validate_config(config)  # Validate and convert cooldown periods to integers
    return config


def ensure_results_dir():
    """
    Create results directory and subdirectories if they don't exist.
    
    Returns:
        tuple: Containing:
            - results_dir (Path): Main results directory
            - params_dir (Path): Directory for parameter files
            - backtests_dir (Path): Directory for backtest results
    """
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create params subdirectory
    params_dir = results_dir / "params"
    params_dir.mkdir(exist_ok=True)

    # Create backtests subdirectory
    backtests_dir = results_dir / "backtests"
    backtests_dir.mkdir(exist_ok=True)

    return results_dir, params_dir, backtests_dir


def get_timeframe_id(timeframe):
    """
    Convert timeframe string to standardized identifier for filenames.
    
    Args:
        timeframe (str): Timeframe string (e.g., "30T", "1H", "4H")
        
    Returns:
        str: Standardized timeframe identifier (e.g., "30min", "1h", "4h")
    """
    return timeframe.replace("T", "min")


def validate_config(config):
    """
    Validate configuration settings.
    
    Args:
        config (dict): Configuration dictionary
        
    Raises:
        ValueError: If required settings are missing or invalid
    """
    required_sections = [
        "timeframes",
        "data",
        "regime",
        "optimization",
        "objective",
        "strategy_params"
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate timeframes
    if not all(tf in config["timeframes"] for tf in ["base", "analysis", "regime"]):
        raise ValueError("Missing required timeframe settings")
    
    # Validate optimization settings
    if "n_trials" not in config["optimization"]:
        raise ValueError("Missing n_trials in optimization settings")
    
    # Validate cooldown periods
    if "cooldown_periods" not in config["optimization"]:
        raise ValueError("Missing cooldown_periods in optimization settings")
    
    # Validate objective settings
    if "metrics" not in config["objective"]:
        raise ValueError("Missing metrics in objective settings")
    
    # Validate strategy parameters
    required_strategies = [
        "ma", "macd", "rsi", "bbands", "psar",
        "rsi_mean_reversion", "mean_reversion"
    ]
    for strategy in required_strategies:
        if strategy not in config["strategy_params"]:
            raise ValueError(f"Missing parameters for strategy: {strategy}")
        # Validate that each strategy has cooldown_period parameter
        if "cooldown_period" not in config["strategy_params"][strategy]:
            raise ValueError(f"Missing cooldown_period parameter for strategy: {strategy}")
        # Ensure cooldown_period ranges are integers
        cooldown_range = config["strategy_params"][strategy]["cooldown_period"]
        if not isinstance(cooldown_range, list) or len(cooldown_range) != 2:
            raise ValueError(f"Invalid cooldown_period range for strategy: {strategy}")
        config["strategy_params"][strategy]["cooldown_period"] = [int(cooldown_range[0]), int(cooldown_range[1])]


def get_default_cooldown_period(timeframe):
    """
    Get the default cooldown period for a given timeframe.
    
    Args:
        timeframe (str): Timeframe string (e.g., "1min", "5min", "1H", "1D")
        
    Returns:
        int: Default cooldown period in number of candles
    """
    # Load config to get cooldown periods
    config = load_config()
    cooldown_periods = config["optimization"]["cooldown_periods"]
    
    # Standardize timeframe format
    tf_standardized = timeframe.replace("T", "min")
    
    # Return default cooldown period or raise error if timeframe not found
    if tf_standardized in cooldown_periods:
        return cooldown_periods[tf_standardized]
    else:
        raise ValueError(f"No default cooldown period defined for timeframe: {timeframe}")