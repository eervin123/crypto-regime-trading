"""
Simple tester for alternative trading strategies.
Uses the same data pipeline as recent_strategy_analysis.py but focused on testing
the new strategies from alt_strategies.py
"""

# Data analysis libraries
import pandas as pd
import vectorbtpro as vbt

# Local imports
from .alt_strategies import volatility_breakout, volume_profile_range
from .regimes_multi_strat_pf import calculate_regimes_nb

def load_test_data(lookback_days=365):
    """Load and prepare test data"""
    data = vbt.BinanceData.load("data/m1_data.pkl")
    
    # Get end date and calculate start date
    end_idx = data.index[-1]
    start_idx = end_idx - pd.Timedelta(days=lookback_days)
    
    # Resample to analysis timeframe (e.g., 1H) and regime timeframe (1D)
    data_1h = {
        "BTC": data.resample("1H").data["BTCUSDT"][start_idx:],
        "ETH": data.resample("1H").data["ETHUSDT"][start_idx:]
    }
    
    data_1d = {
        "BTC": data.resample("1D").data["BTCUSDT"][start_idx:],
        "ETH": data.resample("1D").data["ETHUSDT"][start_idx:]
    }
    
    # Add returns for regime calculation
    for symbol in ["BTC", "ETH"]:
        data_1d[symbol]["Return"] = data_1d[symbol]["Close"].pct_change()
    
    return data_1h, data_1d

def calculate_test_regimes(data_1d, data_1h):
    """Calculate and align market regimes"""
    # Create regime indicator factory
    RegimeIndicator = vbt.IndicatorFactory(
        class_name="RegimeIndicator",
        input_names=["price", "returns"],
        param_names=["ma_short_window", "ma_long_window", "vol_short_window", "avg_vol_window"],
        output_names=["regimes"]
    ).with_apply_func(calculate_regimes_nb)
    
    aligned_regimes = {}
    
    for symbol in ["BTC", "ETH"]:
        # Calculate regime indicators
        regime_indicator = RegimeIndicator.run(
            data_1d[symbol]["Close"],
            data_1d[symbol]["Return"],
            ma_short_window=21,
            ma_long_window=88,
            vol_short_window=21,
            avg_vol_window=365,
        )
        
        # Resample daily regimes to hourly and align
        regime_data = pd.Series(regime_indicator.regimes.values, index=data_1d[symbol].index)
        hourly_regime_data = regime_data.resample("1H").ffill()
        
        # Align with analysis timeframe data
        aligned_regimes[symbol] = hourly_regime_data.reindex(
            data_1h[symbol].index, method="ffill"
        )
    
    return aligned_regimes

def test_strategies():
    """Test the alternative strategies"""
    # Load and prepare data
    print("Loading data...")
    data_1h, data_1d = load_test_data(lookback_days=365)
    
    print("Calculating regimes...")
    regimes = calculate_test_regimes(data_1d, data_1h)
    
    # Test volatility breakout strategy
    print("\nTesting Volatility Breakout strategy...")
    vol_breakout_results = {}
    for symbol in ["BTC", "ETH"]:
        vol_breakout_results[symbol] = volatility_breakout(
            symbol_ohlcv_df=data_1h[symbol],
            regime_data=regimes[symbol],
            lookback_window=20,
            vol_multiplier=1.5,
            direction="both",
            allowed_regimes=[1, 2]  # Test in trending regimes
        )
        
        print(f"\n{symbol} Volatility Breakout Results:")
        print(vol_breakout_results[symbol].stats())
        vol_breakout_results[symbol].plot_cum_returns().show()
    
    # Test volume profile range strategy
    print("\nTesting Volume Profile Range strategy...")
    vol_profile_results = {}
    for symbol in ["BTC", "ETH"]:
        vol_profile_results[symbol] = volume_profile_range(
            symbol_ohlcv_df=data_1h[symbol],
            regime_data=regimes[symbol],
            profile_period="1D",
            value_area_pct=0.70,
            direction="both",
            allowed_regimes=[3, 4]  # Test in ranging regimes
        )
        
        print(f"\n{symbol} Volume Profile Range Results:")
        print(vol_profile_results[symbol].stats())
        vol_profile_results[symbol].plot_cum_returns().show()
    
    # Create combined portfolio
    print("\nCreating combined portfolio...")
    combined_portfolio = vbt.Portfolio.column_stack(
        list(vol_breakout_results.values()) + list(vol_profile_results.values()),
        cash_sharing=True,
        group_by=True,
        init_cash=1000
    )
    
    print("\nCombined Portfolio Results:")
    print(combined_portfolio.stats())
    combined_portfolio.plot_cum_returns().show()

if __name__ == "__main__":
    test_strategies() 