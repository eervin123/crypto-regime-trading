# Crypto Trading Strategy Framework

A framework for developing, optimizing, and backtesting multiple trading strategies across different market regimes.

## Key Features

- Multiple trading strategies including:
  - Moving Average Crossover (Long/Short)
  - MACD Divergence (Long/Short)
  - RSI Divergence (Long/Short)
  - Bollinger Bands (Long/Short)
  - Parabolic SAR (Long/Short)
  - RSI Mean Reversion (Long/Short)
  - Mean Reversion (Long/Short)
- Market regime classification
- Parameter optimization using Optuna
- Comprehensive backtesting framework
- Support for multiple timeframes
- Multi-asset support (BTC, ETH)

## Project Structure

# Crypto Regime Trading

## Strategy Optimization

The project includes two implementations of the strategy optimization framework:

### Production Implementation
- `optuna_multistrat.py`: Main implementation of the multi-strategy optimization framework
- Stable, well-tested implementation for production use
- Sequential processing for reliability and debugging

### Experimental Parallel Implementation
- `optuna_multistrat_parallel.py`: Experimental parallel version
- Uses parallel processing for performance optimization
- Includes performance monitoring and logging
- Maintains same functionality as production version

## Performance Comparison
To compare performance between implementations: