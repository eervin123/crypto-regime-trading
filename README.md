# Crypto Trading Strategy Optimization

This repository contains a framework for developing, optimizing, and analyzing various cryptocurrency trading strategies using VectorBT Pro. The framework supports multiple strategy types and market regime detection.

## Data Requirements

This project uses `m1_data.pkl`, a minutely resolution VectorBT Pro object containing BTC and ETH data from Binance. You can download the data file here:
[m1_data.pkl (Google Drive)](https://drive.google.com/file/d/13f_MM_drI-8rOHMr8FVYFhwX-cTl-3FO/view?usp=sharing)
Create a `data` directory and place the file in the directory.
## Key Components

### optuna_multistrat.py

The core optimization script that:

1. Loads strategy configurations from `config/optuna_config.yaml`
2. Performs hyperparameter optimization using Optuna for each strategy
3. Tests strategies across different market regimes
4. Generates both in-sample and out-of-sample results
5. Saves detailed performance metrics and visualizations

#### How to Use

1. Configure your strategy parameters in `config/optuna_config.yaml`:

## Features

- Multiple trading strategies:
  - Moving Average (MA)
  - MACD Divergence
  - RSI Divergence
  - Bollinger Bands
  - Parabolic SAR
  - RSI Mean Reversion
  - Mean Reversion

- Market regime detection
- Strategy optimization using Optuna
- In-sample and out-of-sample testing
- Comprehensive performance metrics
- Configurable parameters via YAML

## Project Structure 