"""
Results Utilities Module

This module handles the creation, formatting, and saving of strategy results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate
import vectorbtpro as vbt
import plotly.graph_objects as go


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
        strategy_data = pd.Series(dtype="object")

        # Add metrics with corrected Total Return formatting
        strategy_data["Total Return"] = (
            f"{row['Total Return'] * 100:.2f}%"  # Convert decimal to percentage
        )
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


def save_optimal_params(results_df, timeframe_id, target_regimes, config):
    """
    Save optimal parameters for each strategy and direction.

    Args:
        results_df (pd.DataFrame): DataFrame containing strategy results
        timeframe_id (str): Timeframe identifier for filenames
        target_regimes (list): List of target regime labels
        config (dict): Configuration dictionary
    """
    # Create results directory and get subdirectories
    results_dir = Path("results")
    params_dir = results_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    # Get number of best strategies to save
    n_best = config["optimization"].get("n_best", 3)

    # Split results by direction
    long_results = results_df[results_df["Direction"] == "long"]
    short_results = results_df[results_df["Direction"] == "short"]

    # Get top N strategies for each direction based on total return
    top_long = long_results.nlargest(n_best, "Total Return")
    top_short = short_results.nlargest(n_best, "Total Return")

    # Combine top strategies
    top_strategies = pd.concat([top_long, top_short])

    # Create filename with timeframe and regime info
    filename = f"optimal_params_regimes_{'_'.join(map(str, target_regimes))}_{timeframe_id}.csv"
    filepath = params_dir / filename

    # Save to CSV
    top_strategies.to_csv(filepath)
    print(f"\nOptimal parameters saved to: {filepath}")


def save_results_to_csv(results_dict, target_regimes, timeframe_id, backtests_dir):
    """Save in-sample, out-of-sample, and combined results to CSV files."""
    for period, results in results_dict.items():
        try:
            results_list = [r for r in results if isinstance(r, dict)]
            
            if results_list:
                csv_df = pd.DataFrame(results_list)
                if not csv_df.empty:
                    csv_df = csv_df.set_index(["Symbol", "Strategy", "Direction"])
                    csv_df = csv_df.unstack(["Strategy", "Direction"])

                    filename = f"{period}_results_regimes_{'_'.join(map(str, target_regimes))}_{timeframe_id}.csv"
                    filepath = backtests_dir / filename
                    csv_df.to_csv(filepath)
                    print(f"Saved {period} results to {filepath}")
            else:
                print(f"No valid results for {period}")
        except Exception as e:
            print(f"Error processing {period} results: {str(e)}")


def display_results_tables(results_dict, symbols=["BTC", "ETH"]):
    """Display formatted results tables for each symbol and period."""
    for symbol in symbols:
        for period, results in results_dict.items():
            print(f"\n{symbol} {period.title()} Results:")
            results_df = pd.DataFrame(results)
            symbol_results = results_df[results_df["Symbol"] == symbol]
            
            print(
                tabulate(
                    format_results_table(symbol_results),
                    headers="keys",
                    tablefmt="pipe",
                    floatfmt=".4f",
                )
            )
        print("\n" + "=" * 80)


def plot_strategy_results(results_dict, symbols=["BTC", "ETH"]):
    """Create and display strategy performance plots."""
    for symbol in symbols:
        # Get unique strategies
        strategies = pd.DataFrame(results_dict["in_sample"])["Strategy"].apply(
            lambda x: x.split(" (")[0]
        ).unique()
        n_strategies = len(strategies)

        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(n_strategies)))
        n_rows = int(np.ceil(n_strategies / n_cols))

        # Plot each period
        for period_name, results in results_dict.items():
            fig = vbt.make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=[f"{strat}" for strat in strategies],
                vertical_spacing=0.1,
            )

            # Plot results
            for idx, strategy in enumerate(strategies):
                row = idx // n_cols + 1
                col = idx % n_cols + 1

                # Convert results to DataFrame first
                results_df = pd.DataFrame(results)
                strategy_results = results_df[
                    (results_df["Symbol"] == symbol) & 
                    (results_df["Strategy"].str.startswith(strategy))
                ]
                
                if not strategy_results.empty:
                    for _, result in strategy_results.iterrows():
                        if result["Portfolio"] is not None:
                            pf_fig = result["Portfolio"].plot_cum_returns()
                            for trace in pf_fig.data:
                                trace.name = f"{result['Direction']}"
                                fig.add_trace(trace, row=row, col=col)

            fig.update_layout(
                height=300 * n_rows,
                width=1200,
                title=f"{symbol} {period_name} Performance by Strategy",
                showlegend=True,
            )
            fig.show()


def create_results_table(df, title):
    """
    Create a Plotly table figure from DataFrame.
    
    Args:
        df (pd.DataFrame): Results DataFrame
        title (str): Title for the table
    
    Returns:
        plotly.graph_objects.Figure: Interactive table figure
    """
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Metric'] + list(df.columns),
            fill_color='darkslategray',
            align='left',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[df.index] + [df[col] for col in df.columns],
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


def plot_results(results_dict, symbol, mode='subplots'):
    """
    Plot strategy results in either subplot or combined mode.
    
    Args:
        results_dict (dict): Dictionary containing results for different periods
        symbol (str): Symbol to plot ('BTC' or 'ETH')
        mode (str): Plot mode ('subplots' or 'combined')
    """
    if mode == 'subplots':
        # Original subplot implementation
        plot_strategy_results(results_dict, symbols=[symbol])
    elif mode == 'combined':
        # Combined plot implementation using make_subplots
        for period_name, results in results_dict.items():
            results_df = pd.DataFrame(results)
            symbol_results = results_df[results_df["Symbol"] == symbol]
            
            if not symbol_results.empty:
                # Create single subplot for combined view
                fig = vbt.make_subplots(rows=1, cols=1)
                
                for _, result in symbol_results.iterrows():
                    if result["Portfolio"] is not None:
                        pf_fig = result["Portfolio"].plot_cum_returns()
                        for trace in pf_fig.data:
                            trace.name = f"{result['Strategy']} ({result['Direction']})"
                            fig.add_trace(trace)

                fig.update_layout(
                    title=f"{symbol} {period_name} Performance (Combined)",
                    template="plotly_dark",
                    yaxis_type="log",
                    height=600,
                    width=1200,
                    showlegend=True
                )
                fig.show()


def plot_performance_comparison(in_sample_df, out_sample_df, full_df=None, symbols=["BTC", "ETH"]):
    """
    Plot vertical comparison of in-sample, out-of-sample, and full backtest results.
    
    Args:
        in_sample_df (pd.DataFrame): In-sample results
        out_sample_df (pd.DataFrame): Out-of-sample results
        full_df (pd.DataFrame, optional): Full backtest results
        symbols (list): List of symbols to plot
    """
    for symbol in symbols:
        # Determine number of rows based on available data
        n_rows = 3 if full_df is not None else 2
        
        # Create vertical subplots
        fig = vbt.make_subplots(
            rows=n_rows,
            cols=1,
            subplot_titles=[
                f"{symbol} In-Sample Performance",
                f"{symbol} Out-of-Sample Performance",
                f"{symbol} Full Backtest Performance" if full_df is not None else None,
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
                if result["Portfolio"] is not None:
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
                if result["Portfolio"] is not None:
                    pf_fig = result["Portfolio"].plot_cum_returns()
                    for trace in pf_fig.data:
                        trace.name = f"{strategy} ({result['Direction']})"
                        fig.add_trace(trace, row=2, col=1)

        # Plot full backtest results if available
        if full_df is not None:
            for strategy in strategies:
                strategy_results = full_df[
                    (full_df["Symbol"] == symbol) & (full_df["Strategy"] == strategy)
                ]
                for _, result in strategy_results.iterrows():
                    if result["Portfolio"] is not None:
                        pf_fig = result["Portfolio"].plot_cum_returns()
                        for trace in pf_fig.data:
                            trace.name = f"{strategy} ({result['Direction']})"
                            fig.add_trace(trace, row=3, col=1)

        # Update layout
        fig.update_layout(
            height=400 * n_rows,
            width=1200,
            title=f"{symbol} Strategy Performance Comparison",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
        )

        fig.show()


def display_results(results_dict, symbols=["BTC", "ETH"], display_mode='both'):
    """
    Display results in specified format.
    
    Args:
        results_dict (dict): Dictionary containing results for different periods
        symbols (list): List of symbols to display
        display_mode (str): Display mode ('table', 'grid', 'vertical', 'both')
    """
    for symbol in symbols:
        for period, results in results_dict.items():
            results_df = pd.DataFrame(results)
            symbol_results = results_df[results_df["Symbol"] == symbol]
            
            if display_mode in ['table', 'both']:
                # Create and show browser table
                table_fig = create_results_table(
                    format_results_table(symbol_results),
                    f"{symbol} {period.title()} Results"
                )
                table_fig.show()
            
            if display_mode in ['grid', 'both']:
                # Show grid style plots
                plot_strategy_results(results_dict, symbols=[symbol])
                
            if display_mode in ['vertical', 'both']:
                # Show vertical comparison plots
                if all(k in results_dict for k in ['in_sample', 'out_sample']):
                    plot_performance_comparison(
                        results_dict['in_sample'],
                        results_dict['out_sample'],
                        results_dict.get('full', None),  # Optional full backtest results
                        symbols=[symbol]
                    ) 