"""
Script to analyze interim results from ongoing optimizations.
Can be run while the main optimization is still running.
"""

import pandas as pd
import optuna
from pathlib import Path
from optuna.storages import RDBStorage
from tabulate import tabulate
import logging
import sqlite3

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def analyze_interim_results():
    """Analyze current optimization results from database and saved files."""
    # Connect to the database
    db_dir = Path("results/optuna_db")
    if not db_dir.exists():
        print("No database found. Has the optimization started?")
        return

    storage = RDBStorage(
        url=f"sqlite:///{db_dir}/optuna_studies.db",
        heartbeat_interval=60,
        grace_period=120,
    )

    # Get all study names using direct SQL query
    conn = sqlite3.connect(f"{db_dir}/optuna_studies.db")
    cursor = conn.cursor()
    cursor.execute("SELECT study_name FROM studies")
    study_names = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"\nFound {len(study_names)} studies")
    
    # Group studies by symbol and strategy
    study_results = []
    for study_name in study_names:
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        # Parse study name for information
        parts = study_name.split('_')
        symbol = parts[0].upper()
        direction = parts[-2]  # Assuming format: symbol_strategy_direction_timestamp
        strategy = ' '.join(parts[1:-2]).title()
        
        # Get study statistics
        n_trials = len(study.trials)
        best_value = study.best_value if study.best_trial else "No trials completed"
        best_params = study.best_params if study.best_trial else {}
        
        study_results.append({
            'Symbol': symbol,
            'Strategy': strategy,
            'Direction': direction,
            'Trials': n_trials,
            'Best Value': best_value,
            'Best Params': best_params
        })
    
    # Convert to DataFrame for better display
    if study_results:
        df = pd.DataFrame(study_results)
        for symbol in df['Symbol'].unique():
            print(f"\n{symbol} Optimization Progress:")
            symbol_df = df[df['Symbol'] == symbol]
            print(tabulate(
                symbol_df[['Strategy', 'Direction', 'Trials', 'Best Value']],
                headers='keys',
                tablefmt='pipe',
                floatfmt='.4f'
            ))

    # Check saved results
    results_dir = Path("results/backtests")
    if results_dir.exists():
        print("\nAnalyzing saved results:")
        for csv_file in results_dir.glob("*.csv"):
            if "results" in csv_file.name:
                print(f"\nFound results file: {csv_file.name}")
                try:
                    # Read raw CSV data first
                    with open(csv_file, 'r') as f:
                        lines = f.readlines()
                    
                    if len(lines) >= 4:  # Need at least header rows and one data row
                        # Parse header rows
                        metrics = [m.strip() for m in lines[0].strip().split(',')]
                        strategies = [s.strip() for s in lines[1].strip().split(',')]
                        directions = [d.strip() for d in lines[2].strip().split(',')]
                        symbols = [s.strip() for s in lines[3].strip().split(',')]
                        data = [d.strip() for d in lines[4].strip().split(',')]  # First data row
                        
                        # Group results by strategy
                        strategy_results = {}
                        for i in range(1, len(metrics)):  # Skip first column (usually empty or 'Strategy')
                            if strategies[i] and strategies[i] != 'Strategy':
                                strategy = strategies[i]
                                if strategy not in strategy_results:
                                    strategy_results[strategy] = {}
                                
                                # Get base metric name without the numbered suffix
                                metric = metrics[i].split('.')[0]
                                try:
                                    if data[i] and data[i] != '':
                                        value = float(data[i])
                                        if metric not in strategy_results[strategy]:
                                            strategy_results[strategy][metric] = []
                                        strategy_results[strategy][metric].append(value)
                                except (ValueError, IndexError) as e:
                                    continue
                        
                        # Print results
                        print(f"\nStrategies found: {len(strategy_results)}")
                        for strategy, metrics in strategy_results.items():
                            if metrics:  # Only print if we have metrics
                                print(f"\n{strategy}:")
                                for metric, values in metrics.items():
                                    if values:  # Only print if we have values
                                        avg_value = sum(values) / len(values)
                                        print(f"  {metric}: {avg_value:.4f}")
                except Exception as e:
                    print(f"Error reading {csv_file.name}: {e}")

def main():
    setup_logging()
    print("Analyzing interim optimization results...")
    analyze_interim_results()

if __name__ == "__main__":
    main() 