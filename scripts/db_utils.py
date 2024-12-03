"""
Database Utilities Module

This module handles database operations for storing and retrieving Optuna studies.
"""

from pathlib import Path
from optuna.storages import RDBStorage
import logging

def initialize_optuna_database():
    """
    Initialize SQLite database for Optuna studies.
    
    Returns:
        RDBStorage: Optuna storage object for database operations
    
    Raises:
        Exception: If database initialization fails
    """
    try:
        # Create results directory for database
        db_dir = Path("results/optuna_db")
        db_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database
        storage = RDBStorage(
            url=f"sqlite:///{db_dir}/optuna_studies.db",
            heartbeat_interval=60,
            grace_period=120,
        )

        return storage
    except Exception as e:
        logging.error(f"Failed to initialize database: {str(e)}")
        raise 