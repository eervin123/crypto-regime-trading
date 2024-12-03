import logging
import psutil

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("parallel_optimization.log"),
            logging.StreamHandler(),
        ],
    )

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB") 