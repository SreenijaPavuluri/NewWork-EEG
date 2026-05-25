"""Structured logging for training runs."""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "results/logs", level: int = logging.INFO) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run_{timestamp}.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
