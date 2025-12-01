"""
RQ Worker entry point.
"""

import os
import sys

from redis import Redis
from rq import Connection, Worker

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import settings
from app.core.logging import setup_logging, get_logger

logger = get_logger(__name__)


def get_redis_connection() -> Redis:
    """Get Redis connection for the worker."""
    return Redis.from_url(settings.redis_url)


def run_worker(queues: list[str] | None = None):
    """
    Run the RQ worker.
    
    Args:
        queues: List of queue names to listen to (default: ["default", "high", "low"])
    """
    setup_logging()
    
    if queues is None:
        queues = ["high", "default", "low"]
    
    logger.info(f"Starting RQ worker for queues: {queues}")
    
    redis_conn = get_redis_connection()
    
    with Connection(redis_conn):
        worker = Worker(queues)
        worker.work(with_scheduler=True)


def main():
    """Main entry point for the worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock Prediction RQ Worker")
    parser.add_argument(
        "--queues",
        nargs="+",
        default=["high", "default", "low"],
        help="Queue names to listen to",
    )
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Run in burst mode (quit after all jobs complete)",
    )
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info(f"Starting worker for queues: {args.queues}")
    
    redis_conn = get_redis_connection()
    
    with Connection(redis_conn):
        worker = Worker(args.queues)
        worker.work(burst=args.burst, with_scheduler=True)


if __name__ == "__main__":
    main()

