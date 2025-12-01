"""
Redis Queue (RQ) integration for background job processing.
"""

from typing import Any, Callable

from redis import Redis
from rq import Queue
from rq.job import Job

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_redis_connection() -> Redis:
    """
    Get a Redis connection.
    
    Returns:
        Redis connection instance
    """
    return Redis.from_url(settings.redis_url)


def get_queue(name: str = "default") -> Queue:
    """
    Get an RQ queue.
    
    Args:
        name: Queue name (default, high, low)
        
    Returns:
        RQ Queue instance
    """
    return Queue(name, connection=get_redis_connection())


def enqueue_job(
    func: Callable,
    *args: Any,
    queue_name: str = "default",
    job_timeout: int = 3600,
    result_ttl: int = 86400,
    **kwargs: Any,
) -> Job:
    """
    Enqueue a job for background processing.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        queue_name: Queue name
        job_timeout: Job timeout in seconds (default 1 hour)
        result_ttl: Result TTL in seconds (default 1 day)
        **kwargs: Keyword arguments for the function
        
    Returns:
        RQ Job instance
    """
    queue = get_queue(queue_name)
    job = queue.enqueue(
        func,
        *args,
        job_timeout=job_timeout,
        result_ttl=result_ttl,
        **kwargs,
    )
    logger.info(f"Enqueued job {job.id} to queue '{queue_name}'")
    return job


def get_job(job_id: str) -> Job | None:
    """
    Get a job by ID.
    
    Args:
        job_id: Job ID
        
    Returns:
        Job instance or None
    """
    try:
        return Job.fetch(job_id, connection=get_redis_connection())
    except Exception:
        return None


def cancel_job(job_id: str) -> bool:
    """
    Cancel a job.
    
    Args:
        job_id: Job ID
        
    Returns:
        True if cancelled
    """
    job = get_job(job_id)
    if job:
        job.cancel()
        logger.info(f"Cancelled job {job_id}")
        return True
    return False


# Convenience functions for specific queues
def enqueue_training_job(run_id: str) -> Job:
    """
    Enqueue a training experiment job.
    
    Args:
        run_id: Experiment run ID
        
    Returns:
        RQ Job instance
    """
    # Import here to avoid circular imports
    from worker.tasks.experiments import run_training_experiment
    
    return enqueue_job(
        run_training_experiment,
        run_id,
        queue_name="default",
        job_timeout=7200,  # 2 hours
    )


def enqueue_data_fetch_job(stock_symbol: str, to_date: str | None = None) -> Job:
    """
    Enqueue a data fetch job.
    
    Args:
        stock_symbol: Stock ticker symbol
        to_date: Optional end date
        
    Returns:
        RQ Job instance
    """
    # Import here to avoid circular imports
    from worker.tasks.experiments import fetch_stock_data_task
    
    return enqueue_job(
        fetch_stock_data_task,
        stock_symbol,
        to_date,
        queue_name="default",
        job_timeout=300,  # 5 minutes
    )


class QueueStats:
    """Queue statistics helper."""

    def __init__(self):
        """Initialize queue stats."""
        self.redis = get_redis_connection()

    def get_queue_length(self, queue_name: str = "default") -> int:
        """Get number of jobs in queue."""
        queue = get_queue(queue_name)
        return len(queue)

    def get_failed_count(self) -> int:
        """Get number of failed jobs."""
        from rq.registry import FailedJobRegistry
        registry = FailedJobRegistry(queue=get_queue())
        return len(registry)

    def get_stats(self) -> dict[str, Any]:
        """Get all queue stats."""
        return {
            "default_queue_length": self.get_queue_length("default"),
            "high_queue_length": self.get_queue_length("high"),
            "low_queue_length": self.get_queue_length("low"),
            "failed_jobs": self.get_failed_count(),
        }


queue_stats = QueueStats()

