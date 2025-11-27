"""
External integrations - Airflow, S3, Redis Queue.
"""

from app.integrations.airflow_client import AirflowClient, airflow_client
from app.integrations.queue import (
    QueueStats,
    cancel_job,
    enqueue_data_fetch_job,
    enqueue_job,
    enqueue_training_job,
    get_job,
    get_queue,
    get_redis_connection,
    queue_stats,
)
from app.integrations.storage_s3 import (
    S3StorageClient,
    storage_client,
    upload_model_artifact,
)

__all__ = [
    # Airflow
    "AirflowClient",
    "airflow_client",
    # S3
    "S3StorageClient",
    "storage_client",
    "upload_model_artifact",
    # Queue
    "get_redis_connection",
    "get_queue",
    "enqueue_job",
    "get_job",
    "cancel_job",
    "enqueue_training_job",
    "enqueue_data_fetch_job",
    "QueueStats",
    "queue_stats",
]
