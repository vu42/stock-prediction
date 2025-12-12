"""
S3/MinIO storage client for artifact management.
"""

import os
from datetime import datetime, timedelta
from typing import Any, BinaryIO

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.errors import ExternalServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)


class S3StorageClient:
    """Client for S3-compatible storage (AWS S3 or MinIO)."""

    def __init__(self):
        """Initialize the S3 client."""
        self._client = None
        self._bucket = settings.s3_bucket_name

    @property
    def client(self):
        """Get or create S3 client."""
        if self._client is None:
            config = Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            )

            client_kwargs: dict[str, Any] = {
                "service_name": "s3",
                "config": config,
                "region_name": settings.s3_region,
            }

            # Use custom endpoint for MinIO
            if settings.s3_endpoint_url:
                client_kwargs["endpoint_url"] = settings.s3_endpoint_url

            # Use credentials if provided
            if settings.s3_access_key_id and settings.s3_secret_access_key:
                client_kwargs["aws_access_key_id"] = settings.s3_access_key_id
                client_kwargs["aws_secret_access_key"] = settings.s3_secret_access_key

            self._client = boto3.client(**client_kwargs)

        return self._client

    def ensure_bucket_exists(self) -> bool:
        """
        Ensure the bucket exists, create if it doesn't.

        Returns:
            True if bucket exists or was created
        """
        try:
            self.client.head_bucket(Bucket=self._bucket)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                # Bucket doesn't exist, create it
                try:
                    self.client.create_bucket(
                        Bucket=self._bucket,
                        CreateBucketConfiguration={
                            "LocationConstraint": settings.s3_region,
                        },
                    )
                    logger.info(f"Created bucket: {self._bucket}")
                    return True
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
                    raise ExternalServiceError(
                        "S3", f"Failed to create bucket: {create_error}"
                    )
            else:
                logger.error(f"Error checking bucket: {e}")
                raise ExternalServiceError("S3", str(e))

    def upload_file(
        self,
        file_path: str,
        key: str,
        content_type: str | None = None,
    ) -> str:
        """
        Upload a file to S3.

        Args:
            file_path: Local file path
            key: S3 object key
            content_type: Optional content type

        Returns:
            S3 URL of uploaded file
        """
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        try:
            self.client.upload_file(
                file_path,
                self._bucket,
                key,
                ExtraArgs=extra_args if extra_args else None,
            )
            logger.info(f"Uploaded file to s3://{self._bucket}/{key}")
            return self.get_url(key)
        except ClientError as e:
            logger.error(f"Failed to upload file: {e}")
            raise ExternalServiceError("S3", f"Failed to upload file: {e}")

    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        key: str,
        content_type: str | None = None,
    ) -> str:
        """
        Upload a file object to S3.

        Args:
            file_obj: File-like object
            key: S3 object key
            content_type: Optional content type

        Returns:
            S3 URL of uploaded file
        """
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        try:
            self.client.upload_fileobj(
                file_obj,
                self._bucket,
                key,
                ExtraArgs=extra_args if extra_args else None,
            )
            logger.info(f"Uploaded file object to s3://{self._bucket}/{key}")
            return self.get_url(key)
        except ClientError as e:
            logger.error(f"Failed to upload file object: {e}")
            raise ExternalServiceError("S3", f"Failed to upload file: {e}")

    def download_file(self, key: str, file_path: str) -> str:
        """
        Download a file from S3.

        Args:
            key: S3 object key
            file_path: Local file path to save to

        Returns:
            Local file path
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            self.client.download_file(self._bucket, key, file_path)
            logger.info(f"Downloaded s3://{self._bucket}/{key} to {file_path}")
            return file_path
        except ClientError as e:
            logger.error(f"Failed to download file: {e}")
            raise ExternalServiceError("S3", f"Failed to download file: {e}")

    def get_url(self, key: str) -> str:
        """
        Get the URL for an S3 object.

        For MinIO: uses s3_public_url if set (browser-accessible),
        otherwise falls back to s3_endpoint_url.

        Args:
            key: S3 object key

        Returns:
            S3 URL accessible from browser
        """
        # Prefer public URL for browser access (e.g., http://localhost:9000)
        if settings.s3_public_url:
            return f"{settings.s3_public_url}/{self._bucket}/{key}"
        # Fall back to endpoint URL (may be internal Docker URL)
        if settings.s3_endpoint_url:
            return f"{settings.s3_endpoint_url}/{self._bucket}/{key}"
        return f"https://{self._bucket}.s3.{settings.s3_region}.amazonaws.com/{key}"

    def get_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        method: str = "get_object",
    ) -> str:
        """
        Generate a presigned URL for an S3 object.

        Args:
            key: S3 object key
            expiration: URL expiration in seconds
            method: S3 operation (get_object, put_object)

        Returns:
            Presigned URL
        """
        try:
            url = self.client.generate_presigned_url(
                method,
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise ExternalServiceError("S3", f"Failed to generate presigned URL: {e}")

    def delete_file(self, key: str) -> bool:
        """
        Delete a file from S3.

        Args:
            key: S3 object key

        Returns:
            True if deleted
        """
        try:
            self.client.delete_object(Bucket=self._bucket, Key=key)
            logger.info(f"Deleted s3://{self._bucket}/{key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file: {e}")
            raise ExternalServiceError("S3", f"Failed to delete file: {e}")

    def list_files(self, prefix: str = "") -> list[dict[str, Any]]:
        """
        List files in S3 with given prefix.

        Args:
            prefix: Key prefix to filter by

        Returns:
            List of file info dicts
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=prefix,
            )
            files = []
            for obj in response.get("Contents", []):
                files.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"],
                    }
                )
            return files
        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            raise ExternalServiceError("S3", f"Failed to list files: {e}")


# Singleton instance
storage_client = S3StorageClient()


# Helper functions for artifact management
def upload_model_artifact(
    stock_symbol: str,
    run_id: str,
    artifact_type: str,
    file_path: str,
) -> str:
    """
    Upload a model artifact to S3.

    Args:
        stock_symbol: Stock ticker symbol
        run_id: Experiment run ID
        artifact_type: Type of artifact (model, scaler, evaluation_png, etc.)
        file_path: Local file path

    Returns:
        S3 URL of uploaded artifact
    """
    extension = os.path.splitext(file_path)[1]
    key = f"artifacts/{run_id}/{stock_symbol}/{artifact_type}{extension}"

    content_type = None
    if extension == ".png":
        content_type = "image/png"
    elif extension == ".csv":
        content_type = "text/csv"
    elif extension == ".pkl":
        content_type = "application/octet-stream"

    return storage_client.upload_file(file_path, key, content_type)
