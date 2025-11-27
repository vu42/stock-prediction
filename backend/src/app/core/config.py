"""
Application configuration using pydantic-settings.
Loads settings from environment variables and .env file.
"""

import os
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # APPLICATION
    # =========================================================================
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    secret_key: str = Field(default="dev-secret-key", alias="SECRET_KEY")

    # =========================================================================
    # DATABASE - PostgreSQL
    # =========================================================================
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/stock_prediction",
        alias="DATABASE_URL",
    )

    # =========================================================================
    # REDIS - For RQ Task Queue
    # =========================================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
    )

    # =========================================================================
    # JWT AUTHENTICATION
    # =========================================================================
    jwt_secret_key: str = Field(
        default="jwt-dev-secret-key",
        alias="JWT_SECRET_KEY",
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(
        default=30, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7, alias="JWT_REFRESH_TOKEN_EXPIRE_DAYS"
    )

    # =========================================================================
    # CORS
    # =========================================================================
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        alias="CORS_ORIGINS",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v

    # =========================================================================
    # AIRFLOW INTEGRATION
    # =========================================================================
    airflow_base_url: str = Field(
        default="http://localhost:8080",
        alias="AIRFLOW_BASE_URL",
    )
    airflow_username: str = Field(default="admin", alias="AIRFLOW_USERNAME")
    airflow_password: str = Field(default="admin", alias="AIRFLOW_PASSWORD")

    # =========================================================================
    # AWS S3 / MINIO - Artifact Storage
    # =========================================================================
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    s3_access_key_id: str | None = Field(default=None, alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str | None = Field(default=None, alias="S3_SECRET_ACCESS_KEY")
    s3_bucket_name: str = Field(
        default="stock-prediction-artifacts",
        alias="S3_BUCKET_NAME",
    )
    s3_region: str = Field(default="us-east-1", alias="S3_REGION")

    # =========================================================================
    # EMAIL - SendGrid
    # =========================================================================
    sendgrid_api_key: str | None = Field(default=None, alias="SENDGRID_API_KEY")
    email_sender: str = Field(default="noreply@example.com", alias="EMAIL_SENDER")
    email_recipient: str = Field(default="admin@example.com", alias="EMAIL_RECIPIENT")

    # =========================================================================
    # VN30 STOCK DATA
    # =========================================================================
    vndirect_api_url: str = Field(
        default="https://api-finfo.vndirect.com.vn/v4/stock_prices",
        alias="VNDIRECT_API_URL",
    )
    data_start_date: str = Field(default="2000-01-01", alias="DATA_START_DATE")

    # VN30 Stock List
    vn30_stocks: List[str] = Field(
        default=[
            "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
            "KDH", "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SSI",
            "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB",
        ],
        alias="VN30_STOCKS",
    )

    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    sequence_length: int = Field(default=60, alias="SEQUENCE_LENGTH")
    future_days: int = Field(default=30, alias="FUTURE_DAYS")
    random_forest_estimators: int = Field(default=100, alias="RANDOM_FOREST_ESTIMATORS")
    gradient_boosting_estimators: int = Field(
        default=100, alias="GRADIENT_BOOSTING_ESTIMATORS"
    )
    svr_c: float = Field(default=1.0, alias="SVR_C")
    ridge_alpha: float = Field(default=1.0, alias="RIDGE_ALPHA")
    continue_training: bool = Field(default=True, alias="CONTINUE_TRAINING")

    # =========================================================================
    # OUTPUT PATHS
    # =========================================================================
    output_dir: str = Field(default="output", alias="OUTPUT_DIR")

    @property
    def base_output_dir(self) -> str:
        """Get absolute path to output directory."""
        if os.path.isabs(self.output_dir):
            return self.output_dir
        # Relative to project root (two levels up from this file)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        )
        return os.path.join(project_root, self.output_dir)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()

