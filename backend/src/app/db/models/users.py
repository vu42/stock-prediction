"""
User and authentication models.
Maps to SPECS.md Section 6.2: users, auth_tokens
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class UserRole(str, Enum):
    """User role enumeration."""

    END_USER = "end_user"
    DATA_SCIENTIST = "data_scientist"
    ADMIN = "admin"


class User(Base, TimestampMixin):
    """
    Application users with role-based access.

    Roles:
    - end_user: Can view stocks, predictions, charts
    - data_scientist: Can configure training, run experiments, manage pipelines
    - admin: Full access
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=UserRole.END_USER.value,
    )
    email: Mapped[str | None] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    auth_tokens: Mapped[list["AuthToken"]] = relationship(
        "AuthToken",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    training_configs: Mapped[list["TrainingConfig"]] = relationship(
        "TrainingConfig",
        back_populates="owner",
        foreign_keys="TrainingConfig.owner_user_id",
    )
    experiment_runs: Mapped[list["ExperimentRun"]] = relationship(
        "ExperimentRun",
        back_populates="owner",
        foreign_keys="ExperimentRun.owner_user_id",
    )
    saved_stocks: Mapped[list["UserSavedStock"]] = relationship(
        "UserSavedStock",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"


class AuthToken(Base):
    """
    Authentication tokens for session management.
    Supports both access and refresh tokens.
    """

    __tablename__ = "auth_tokens"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="auth_tokens")

    @property
    def is_valid(self) -> bool:
        """Check if token is valid (not expired and not revoked)."""
        now = datetime.now(self.expires_at.tzinfo)
        return self.revoked_at is None and self.expires_at > now

    def __repr__(self) -> str:
        return f"<AuthToken(id={self.id}, user_id={self.user_id})>"


# Forward references for relationships (resolved at runtime)
from app.db.models.stocks import UserSavedStock  # noqa: E402, F401
from app.db.models.training import ExperimentRun, TrainingConfig  # noqa: E402, F401
