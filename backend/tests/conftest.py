"""
Pytest configuration and fixtures for backend tests.
"""

import os
import sys
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app.core.config import settings
from app.core.security import create_access_token, get_password_hash
from app.db.base import Base
from app.db.models import User, UserRole
from app.db.session import get_db
from app.main import app


# Test database URL (use SQLite for faster tests)
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def engine():
    """Create test database engine."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine


@pytest.fixture(scope="session")
def tables(engine):
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db(engine, tables) -> Generator[Session, None, None]:
    """Get a test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(db: Session) -> Generator[TestClient, None, None]:
    """Get a test client with database session override."""
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db: Session) -> User:
    """Create a test end user."""
    user = User(
        username="testuser",
        password_hash=get_password_hash("testpass123"),
        display_name="Test User",
        role=UserRole.END_USER.value,
        email="test@example.com",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_ds_user(db: Session) -> User:
    """Create a test data scientist user."""
    user = User(
        username="testds",
        password_hash=get_password_hash("testpass123"),
        display_name="Test Data Scientist",
        role=UserRole.DATA_SCIENTIST.value,
        email="ds@example.com",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_admin_user(db: Session) -> User:
    """Create a test admin user."""
    user = User(
        username="testadmin",
        password_hash=get_password_hash("testpass123"),
        display_name="Test Admin",
        role=UserRole.ADMIN.value,
        email="admin@example.com",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def user_token(test_user: User) -> str:
    """Get access token for test user."""
    return create_access_token(data={"sub": str(test_user.id)})


@pytest.fixture
def ds_token(test_ds_user: User) -> str:
    """Get access token for test data scientist."""
    return create_access_token(data={"sub": str(test_ds_user.id)})


@pytest.fixture
def admin_token(test_admin_user: User) -> str:
    """Get access token for test admin."""
    return create_access_token(data={"sub": str(test_admin_user.id)})


@pytest.fixture
def auth_headers(user_token: str) -> dict:
    """Get auth headers for test user."""
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture
def ds_auth_headers(ds_token: str) -> dict:
    """Get auth headers for test data scientist."""
    return {"Authorization": f"Bearer {ds_token}"}


@pytest.fixture
def admin_auth_headers(admin_token: str) -> dict:
    """Get auth headers for test admin."""
    return {"Authorization": f"Bearer {admin_token}"}


# Mock API response fixture
@pytest.fixture
def mock_vndirect_response():
    """Sample VNDirect API response."""
    return {
        "data": [
            {
                "date": "2024-01-01",
                "open": 100.0,
                "high": 105.0,
                "low": 98.0,
                "close": 103.0,
                "nmvolume": 1000000,
                "code": "VCB",
            },
            {
                "date": "2024-01-02",
                "open": 103.0,
                "high": 107.0,
                "low": 102.0,
                "close": 106.0,
                "nmvolume": 1200000,
                "code": "VCB",
            },
        ]
    }

