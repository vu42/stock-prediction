"""
Tests for authentication API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.security import get_password_hash
from app.db.models import User, UserRole


class TestLogin:
    """Test cases for login endpoint."""

    def test_login_success(self, client: TestClient, test_user: User):
        """Test successful login."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "accessToken" in data
        assert "refreshToken" in data
        assert data["user"]["username"] == "testuser"
        assert data["user"]["role"] == "end_user"

    def test_login_invalid_password(self, client: TestClient, test_user: User):
        """Test login with invalid password."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "testuser", "password": "wrongpassword"},
        )
        
        assert response.status_code == 401

    def test_login_invalid_username(self, client: TestClient):
        """Test login with non-existent username."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "nonexistent", "password": "somepassword"},
        )
        
        assert response.status_code == 401


class TestGetCurrentUser:
    """Test cases for get current user endpoint."""

    def test_get_me_success(self, client: TestClient, auth_headers: dict, test_user: User):
        """Test getting current user info."""
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["displayName"] == "Test User"

    def test_get_me_no_auth(self, client: TestClient):
        """Test getting current user without auth."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == 403  # No auth header


class TestRegister:
    """Test cases for user registration endpoint."""

    def test_register_success(self, client: TestClient, db):
        """Test successful user registration."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "password": "newpass123",
                "displayName": "New User",
                "role": "end_user",
                "email": "new@example.com",
            },
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newuser"

    def test_register_duplicate_username(self, client: TestClient, test_user: User):
        """Test registration with existing username."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",  # Already exists
                "password": "newpass123",
                "displayName": "Another User",
            },
        )
        
        assert response.status_code == 409


class TestRefreshToken:
    """Test cases for token refresh endpoint."""

    def test_refresh_success(self, client: TestClient, test_user: User):
        """Test successful token refresh."""
        # First login to get tokens
        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        refresh_token = login_response.json()["refreshToken"]
        
        # Refresh
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refreshToken": refresh_token},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "accessToken" in data
        assert "refreshToken" in data

    def test_refresh_invalid_token(self, client: TestClient):
        """Test refresh with invalid token."""
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refreshToken": "invalid-token"},
        )
        
        assert response.status_code == 401

