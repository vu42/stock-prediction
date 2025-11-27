"""
Authentication schemas for request/response validation.
"""

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """Login request schema."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    """Token response schema."""

    access_token: str = Field(..., alias="accessToken")
    refresh_token: str = Field(..., alias="refreshToken")
    token_type: str = Field(default="bearer", alias="tokenType")
    user: "UserResponse"

    class Config:
        populate_by_name = True


class UserResponse(BaseModel):
    """User response schema."""

    id: str
    username: str
    role: str
    display_name: str = Field(..., alias="displayName")
    email: str | None = None

    class Config:
        populate_by_name = True
        from_attributes = True


class UserCreate(BaseModel):
    """User creation schema."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    display_name: str = Field(..., min_length=1, max_length=100, alias="displayName")
    role: str = Field(default="end_user")
    email: EmailStr | None = None

    class Config:
        populate_by_name = True


class PasswordChange(BaseModel):
    """Password change schema."""

    current_password: str = Field(..., alias="currentPassword")
    new_password: str = Field(..., min_length=6, alias="newPassword")

    class Config:
        populate_by_name = True


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""

    refresh_token: str = Field(..., alias="refreshToken")

    class Config:
        populate_by_name = True


# Update forward references
TokenResponse.model_rebuild()

