"""
Authentication API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import (
    CurrentUser,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
)
from app.db.models import User
from app.db.session import get_db
from app.schemas import (
    LoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    """
    Authenticate user and return access/refresh tokens.
    """
    user = authenticate_user(db, request.username, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive",
        )
    
    # Create tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return TokenResponse(
        accessToken=access_token,
        refreshToken=refresh_token,
        tokenType="bearer",
        user=UserResponse(
            id=str(user.id),
            username=user.username,
            role=user.role,
            displayName=user.display_name,
            email=user.email,
        ),
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: CurrentUser):
    """
    Get current authenticated user information.
    """
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        role=current_user.role,
        displayName=current_user.display_name,
        email=current_user.email,
    )


@router.post("/logout")
async def logout(current_user: CurrentUser):
    """
    Logout current user (invalidate tokens).
    Note: For stateless JWT, this is mostly a client-side operation.
    In production, you might want to implement token blacklisting.
    """
    return {"message": "Successfully logged out"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db),
):
    """
    Refresh access token using refresh token.
    """
    try:
        payload = decode_token(request.refresh_token)
        user_id = payload.get("sub")
        token_type = payload.get("type")
        
        if not user_id or token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
        
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    # Get user
    from sqlalchemy import select
    stmt = select(User).where(User.id == user_id)
    user = db.execute(stmt).scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    # Create new tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    new_refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return TokenResponse(
        accessToken=access_token,
        refreshToken=new_refresh_token,
        tokenType="bearer",
        user=UserResponse(
            id=str(user.id),
            username=user.username,
            role=user.role,
            displayName=user.display_name,
            email=user.email,
        ),
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: UserCreate,
    db: Session = Depends(get_db),
):
    """
    Register a new user.
    Note: In production, this might be admin-only or have additional validation.
    """
    from sqlalchemy import select
    
    # Check if username exists
    stmt = select(User).where(User.username == request.username)
    existing = db.execute(stmt).scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )
    
    # Check if email exists
    if request.email:
        stmt = select(User).where(User.email == request.email)
        existing = db.execute(stmt).scalar_one_or_none()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )
    
    # Create user
    user = User(
        username=request.username,
        password_hash=get_password_hash(request.password),
        display_name=request.display_name,
        role=request.role,
        email=request.email,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return UserResponse(
        id=str(user.id),
        username=user.username,
        role=user.role,
        displayName=user.display_name,
        email=user.email,
    )

