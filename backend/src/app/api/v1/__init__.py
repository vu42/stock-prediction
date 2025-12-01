"""
API v1 module.
Aggregates all v1 routers.
"""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.pipelines import router as pipelines_router
from app.api.v1.predictions import router as predictions_router
from app.api.v1.stocks import router as stocks_router
from app.api.v1.training import router as training_router

# Create main v1 router
api_router = APIRouter()

# Include all routers
api_router.include_router(auth_router)
api_router.include_router(stocks_router)
api_router.include_router(predictions_router)
api_router.include_router(training_router)
api_router.include_router(pipelines_router)

__all__ = ["api_router"]
