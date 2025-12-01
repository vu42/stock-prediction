"""
Business logic services.
"""

from app.services.data_fetcher import (
    fetch_all_stocks,
    fetch_stock_data,
    get_database_stats,
    get_last_data_date,
    load_stock_data_from_db,
)
from app.services.email_service import (
    send_email_notification,
    send_training_completion_email,
)
from app.services.model_trainer import (
    evaluate_model,
    get_stock_file_paths,
    predict_future_prices,
    train_prediction_model,
)
from app.services.predictions import (
    get_chart_data,
    get_current_price,
    get_market_table_data,
    get_model_status,
    get_models_overview,
    get_stock_predictions,
    get_top_picks,
)
from app.services.stock_service import (
    create_stock,
    delete_stock,
    get_all_stocks,
    get_stock_by_id,
    get_stock_by_ticker,
    get_stock_detail,
    seed_vn30_stocks,
    update_stock,
)

__all__ = [
    # Data fetcher
    "fetch_stock_data",
    "fetch_all_stocks",
    "load_stock_data_from_db",
    "get_last_data_date",
    "get_database_stats",
    # Model trainer
    "train_prediction_model",
    "evaluate_model",
    "predict_future_prices",
    "get_stock_file_paths",
    # Predictions
    "get_top_picks",
    "get_market_table_data",
    "get_stock_predictions",
    "get_chart_data",
    "get_model_status",
    "get_models_overview",
    "get_current_price",
    # Email
    "send_email_notification",
    "send_training_completion_email",
    # Stock service
    "get_stock_by_ticker",
    "get_stock_by_id",
    "get_all_stocks",
    "create_stock",
    "update_stock",
    "delete_stock",
    "get_stock_detail",
    "seed_vn30_stocks",
]
