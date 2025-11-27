"""
Email notification service using SendGrid.
Migrated from modules/email_notifier.py.
"""

import base64
import os
from datetime import datetime

import pandas as pd
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Attachment,
    Disposition,
    FileContent,
    FileName,
    FileType,
    Mail,
)

from app.core.config import settings
from app.core.logging import get_logger
from app.services.model_trainer import get_stock_file_paths

logger = get_logger(__name__)


def send_email_notification(stock_symbols: list[str]) -> bool:
    """
    Send email summary report for all processed stocks.
    
    Args:
        stock_symbols: List of all stock symbols
        
    Returns:
        True if email sent successfully
    """
    sender_email = settings.email_sender
    recipient_email = settings.email_recipient
    
    # Count successful stocks (those with models)
    successful_stocks = []
    for stock in stock_symbols:
        paths = get_stock_file_paths(stock)
        if os.path.exists(paths["model"]):
            successful_stocks.append(stock)
    
    # Create email body
    email_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2>VN30 Stock Price Analysis Report</h2>
        <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        <p><strong>Stocks Processed:</strong> {len(successful_stocks)}/{len(stock_symbols)}</p>
        <hr>
        
        <h3>Successfully Processed:</h3>
        <p style="line-height: 1.8;">
            {", ".join(successful_stocks) if successful_stocks else "None"}
        </p>
        <hr>
        
        <h3>Analysis Performed:</h3>
        <ul>
            <li>Ensemble Model Training (RF + GB + SVR + Ridge)</li>
            <li>Model Evaluation with RMSE/MAE/MAPE metrics</li>
            <li>30-day Future Price Predictions</li>
            <li>Visualization Charts Generated</li>
            <li>Incremental Data Crawling (PostgreSQL)</li>
        </ul>
        <hr>
        
        <p><strong>Files Location:</strong> <code>output/{{stock_symbol}}/</code></p>
        <p>Best regards,<br><strong>VN30 Stock Prediction System</strong></p>
    </body>
    </html>
    """
    
    message = Mail(
        from_email=sender_email,
        to_emails=recipient_email,
        subject=f'VN30 Stock Analysis Report - {len(successful_stocks)} Stocks ({datetime.now().strftime("%Y-%m-%d")})',
        html_content=email_body,
    )
    
    try:
        attachments = []
        
        # Create summary CSV (first 5 stocks as sample)
        summary_data = []
        for stock in successful_stocks[:5]:
            paths = get_stock_file_paths(stock)
            if os.path.exists(paths["future_csv"]):
                df = pd.read_csv(paths["future_csv"])
                df["Stock"] = stock
                summary_data.append(df)
        
        if summary_data:
            summary_df = pd.concat(summary_data, ignore_index=True)
            summary_csv = os.path.join(settings.base_output_dir, "vn30_summary.csv")
            summary_df.to_csv(summary_csv, index=False)
            
            with open(summary_csv, "rb") as file:
                attachments.append(
                    Attachment(
                        FileContent(base64.b64encode(file.read()).decode()),
                        FileName("VN30_Predictions_Sample.csv"),
                        FileType("text/csv"),
                        Disposition("attachment"),
                    )
                )
        
        # Attach sample charts (first 3 stocks)
        for stock in successful_stocks[:3]:
            paths = get_stock_file_paths(stock)
            
            # Evaluation plot
            if os.path.exists(paths["plot"]):
                with open(paths["plot"], "rb") as file:
                    attachments.append(
                        Attachment(
                            FileContent(base64.b64encode(file.read()).decode()),
                            FileName(f"{stock}_evaluation.png"),
                            FileType("image/png"),
                            Disposition("attachment"),
                        )
                    )
            
            # Future prediction plot
            if os.path.exists(paths["future_plot"]):
                with open(paths["future_plot"], "rb") as file:
                    attachments.append(
                        Attachment(
                            FileContent(base64.b64encode(file.read()).decode()),
                            FileName(f"{stock}_future.png"),
                            FileType("image/png"),
                            Disposition("attachment"),
                        )
                    )
        
        # Add attachments to message
        if attachments:
            message.attachment = attachments
        
        # Send email
        api_key = settings.sendgrid_api_key
        if api_key and api_key != "YOUR_SENDGRID_API_KEY":
            client = SendGridAPIClient(api_key)
            response = client.send(message)
            logger.info(
                f"Email sent successfully. "
                f"Stocks: {len(successful_stocks)}/{len(stock_symbols)}, "
                f"Attachments: {len(attachments)}"
            )
            return True
        else:
            logger.warning("SendGrid API key not configured - skipping email")
            return True  # Don't fail the pipeline
        
    except Exception as error:
        logger.error(f"ERROR sending email: {str(error)}")
        return False


def send_training_completion_email(
    stock_symbol: str,
    metrics: dict,
    success: bool = True,
) -> bool:
    """
    Send email notification for a single stock training completion.
    
    Args:
        stock_symbol: Stock symbol that was trained
        metrics: Training metrics dict
        success: Whether training was successful
        
    Returns:
        True if email sent successfully
    """
    sender_email = settings.email_sender
    recipient_email = settings.email_recipient
    
    status = "Success" if success else "Failed"
    
    email_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2>Training Completion: {stock_symbol}</h2>
        <p><strong>Status:</strong> {status}</p>
        <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        <hr>
        
        <h3>Metrics:</h3>
        <ul>
            <li>RMSE: {metrics.get('rmse', 'N/A')}</li>
            <li>MAE: {metrics.get('mae', 'N/A')}</li>
            <li>MAPE: {metrics.get('mape', 'N/A')}%</li>
            <li>R²: {metrics.get('r2_score', 'N/A')}</li>
        </ul>
        
        <p>Best regards,<br><strong>VN30 Stock Prediction System</strong></p>
    </body>
    </html>
    """
    
    message = Mail(
        from_email=sender_email,
        to_emails=recipient_email,
        subject=f"[{status}] {stock_symbol} Training Complete",
        html_content=email_body,
    )
    
    try:
        api_key = settings.sendgrid_api_key
        if api_key and api_key != "YOUR_SENDGRID_API_KEY":
            client = SendGridAPIClient(api_key)
            client.send(message)
            logger.info(f"Training completion email sent for {stock_symbol}")
            return True
        return True
    except Exception as error:
        logger.error(f"ERROR sending training email: {str(error)}")
        return False

