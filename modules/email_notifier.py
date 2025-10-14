"""
Email notification module using SendGrid
"""
import os
import base64
from datetime import datetime
import pandas as pd

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

from config import EMAIL_CONFIG, BASE_OUTPUT_DIR
from modules.model_trainer import get_stock_file_paths


def send_email_notification(stock_symbols):
    """
    Send email summary report for all processed stocks.
    
    Args:
        stock_symbols: List of all stock symbols
        
    Returns:
        bool: True if email sent successfully
    """
    sender_email = EMAIL_CONFIG['sender']
    recipient_email = EMAIL_CONFIG['recipient']
    
    # Count successful stocks (those with models)
    successful_stocks = []
    for stock in stock_symbols:
        paths = get_stock_file_paths(stock)
        if os.path.exists(paths['model']):
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
            <li>LSTM Model Training (100 epochs)</li>
            <li>Model Evaluation with RMSE/MAE metrics</li>
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
        html_content=email_body
    )
    
    try:
        attachments = []
        
        # Create summary CSV (first 5 stocks as sample)
        summary_data = []
        for stock in successful_stocks[:5]:
            paths = get_stock_file_paths(stock)
            if os.path.exists(paths['future_csv']):
                df = pd.read_csv(paths['future_csv'])
                df['Stock'] = stock
                summary_data.append(df)
        
        if summary_data:
            summary_df = pd.concat(summary_data, ignore_index=True)
            summary_csv = os.path.join(BASE_OUTPUT_DIR, 'vn30_summary.csv')
            summary_df.to_csv(summary_csv, index=False)
            
            with open(summary_csv, 'rb') as file:
                attachments.append(Attachment(
                    FileContent(base64.b64encode(file.read()).decode()),
                    FileName('VN30_Predictions_Sample.csv'),
                    FileType('text/csv'),
                    Disposition('attachment')
                ))
        
        # Attach sample charts (first 3 stocks)
        for stock in successful_stocks[:3]:
            paths = get_stock_file_paths(stock)
            
            # Evaluation plot
            if os.path.exists(paths['plot']):
                with open(paths['plot'], 'rb') as file:
                    attachments.append(Attachment(
                        FileContent(base64.b64encode(file.read()).decode()),
                        FileName(f'{stock}_evaluation.png'),
                        FileType('image/png'),
                        Disposition('attachment')
                    ))
            
            # Future prediction plot
            if os.path.exists(paths['future_plot']):
                with open(paths['future_plot'], 'rb') as file:
                    attachments.append(Attachment(
                        FileContent(base64.b64encode(file.read()).decode()),
                        FileName(f'{stock}_future.png'),
                        FileType('image/png'),
                        Disposition('attachment')
                    ))
        
        # Add attachments to message
        if attachments:
            message.attachment = attachments
        
        # Send email
        api_key = EMAIL_CONFIG['sendgrid_api_key']
        if api_key and api_key != 'YOUR_SENDGRID_API_KEY':
            client = SendGridAPIClient(api_key)
            response = client.send(message)
            print(f"Email sent successfully")
            print(f"   Stocks: {len(successful_stocks)}/{len(stock_symbols)}")
            print(f"   Attachments: {len(attachments)}")
            return True
        else:
            print("WARNING: SendGrid API key not configured - skipping email")
            return True  # Don't fail the pipeline
        
    except Exception as error:
        print(f"ERROR sending email: {str(error)}")
        return False

