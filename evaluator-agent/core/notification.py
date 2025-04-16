"""
Notification functionality using SendGrid for SMS notifications.
"""

import os
import logging
from typing import Optional
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Configure logging
logger = logging.getLogger(__name__)

# Get SendGrid API key from environment
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "noreply@clustering-system.com")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

async def send_notification(
    eval_id: str,
    phone_number: Optional[str],
    carrier_gateway: Optional[str]
) -> bool:
    """
    Send an SMS notification using SendGrid email-to-SMS.
    
    Args:
        eval_id: Evaluation ID
        phone_number: Recipient phone number
        carrier_gateway: SMS gateway for the carrier (e.g., vtext.com for Verizon)
        
    Returns:
        success: Whether the notification was sent successfully
    """
    if not SENDGRID_API_KEY:
        logger.warning("SendGrid API key not configured. Skipping notification.")
        return False
        
    if not phone_number or not carrier_gateway:
        logger.warning("Phone number or carrier gateway not provided. Skipping notification.")
        return False
    
    try:
        # Construct the email-to-SMS address
        to_email = f"{phone_number}@{carrier_gateway}"
        
        # Prepare the message
        message = Mail(
            from_email=FROM_EMAIL,
            to_emails=to_email,
            subject="Clustering Evaluation Complete",
            plain_text_content=(
                f"Your clustering analysis is complete. "
                f"View report at: {BASE_URL}/api/v1/evaluator/report/{eval_id}"
            )
        )
        
        # Send the message
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        
        logger.info(f"Notification sent to {to_email} with status code {response.status_code}")
        return response.status_code in [200, 201, 202]
        
    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")
        return False