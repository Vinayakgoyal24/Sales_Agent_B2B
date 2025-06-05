from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()

def send_whatsapp_pdf(recipient_number: str, media_url: str) -> str:
    try:
        client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )

        message = client.messages.create(
            from_=os.getenv("TWILIO_SANDBOX_NUMBER"),
            to=f"whatsapp:{recipient_number}",
            body="🧾 Here is your quotation from Otsuka Corporation.",
            media_url=[media_url]
        )

        return f"✅ WhatsApp message sent successfully! SID: {message.sid}"

    except Exception as e:
        return f"❌ Failed to send WhatsApp message: {e}"
