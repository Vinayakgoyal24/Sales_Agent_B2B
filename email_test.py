import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

load_dotenv()

msg = EmailMessage()
msg["Subject"] = "Test Email from Streamlit App"
msg["From"] = os.getenv("EMAIL_SENDER")
msg["To"] = "vinayakgoyal2410.com"
msg.set_content("This is a test email.")

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_PASSWORD"))
        smtp.send_message(msg)
    print("✅ Email sent successfully.")
except Exception as e:
    print("❌ Failed to send email:", e)
