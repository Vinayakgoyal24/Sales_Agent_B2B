import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Setup port number and server name

smtp_port = 587                 # Standard secure SMTP port
smtp_server = "smtp.gmail.com"  # Google SMTP Server

# Set up the email lists
email_from = "vinayak.otsuka@gmail.com"
email_list = ["vinayakgoyal2410@gmail.com"]

# Define the password (better to reference externally)
pswd = "djjvyfubleftjmwh" # As shown in the video this password is now dead, left in as example only


# name the email subject
subject = "New email from TIE with attachments!!"



# Define the email function (dont call it email!)
def send_emails(email_list):
    for person in email_list:
        body = """
        line 1
        line 2
        line 3
        etc
        """

        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = person
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        filename = "static/hardware_quotation.pdf"
        try:
            with open(filename, 'rb') as attachment:
                attachment_package = MIMEBase('application', 'octet-stream')
                attachment_package.set_payload(attachment.read())
                encoders.encode_base64(attachment_package)
                attachment_package.add_header('Content-Disposition', f"attachment; filename={filename}")
                msg.attach(attachment_package)
        except FileNotFoundError:
            print(f"Attachment file not found: {filename}")
            continue

        text = msg.as_string()

        try:
            print("Connecting to server...")
            TIE_server = smtplib.SMTP(smtp_server, smtp_port)
            TIE_server.starttls()
            TIE_server.login(email_from, pswd)
            print("Successfully connected to server")

            TIE_server.sendmail(email_from, person, text)
            print(f"Email sent to {person}")

            TIE_server.quit()
        except Exception as e:
            print(f"Failed to send email to {person}. Error: {e}")


if __name__ == "__main__":
    send_emails(email_list)