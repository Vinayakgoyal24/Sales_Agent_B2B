import os
import re
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from io import BytesIO
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase.pdfmetrics import stringWidth
from datetime import datetime, timedelta
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from PIL import Image as PILImage
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
import base64
import tiktoken
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from io import BytesIO
import os
import re

# --- STT and TTS libraries ---
from io import BytesIO
import threading
import tempfile
import torch
from faster_whisper import WhisperModel
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS


def send_email_with_attachment(to_email: str, subject: str, body: str):

    smtp_port = 587
    smtp_server = "smtp.gmail.com"
    sender_email = "vinayak.otsuka@gmail.com"
    pswd = "djjvyfubleftjmwh"

    if not sender_email or not pswd:
        return False, "Missing sender email or password in environment variables."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
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
        return False, f"Attachment file not found: {filename}"

    text = msg.as_string()

    try:
        print("Connecting to server...")
        TIE_server = smtplib.SMTP(smtp_server, smtp_port)
        TIE_server.starttls()
        TIE_server.login(sender_email, pswd)
        print("Successfully connected to server")

        TIE_server.sendmail(sender_email, to_email, text)
        TIE_server.quit()
        return True, f"Email successfully sent to {to_email}"
    except Exception as e:
        return False, f"Failed to send email to {to_email}. Error: {e}"