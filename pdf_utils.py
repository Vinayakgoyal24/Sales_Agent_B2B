import os
import time
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
from reportlab.platypus import PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT

pdfmetrics.registerFont(TTFont("NotoSansJP-Regular", "statics/fonts/NotoSansJP-VariableFont_wght.ttf"))
pdfmetrics.registerFont(TTFont("NotoSansJP", "statics/fonts/NotoSansJP-Bold.ttf"))

def contains_japanese(text):
    return bool(re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", text))

def generate_pdf(quotation_text: str, client_info: dict) -> BytesIO:
    startpdf=time.time()
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer,
                            pagesize=A4,
                            rightMargin=30, leftMargin=30,
                            topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()

    has_japanese= contains_japanese(quotation_text)
    font_regular = "NotoSansJP-Regular" if has_japanese else "Helvetica"
    font_bold = "NotoSansJP" if has_japanese else "Helvetica-Bold"
    # Styles
    title_style = ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontName=font_bold)
    normal_style = ParagraphStyle(name="NormalText", parent=styles["Normal"], fontName=font_regular, fontSize=12, leading=12, alignment=TA_LEFT)
    highlight_style = ParagraphStyle(name="Highlight", parent=styles["Normal"], fontName=font_regular, textColor=colors.black, fontSize=12)
    heading2_style = ParagraphStyle(name="Heading2", parent=styles["Heading2"], fontName=font_bold)
    heading3_style = ParagraphStyle(name="Heading3", parent=styles["Heading3"], fontName=font_bold)
    heading4_style = ParagraphStyle(name="Heading4", parent=styles["Heading4"], fontName=font_bold)


    elements = []

    # --- Header: Company Info ---
    logo_path = "otsuka_im.png"
    if os.path.exists(logo_path):
        logo = Image(logo_path, width=100, height=50)
        logo.hAlign = "LEFT"
        elements.append(logo)

    elements.append(Paragraph("Otsuka Shokai", title_style))
    elements.append(Paragraph("Head Office, 2-18-4 Iidabashi, Chiyoda-ku, Tokyo 102-8573", styles["Normal"]))
    elements.append(Paragraph("Website: <a href='https://www.otsuka-shokai.co.jp/'>www.otsuka-shokai.co.jp</a>",
                              styles["Normal"]))
    elements.append(Spacer(1, 12))

    # --- Quotation Info ---
    header_text = "--- 見積もり ---" if has_japanese else "--- Quotation ---"
    elements.append(Paragraph(f"<b>{header_text}</b>", heading2_style))
    today = datetime.now()
    validity = today + timedelta(days=7)
    elements.append(Paragraph(f"{'1. 発行日' if has_japanese else '1. Date of Issue'}: {today.strftime('%Y-%m-%d')}", normal_style))
    elements.append(Paragraph(f"{'2. 有効期限' if has_japanese else '2. Validity'}: {validity.strftime('%Y-%m-%d')} (7 days)", normal_style))

    # --- Client Info (Dummy) ---
    client_header = "--- クライアント情報 ---" if has_japanese else "--- Client Information ---"
    elements.append(Paragraph(f"<b>{client_header}</b>", heading3_style))
    if has_japanese:
        client_info_lines = [
        f"<b>1. クライアント名:</b> {client_info.get('name', '')}",
        f"<b>2. 会社名:</b> {client_info.get('company', '')}",
        f"<b>3. メール:</b> {client_info.get('email', '')}",
        f"<b>4. 電話番号:</b> {client_info.get('contact', '')}",
    ]
    else:
        client_info_lines = [
        f"<b>1. Client Name:</b> {client_info.get('name', '')}",
        f"<b>2. Client Company:</b> {client_info.get('company', '')}",
        f"<b>3. Email:</b> {client_info.get('email', '')}",
        f"<b>4. Phone:</b> {client_info.get('contact', '')}",
    ]

    for line in client_info_lines:
        elements.append(Paragraph(line, normal_style))
    elements.append(Spacer(1, 12))

    # --- Parse LLM output and build tables ---
    lines = quotation_text.strip().splitlines()
    table_data = []
    total_prices = []
    price_qty_list = []
    recommendation_lines = []
    current_quotation = ""
    inside_quote = False

    def build_table(title, data, bg_color):
        tbl = [data[0]] + [[Paragraph(str(cell), normal_style) for cell in row] for row in data[1:]]
        col_widths = [min(max(stringWidth(str(item), font_regular, 10) + 20 for item in col), 200) for col in zip(*data)]
        table = Table(tbl, hAlign="LEFT", colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), bg_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), font_regular),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (2, 1), (2, -1), "RIGHT"),
            ("ALIGN", (3, 1), (3, -1), "CENTER"),
            ("VALIGN", (0, 1), (-1, -1), "TOP"),
        ]))
        elements.append(Paragraph(f"<b>{title}</b>", heading4_style))
        elements.append(table)

    for line in lines:
        line = line.strip()
        if line.startswith("## Quotation") or line.startswith("## 見積もり"):
            if table_data:
                subtotal = sum(p * q for p, q in price_qty_list)
                build_table(current_quotation, table_data, colors.HexColor("#4472C4"))
                elements.append(Paragraph(f"<b>- Total Price ({current_quotation}):</b> ¥{subtotal:,.0f}", normal_style))
                elements.append(Spacer(1, 10))
                total_prices.append((current_quotation, f"{subtotal:,.0f}"))
            current_quotation = line.replace("##", "").strip()
            table_data = [["商品名", "仕様", "価格", "数量"]] if has_japanese else [["Product Name", "Specs", "Price", "Qty"]]
            price_qty_list = []
            inside_quote = True
        elif line.startswith("Product Name:") or line.startswith("商品名:"):
            pname = line.split(":", 1)[1].strip()
        elif line.startswith("Specs:") or line.startswith("仕様:"):
            specs = line.split(":", 1)[1].strip()
        elif line.startswith("Price:") or line.startswith("価格:"):
            price = float(re.sub(r"[^\d.]", "", line.split(":", 1)[1].strip()) or 0)
        elif line.startswith("Quantity:") or line.startswith("数量:"):
            qty = int(line.split(":", 1)[1].strip() or 0)
            table_data.append([pname, specs, f"{price:,.0f}", str(qty)])
            price_qty_list.append((price, qty))
        elif line.startswith("## Recommendation") or line.startswith("## 推奨案"):
            if table_data and inside_quote:
                subtotal = sum(p * q for p, q in price_qty_list)
                build_table(current_quotation, table_data, colors.HexColor("#4472C4"))
                elements.append(Paragraph(f"<b>Total Price ({current_quotation}):</b> ¥{subtotal:,.0f}", normal_style))
                elements.append(Spacer(1, 10))
                total_prices.append((current_quotation, f"{subtotal:,.0f}"))
            inside_quote = False
        elif not inside_quote and line:
            recommendation_lines.append(line)

    if table_data and inside_quote:
        subtotal = sum(p * q for p, q in price_qty_list)
        build_table(current_quotation, table_data, colors.HexColor("#4472C4"))
        elements.append(Paragraph(f"<b>Total Price ({current_quotation}):</b> ¥{subtotal:,.0f}", normal_style))
        total_prices.append((current_quotation, f"{subtotal:,.0f}"))

    # Summary & Recommendation
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>{'--- 価格まとめ ---' if has_japanese else '--- Pricing Summary ---'}</b>", heading3_style))
    for qname, tprice in total_prices:
        elements.append(Paragraph(f"• {qname}: ¥{tprice}", normal_style))

    if recommendation_lines:
        elements.append(PageBreak())
        elements.append(Spacer(1, 12))
        rec_header = "--- ベスト推奨案 ---" if contains_japanese(quotation_text) else "--- Best Recommendation ---"
        elements.append(Paragraph(f"<b>{rec_header}</b>", title_style))

        for line in recommendation_lines:
            if contains_japanese(line):
                elements.append(Paragraph(line, highlight_style))
            else:
                elements.append(Paragraph(line, normal_style))


    doc.build(elements)
    buffer.seek(0)
    with open("static/hardware_quotation.pdf", "wb") as f:
        f.write(buffer.getvalue())
    print(f"pdf download: {time.time() - startpdf:.2f} seconds")
    return buffer