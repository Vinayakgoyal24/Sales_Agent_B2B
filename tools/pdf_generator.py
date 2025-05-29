from io import BytesIO
from typing import List, TypedDict
from langchain.schema import Document  # or wherever Document is defined


def generate_pdf(quotation_text: str) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER)
    highlight_style = ParagraphStyle(name="Highlight", parent=styles["Normal"], textColor=colors.green, fontSize=12)
    elements = []

    # --- Header: Company Info ---

    logo_path = "otsuka_im.png"  # Replace with your actual logo path
    if os.path.exists(logo_path):
        logo = Image(logo_path, width=100, height=50)
        logo.hAlign = 'LEFT'
        elements.append(logo)

    elements.append(Paragraph("Otsuka Corporation", title_style))
    elements.append(Paragraph("Head Office, 2-18-4 Iidabashi, Chiyoda-ku, Tokyo 102-8573", styles["Normal"]))
    elements.append(Paragraph("Website: <a href='https://www.otsuka-shokai.co.jp/'>www.otsuka-shokai.co.jp</a>", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # --- Quotation Info ---
    elements.append(Paragraph("<b>🧾 Quotation</b>", styles["Heading2"]))
    today = datetime.now()
    validity = today + timedelta(days=7)
    elements.append(Paragraph(f"Date of Issue: {today.strftime('%Y-%m-%d')}", styles["Normal"]))
    elements.append(Paragraph(f"Validity: {validity.strftime('%Y-%m-%d')} (7 days)", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # --- Client Info (Dummy) ---
    elements.append(Paragraph("<b>📋 Client Information</b>", styles["Heading3"]))
    client_info = [
        "Client Name: Acme Solutions Pvt. Ltd.",
        "Client Address: 123, Innovation Tower, Marunouchi, Tokyo",
        "Contact Person: John Doe",
        "Email: john.doe@example.com",
        "Phone: +81 90-1234-5678"
    ]
    for line in client_info:
        elements.append(Paragraph(line, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # --- Quotation Parsing ---
    lines = quotation_text.strip().splitlines()
    table_data = []
    total_prices = []
    recommendation_lines = []
    current_quotation = ""
    inside_quote = False
    current_total_price= None

    def build_table(title, data, bg_color):
        styles = getSampleStyleSheet()
        cell_style = styles["Normal"]
        font_name = "Helvetica"
        font_size = 10

    # Convert data cells to Paragraphs (except header row)
        table_data = [data[0]]  # keep header row as-is (or also make Paragraph if needed)
        for row in data[1:]:
            wrapped_row = [Paragraph(str(cell), cell_style) for cell in row]
            table_data.append(wrapped_row)

    # Calculate dynamic column widths based on header row and wrapped content
        transposed_data = list(zip(*data))  # transpose original for max width calc
        col_widths = []
        for i, col in enumerate(transposed_data):
            max_width = max([stringWidth(str(item), font_name, font_size) for item in col])
            col_widths.append(min(max_width + 20, 200))  # add padding + cap width

    # Create table with dynamic column widths and styling
        table = Table(table_data, hAlign='LEFT', colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), bg_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.7, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), font_size),
            ('ALIGN', (2, 1), (2, -1), 'RIGHT'),
            ('ALIGN', (3, 1), (3, -1), 'CENTER'),
            ('VALIGN', (0, 1), (-1, -1), 'TOP'),  # vertical alignment to top
    ]))

        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading4"]))
        elements.append(table)

    for line in lines:
        line = line.strip()

        if line.startswith("## Quotation"):
            if table_data:
                build_table(current_quotation, table_data, bg_color=colors.HexColor("#4472C4"))
                if current_total_price:
                    elements.append(Paragraph(f"<b>Total Price ({current_quotation}):</b> ¥{current_total_price}", styles["Normal"]))
                    total_prices.append((current_quotation, current_total_price))
                    elements.append(Spacer(1, 10))
                table_data = []
            current_quotation = line.replace("##", "").strip()
            table_data.append(["Product Name", "Specs", "Price (¥)", "Qty"])
            current_total_price = None
            inside_quote = True

        elif line.startswith("Product Name:"):
            pname = line.split(":", 1)[1].strip()
        elif line.startswith("Specs:"):
            specs = line.split(":", 1)[1].strip()
        elif line.startswith("Price:"):
            price = line.split(":", 1)[1].strip()
        elif line.startswith("Quantity:"):
            qty = line.split(":", 1)[1].strip()
            table_data.append([pname, specs, price, qty])
        elif line.startswith("Total Price:"):
            current_total_price = line.split(":", 1)[1].strip()

        elif line.startswith("## Recommendation"):
            inside_quote = False
            if table_data:
                build_table(current_quotation, table_data, bg_color=colors.HexColor("#4472C4"))
                if current_total_price:
                    elements.append(Paragraph(f"<b>Total Price ({current_quotation}):</b> ¥{current_total_price}", styles["Normal"]))
                    total_prices.append((current_quotation, current_total_price))
                    elements.append(Spacer(1, 10))
            elements.append(Paragraph("<b>🎯 Recommendation</b>", styles["Heading3"]))
        elif not inside_quote and line:
            recommendation_lines.append(line)


    # Last table
    if table_data:
        build_table(current_quotation, table_data, bg_color=colors.HexColor("#4472C4"))

    # Pricing Summary
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>📊 Pricing Summary</b>", styles["Heading3"]))
    for qname, tprice in total_prices:
        elements.append(Paragraph(f"• {qname}: ¥{tprice}", styles["Normal"]))

    # Recommendation Section
    if recommendation_lines:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<b>✅ Best Recommendation</b>", styles["Heading3"]))
        for line in recommendation_lines:
            elements.append(Paragraph(line, highlight_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer