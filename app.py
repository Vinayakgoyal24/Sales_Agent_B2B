import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import tiktoken
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime, timedelta
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from io import BytesIO
from reportlab.platypus import Image
from reportlab.pdfbase.pdfmetrics import stringWidth



# Load env vars
load_dotenv()

# --- Initialize Azure Clients ---
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# --- Load CSVs and Create Documents ---
def load_csv_as_documents(folder_path="data"):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file_name)).head(20)
            for _, row in df.iterrows():
                content = "\n".join([f"{k}: {v}" for k, v in row.items()])
                documents.append(Document(page_content=content, metadata={"source": file_name}))
    return documents

if vector_store._collection.count() == 0:
    with st.spinner("Indexing documents..."):
        docs = load_csv_as_documents("data")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        vector_store.add_documents(splits)
        vector_store.persist()

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional hardware sales assistant at Otsuka Corporation. Based on the user's request and the context, provide 2–3 detailed hardware configuration quotations. "
     "Each quotation should include:\n"
     "- Product Name\n- Specs\n- Price\n- Quantity\n- Total Price\n\n"
     "Use this structure:\n"
     "## Quotation 1\nProduct Name: ...\nSpecs: ...\nPrice: ...\nQuantity: ...\n...\nTotal Price: ...\n"
     "## Quotation 2 ...\n\n"
     "Then provide a **clear comparison** of the quotations and recommend the best one based on:\n"
     "- Price\n- Suitability for the user's need\n- Performance vs cost.\n"
     "Use a section titled:\n"
     "## Recommendation\n"
     "Mention why the chosen quote is the best and highlight key differences with others.\n\n"
     "Keep tone professional and brief. Do not fabricate information if context is insufficient.")
    ,
    ("human", "Question: {question}\n\nContext:\n{context}")
])



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
        elements.append(Paragraph(f"• {qname}: ${tprice}", styles["Normal"]))

    # Recommendation Section
    if recommendation_lines:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<b>✅ Best Recommendation</b>", styles["Heading3"]))
        for line in recommendation_lines:
            elements.append(Paragraph(line, highlight_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer



# --- LangGraph App Logic ---
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    return {"context": vector_store.similarity_search(state["question"])}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# --- Streamlit UI ---

st.title("💻 Computer Hardware Sales Assistant")

# 1. Initialize session state to store result
if "result" not in st.session_state:
    st.session_state.result = None

user_query = st.text_input("Enter your query:", placeholder="E.g., Best PC setup for video editing...")

# 2. Run graph and store in session_state
if user_query and st.button("💬 Get Recommendation"):
    with st.spinner("Processing your query..."):
        st.session_state.result = graph.invoke({"question": user_query})

# 3. Display result if available
if st.session_state.result:
    st.subheader("💡 Suggested Answer")
    st.write(st.session_state.result["answer"])

    # Token usage
    enc = tiktoken.encoding_for_model("gpt-4")
    input_tokens = len(enc.encode(user_query))
    output_tokens = len(enc.encode(st.session_state.result["answer"]))
    st.markdown(f"🔢 **Input tokens:** {input_tokens} | **Output tokens:** {output_tokens} | **Total:** {input_tokens + output_tokens}")

    # 4. Offer PDF Download
    pdf_bytes = generate_pdf(st.session_state.result["answer"])
    st.download_button(
        label="📄 Download Quotation as PDF",
        data=pdf_bytes,
        file_name="hardware_quotation.pdf",
        mime="application/pdf"
    )