from fastapi import FastAPI, Form
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional
from rag_module import retrieve_relevant_chunks, generate_answer
from pydantic import BaseModel, validator
from pdf_utils import generate_pdf # Replace with actual import
from email_utils import send_email_with_attachment
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from pydantic import BaseModel
from query_handler import update_collected_info, get_next_question, session_store
from ppt_utils import generate_slides# Make sure you import it
from fastapi.responses import StreamingResponse
from io import BytesIO
import csv
import os
from fastapi import Body
from datetime import datetime



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Updated for your Vite port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RetrieveRequest(BaseModel):
    query: str
    feedback: str = ""

class RetrieveResponse(BaseModel):
    context: List[str]

class GenerateRequest(BaseModel):
    query: str
    context: List[str]
    feedback: str = ""

class GenerateResponse(BaseModel):
    response: str

class QueryRequest(BaseModel):
    question: str

import re
import json

def parse_llm_output(response: str):
    quotations = []
    recommendation = ""

    # Split into quotations and recommendation
    parts = response.split("##")
    for part in parts:
        part = part.strip()
        if part.lower().startswith("quotation"):
            lines = part.splitlines()
            q_products = []
            total_price = 0
            for i, line in enumerate(lines):
                if line.lower().startswith("product name:"):
                    name = line.split(":", 1)[1].strip()
                    specs = lines[i+1].split(":", 1)[1].strip()
                    price = float(re.findall(r"[\d.]+", lines[i+2])[0])
                    quantity = int(re.findall(r"\d+", lines[i+3])[0])
                    total_price_product = float(re.findall(r"[\d.]+", lines[i+4])[0])
                    q_products.append({
                        "name": name,
                        "specs": specs,
                        "price": price,
                        "quantity": quantity,
                        "total_price": total_price_product,
                    })
                elif line.lower().startswith("overall total price:"):
                    total_price = float(re.findall(r"[\d.]+", line)[0])

            quotations.append({
                "products": q_products,
                "total": total_price,
            })
        elif part.lower().startswith("recommendation"):
            recommendation = part.replace("Recommendation", "").strip()

    return {
        "quotations": quotations,
        "recommendation": recommendation,
    }


class ChatQueryRequest(BaseModel):
    question: str
    step: Optional[str] = None
    collected_info: Optional[Dict[str, str]] = {}

@app.post("/query")
def smart_query_handler(req: ChatQueryRequest):
    info = req.collected_info or {}
    user_input = req.question.lower()

    steps = ["name", "company", "email", "contact", "requirement", "quantity"]
    prompts = {
        "name": "May I know your name?",
        "company": "Can you please share your company name?",
        "email": "Thanks for sharing! Please provide your email address.",
        "contact": "Great, Thanks! Can I know your contact number?",
        "requirement": "Thanks for the information! Please tell me what products you are looking for?",
        "quantity": "Great! I will definitely help you get the best quotation! How many units do you need by the way?",
    }

    current_step = req.step or steps[0]

    # ✅ Check if all info is already collected
    all_info_collected = all(k in info for k in steps)

    if all_info_collected:
        # Treat input as feedback and regenerate quotation
        full_query = f"{info['requirement']} - Quantity: {info['quantity']}"
        context = retrieve_relevant_chunks(full_query, req.question)  # using feedback here
        response_text = generate_answer(full_query, context, req.question)

        return {
            "response": response_text,
            "has_quotation": True,
            "done": True,
            "collected_info": info,
        }

    # 🔽 Normal flow below (collecting info step-by-step)
    if current_step in steps:
        info[current_step] = req.question

    next_index = steps.index(current_step) + 1 if current_step in steps else 0

    if next_index >= len(steps):
        full_query = f"{info['requirement']} - Quantity: {info['quantity']}"
        context = retrieve_relevant_chunks(full_query, "")
        response_text = generate_answer(full_query, context, "")
        return {
            "response": response_text,
            "has_quotation": True,
            "done": True,
            "collected_info": info,
        }
    else:
        next_step = steps[next_index]
        return {
            "response": prompts[next_step],
            "has_quotation": False,
            "step": next_step,
            "collected_info": info,
            "done": False,
        }

class PDFRequest(BaseModel):
    quotation_text: str
    client_info: Dict[str, str]


class EmailRequest(BaseModel):
    to_email: EmailStr
    subject: str
    body: str


from fastapi.responses import StreamingResponse
from io import BytesIO

@app.post("/generate-pdf")
def generate_pdf_endpoint(data: PDFRequest):
    try:
        pdf_stream = generate_pdf(data.quotation_text, data.client_info)  # Already returns BytesIO

        return StreamingResponse(
            pdf_stream,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=quotation.pdf"}
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/generate-slides")
def generate_slides_endpoint(data: PDFRequest):
    try:
        ppt_content = generate_slides(data.quotation_text, data.client_info)
        ppt_content.seek(0)  # just to be safe, can be skipped if already done inside generate_slides
        return StreamingResponse(
            ppt_content,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": "attachment; filename=quotation.pptx"}
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/send-email")
def send_email_endpoint(email_data: EmailRequest):
    success, message = send_email_with_attachment(
        email_data.to_email,
        email_data.subject,
        email_data.body
    )
    status = "success" if success else "error"
    return {"status": status, "message": message}
