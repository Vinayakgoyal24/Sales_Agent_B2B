from fastapi import FastAPI, Form
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional
from rag_module import retrieve_relevant_chunks, generate_answer, generate_avatar_script
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
from language_detector import detect_lang

from ai_metrics import start_metrics_server, HTTP_REQS, HTTP_LAT
from time import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # +two helpers
from fastapi import Response

from fastapi import FastAPI, Form, Request, BackgroundTasks   # ← add BackgroundTasks
from gtts import gTTS                                         # ← NEW
import uuid, re, os, csv, json, shutil
import logging                              # uuid & re already exist

from fastapi.responses import StreamingResponse, FileResponse, Response


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")



# ▼ ADD START: gTTS helper
def _make_tts_mp3(text: str, lang_code: str = "en") -> str:
    """Generate temporary MP3 with gTTS and return its path."""
    tmp_path = f"/tmp/{uuid.uuid4().hex}.mp3"
    gTTS(text=text, lang=lang_code).save(tmp_path)
    return tmp_path
# ▲ ADD END



app = FastAPI()
start_metrics_server(9100)      # <──── listens on /metrics


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080",
                    "http://localhost:5173", ],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def _prom_mw(request: Request, call_next):
    t0 = time()
    resp: Response | None = None                        # ← ensure defined
    try:
        resp = await call_next(request)
        return resp
    finally:
        dur = time() - t0
        status = getattr(resp, "status_code", 500)      # if resp is None
        HTTP_REQS.labels(request.method, request.url.path, status).inc()
        HTTP_LAT.labels(request.method, request.url.path).observe(dur)


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


PROMPTS = {
    "en-US": {
        "name": "May I know your name?",
        "company": "Which company are you representing?",
        "email": "Could you share your email so I can send the quotation?",
        "contact": "May I have your contact number?",
        "requirement": "What product are you looking for?",
        "quantity": "How many units do you need?",
    },
    "ja-JP": {
        "name": "お名前を教えていただけますか？",
        "company": "ご所属の会社名を教えてください。",
        "email": "見積もりを送付するためメールアドレスを教えてください。",
        "contact": "ご連絡先番号を教えていただけますか？",
        "requirement": "ご希望の商品を教えてください。",
        "quantity": "必要な数量を教えてください。",
    },
}

class ChatQueryRequest(BaseModel):
    question: str
    step: Optional[str] = None
    collected_info: Optional[Dict[str, str]] = {}
    language: str = "en-US"

@app.post("/query")
def smart_query_handler(req: ChatQueryRequest):
    info = req.collected_info or {}
    user_input = req.question.lower()

    steps = ["name", "company", "email", "contact", "requirement", "quantity"]
    lang = req.language if req.language in PROMPTS else "en-US"
    prompts = PROMPTS[lang]

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
    to_emails: List[str]
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
        email_data.to_emails,
        email_data.subject,
        email_data.body
    )
    status = "success" if success else "error"
    return {"status": status, "message": message}


class AvatarScriptRequest(BaseModel):
    query: str
    recommendation: str
    feedback: str = ""

@app.post("/generate-avatar-script")
def avatar_script_endpoint(data: AvatarScriptRequest):
    script = generate_avatar_script(
        data.query,
        data.recommendation,
        data.feedback,
    )

    # ▼ ADD START
    logging.info("Avatar pitch script ↓\n%s\n-------------------------------", script)

    # optional: dump to file
    dbg_dir = "debug_scripts"
    os.makedirs(dbg_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"pitch_{timestamp}.txt"
    with open(os.path.join(dbg_dir, fname), "w", encoding="utf-8") as f:
        f.write(script)
    # ▲ ADD END

    return {"script": script}


# audio endpoint
class AudioRequest(BaseModel):
    script: str

@app.post("/generate-avatar-audio")
def avatar_audio_ep(req: AudioRequest, background_tasks: BackgroundTasks):
    lang = "ja" if re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", req.script) else "en"
    mp3_path = _make_tts_mp3(req.script, lang)

    # ▼ ADD START
    debug_audio_dir = "debug_audio"
    os.makedirs(debug_audio_dir, exist_ok=True)
    debug_copy = os.path.join(debug_audio_dir,
                          f"pitch_{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
    shutil.copy(mp3_path, debug_copy)
    logging.info("Saved debug MP3 → %s", debug_copy)
    # ▲ ADD END

    # keep cleanup of the temp file
    background_tasks.add_task(os.remove, mp3_path)
    return FileResponse(mp3_path,
                        media_type="audio/mpeg",
                        filename="avatar.mp3")