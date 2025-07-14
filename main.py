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
from language_detector import detect_lang
from metrics import RETRIEVER_LATENCY, GENERATOR_LATENCY, PROMPT_LENGTH, RECOMMENDED_PRODUCT_COUNTER
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Updated for your Vite port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)

    return response


from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Latency for HTTP requests in seconds",
    ["endpoint"]
)

NEW_USERS = Counter("new_users_total", "Number of new users signed up", ["day"])
RECURRING_USERS = Counter("recurring_users_total", "Number of returning users (via login)", ["day"])
DAILY_SESSIONS = Counter("daily_sessions_total", "Daily session count", ["day"])
PDF_GENERATED = Counter("pdf_generated_total", "PDF files generated")
PDF_FAILED = Counter("pdf_failed_total", "PDF generation failed")
PDF_LATENCY = Histogram("pdf_generation_duration_seconds", "Time taken to generate PDF")
SLIDES_GENERATED = Counter("slides_generated_total", "PPTX slides generated")
SLIDES_FAILED = Counter("slides_failed_total", "PPTX slide generation failed")
SLIDES_LATENCY = Histogram("slides_generation_duration_seconds", "Time taken to generate slides")
EMAILS_SENT = Counter("emails_sent_total", "Total emails successfully sent")
EMAILS_FAILED = Counter("emails_failed_total", "Total emails that failed to send")
EMAIL_SEND_LATENCY = Histogram("email_send_duration_seconds", "Time taken to send email")
QUERY_REQUESTS = Counter("query_requests_total", "Total /query endpoint requests")
QUERY_FAILURES = Counter("query_failures_total", "Failed /query requests")
QUERY_DURATION = Histogram("query_duration_seconds", "Duration of /query endpoint")
QUERY_COMPLETED = Counter("query_completed_sessions", "Sessions where all info was collected")

# DB setup
DATABASE_URL = "sqlite:///./users.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Password hash & JWT setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
SECRET_KEY = "your_super_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # not used now

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

# Utility functions
def get_user_by_email(db, email: str):
    return db.query(User).filter(User.email == email).first()

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def hash_password(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def today_str():
    return datetime.now().strftime("%Y-%m-%d")



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

import re

def track_recommended_products(response: str):
    try:
        matches = re.findall(r"Product Name:\s*(.+)", response)
        for product in matches:
            # Take only the first 3 words
            cleaned = " ".join(product.strip().lower().split()[:3])
            RECOMMENDED_PRODUCT_COUNTER.labels(product_name=cleaned).inc()
    except Exception as e:
        print("❌ Failed to track recommended products:", e)


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
    QUERY_REQUESTS.inc()
    start_time = time.time()

    try:
        info = req.collected_info or {}
        user_input = req.question.lower()

        steps = ["name", "company", "email", "contact", "requirement", "quantity"]
        lang = req.language if req.language in PROMPTS else "en-US"
        prompts = PROMPTS[lang]

        current_step = req.step or steps[0]
        all_info_collected = all(k in info for k in steps)

        if all_info_collected:
            QUERY_COMPLETED.inc()
            full_query = f"{info['requirement']} - Quantity: {info['quantity']}"
            context = retrieve_relevant_chunks(full_query, req.question)
            response_text = generate_answer(full_query, context, req.question)
            track_recommended_products(response_text)

            QUERY_DURATION.observe(time.time() - start_time)
            return {
                "response": response_text,
                "has_quotation": True,
                "done": True,
                "collected_info": info,
            }

        if current_step in steps:
            info[current_step] = req.question

        next_index = steps.index(current_step) + 1 if current_step in steps else 0

        if next_index >= len(steps):
            QUERY_COMPLETED.inc()
            full_query = f"{info['requirement']} - Quantity: {info['quantity']}"
            context = retrieve_relevant_chunks(full_query, "")
            response_text = generate_answer(full_query, context, "")
            track_recommended_products(response_text)
            QUERY_DURATION.observe(time.time() - start_time)
            return {
                "response": response_text,
                "has_quotation": True,
                "done": True,
                "collected_info": info,
            }
        else:
            next_step = steps[next_index]
            QUERY_DURATION.observe(time.time() - start_time)
            return {
                "response": prompts[next_step],
                "has_quotation": False,
                "step": next_step,
                "collected_info": info,
                "done": False,
            }

    except Exception as e:
        QUERY_FAILURES.inc()
        QUERY_DURATION.observe(time.time() - start_time)
        raise e


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
    start_time = time.time()
    try:
        pdf_stream = generate_pdf(data.quotation_text, data.client_info)
        
        PDF_GENERATED.inc()  # ✅ Increment counter on success
        duration = time.time() - start_time
        PDF_LATENCY.observe(duration)  # ⏱ Track latency

        return StreamingResponse(
            pdf_stream,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=quotation.pdf"}
        )
    except Exception as e:
        PDF_FAILED.inc()  # ❌ Increment failure counter
        return {"status": "error", "message": str(e)}

@app.post("/generate-slides")
def generate_slides_endpoint(data: PDFRequest):
    start_time = time.time()
    try:
        ppt_content = generate_slides(data.quotation_text, data.client_info)
        ppt_content.seek(0)

        SLIDES_GENERATED.inc()  # ✅ Successful generation
        SLIDES_LATENCY.observe(time.time() - start_time)  # ⏱ Track latency

        return StreamingResponse(
            ppt_content,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": "attachment; filename=quotation.pptx"}
        )
    except Exception as e:
        SLIDES_FAILED.inc()  # ❌ Increment failure count
        return {"status": "error", "message": str(e)}


@app.post("/send-email")
def send_email_endpoint(email_data: EmailRequest):
    start_time = time.time()

    success, message = send_email_with_attachment(
        email_data.to_emails,
        email_data.subject,
        email_data.body
    )

    latency = time.time() - start_time
    EMAIL_SEND_LATENCY.observe(latency)

    if success:
        EMAILS_SENT.inc()
        return {"status": "success", "message": message}
    else:
        EMAILS_FAILED.inc()
        return {"status": "error", "message": message}



from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

class UserCreate(BaseModel):
    email: EmailStr
    password: str

@app.post("/signup")
def signup(user: UserCreate, db: SessionLocal = Depends(get_db)):
    if get_user_by_email(db, user.email):
        raise HTTPException(status_code=409, detail="User already exists")

    hashed = hash_password(user.password)
    new_user = User(email=user.email, hashed_password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # 📊 Track new user signup
    NEW_USERS.labels(day=today_str()).inc()
    DAILY_SESSIONS.labels(day=today_str()).inc()

    return {"msg": "User created successfully"}


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: SessionLocal = Depends(get_db)):
    user = get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # 📊 Track recurring login
    RECURRING_USERS.labels(day=today_str()).inc()
    DAILY_SESSIONS.labels(day=today_str()).inc()

    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/me")
def read_me(token: str = Depends(oauth2_scheme), db: SessionLocal = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user = get_user_by_email(db, email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"email": user.email}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

