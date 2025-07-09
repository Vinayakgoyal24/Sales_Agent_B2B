import re
import uuid
import phonenumbers
from email_validator import validate_email, EmailNotValidError


# In-memory session storage
session_store: dict[str, dict] = {}

def get_session(session_id: str):
    """
    Create / return a session record.  Leave language unset so the back-end
    can decide later (EN/JA toggle or auto-detect).
    """
    return session_store.setdefault(session_id, {"info": {}})

required_fields = ["name", "company", "email", "phone", "requirement", "quantity"]

def extract_email(text):
    match = re.search(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b", text)
    return match.group() if match else None

def is_valid_email(email):
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def extract_phone(text):
    match = re.search(r'\+?\d[\d -]{8,}\d', text)
    if match:
        try:
            number = phonenumbers.parse(match.group(), "IN")
            if phonenumbers.is_valid_number(number):
                return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
        except:
            return None
    return None

def extract_quantity(text):
    match = re.search(r'\b\d+\b', text)
    return int(match.group()) if match else None

def update_collected_info(info, message):
    """Try to fill missing fields from user's message"""
    if "email" not in info:
        email = extract_email(message)
        if email and is_valid_email(email):
            info["email"] = email

    if "phone" not in info:
        phone = extract_phone(message)
        if phone:
            info["phone"] = phone

    if "quantity" not in info:
        qty = extract_quantity(message)
        if qty:
            info["quantity"] = qty

    for field in ["name", "company", "requirement"]:
        if field not in info and len(message.split()) <= 10:
            info[field] = message.strip().title()

    return info

def get_next_question(info):
    missing = [field for field in required_fields if field not in info]

    if not missing:
        return "Thanks! I have all the details. Would you like me to generate the quotation?", True, missing

    polite_questions = {
        "name": "May I know your name?",
        "company": "Which company or organization are you representing?",
        "email": "Could you please share your email so I can send the quotation?",
        "phone": "May I have your contact number?",
        "requirement": "What specific product or requirement do you have?",
        "quantity": "How many units are you looking for?",
    }

    return polite_questions[missing[0]], False, missing
