# language_detector.py
import os, functools, regex
from typing import Literal, Tuple
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0           # reproducible
Lang = Literal["en", "ja"]

# ---------------- regex (cheap, old behaviour) -----------------
_JP_CHARS = regex.compile(r"[ぁ-んァ-ヶ一-龯]")
def _regex(text: str) -> Lang:
    return "ja" if _JP_CHARS.search(text) else "en"

# ---------------- heuristic (langdetect + voting) --------------
from nltk.tokenize import sent_tokenize
def _heuristic(text: str) -> Lang:
    votes = {"en": 0, "ja": 0}
    for s in sent_tokenize(text.strip() or "a"):
        try:
            lang = detect(s)
        except Exception:
            lang = "en"
        votes["ja"] += len(s) if lang == "ja" else 0
        votes["en"] += len(s) if lang != "ja" else 0
    return "ja" if votes["ja"] > votes["en"] else "en"

# ---------------- LLM detector (Azure GPT) ----------------------
@functools.lru_cache(maxsize=512)
def _llm(text: str) -> Lang:
    from langchain.chat_models import AzureChatOpenAI
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
        max_tokens=1,
    )
    system = (
        "You are a language detection agent. "
        "Answer with a single token: 'en' for English dominant queries or 'ja' for Japanese."
        " Consider explicit instructions like 'answer in Japanese'. "
        "Reply ONLY 'en' or 'ja'."
    )
    resp = llm.invoke([("system", system), ("user", text[:500])])
    return "ja" if "ja" in resp.content.lower() else "en"

# ---------------- public helper --------------------------------
_MODE = os.getenv("LANG_DETECT_MODE", "regex")      # default keeps old feel

def detect_lang(text: str) -> Lang:
    if _MODE == "regex":
        return _regex(text)
    if _MODE == "heuristic":
        return _heuristic(text)
    if _MODE == "llm":
        return _llm(text)
    # hybrid → fast heuristic, fall back to llm on obvious ENG bias
    lang = _heuristic(text)
    if lang == "en" and _JP_CHARS.search(text):     # mixed & ambiguous
        lang = _llm(text)
    return lang
