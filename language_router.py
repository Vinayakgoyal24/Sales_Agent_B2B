# language_router.py  –  heuristic + optional LLM detector
import os, re, functools
from typing import Tuple, Literal

from langdetect import detect, DetectorFactory
import nltk, regex
from nltk.tokenize import sent_tokenize
DetectorFactory.seed = 0
nltk.download("punkt", quiet=True)

# ---------- CONFIG --------------------------------------------------------
MODE = os.getenv("LANG_DETECT_MODE", "heuristic")      # "heuristic" | "llm" | "hybrid"
LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")   # reuse existing
# -------------------------------------------------------------------------

Lang = Literal["en", "ja"]
_JA_RE = regex.compile(r"[ぁ-んァ-ヶ一-龯]")
_PATTERNS = {
    "ja": [r"日本語で.*(答えて|お願い|返事して)", r"answer.*(in|with).*japanese"],
    "en": [r"英語で.*(答えて|お願い|返事して)", r"answer.*(in|with).*english"],
}
def _instr(text: str) -> Lang | None:
    low = text.lower()
    for lang, pats in _PATTERNS.items():
        for p in pats:
            if regex.search(p, low):
                return lang
    return None

# ---------- Heuristic detector -------------------------------------------
def _sent_lang(sent: str) -> Lang:
    if _JA_RE.search(sent):
        return "ja"
    try:
        return "ja" if detect(sent) == "ja" else "en"
    except Exception:
        return "en"

def _dominant(text: str) -> Lang:
    votes = {"en": 0, "ja": 0}
    for s in sent_tokenize(text.strip() or "a"):
        votes[_sent_lang(s)] += len(s)
    return "ja" if votes["ja"] > votes["en"] else "en"

# ---------- LLM detector --------------------------------------------------
@functools.lru_cache(maxsize=1024)
def _llm_detect(text: str) -> Lang:
    """Returns 'en' or 'ja' ONLY, uses <10 tokens."""
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
    resp = llm.invoke([( "system", system), ("user", text.strip()[:500])])
    return "ja" if "ja" in resp.content.lower() else "en"

# ---------- Public API ----------------------------------------------------
def resolve_langs(text: str) -> Tuple[Lang, Lang]:
    """
    Returns (query_lang, answer_lang)
    """
    req = _instr(text)                 # explicit override?

    if MODE == "llm":
        lang = _llm_detect(text)
    elif MODE == "hybrid":
        lang = _dominant(text)
        if lang == "en" and req is None:   # low confidence case – ask LLM
            lang = _llm_detect(text)
    else:                                  # "heuristic"
        lang = _dominant(text)

    answer_lang = req or lang
    return lang, answer_lang
