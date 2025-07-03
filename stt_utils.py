# stt_utils.py
import tempfile, os, torch
from io import BytesIO
from faster_whisper import WhisperModel
from language_router import resolve_langs   # keeps answer-/query-lang in sync

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model  = WhisperModel("medium", device=_device, 
                       compute_type="int8_float16" if _device=="cuda" else "int8")

def transcribe_wav(raw: bytes) -> dict:
    """Return {'text': str, 'detected_lang': 'en'|'ja'}"""
    if not raw:
        return {"text": "", "detected_lang": "en"}
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw); path = tmp.name
    try:
        segs, info = _model.transcribe(path, beam_size=5, suppress_blank=True)
        text = "".join(s.text for s in segs).strip()
        _, detected = resolve_langs(text)          # 🔄 reuse existing heuristic/LLM
        return {"text": text, "detected_lang": detected}
    finally:
        os.remove(path)
