"""
Prometheus metrics for the Sales-bot / RAG backend.
"""

from time import time
from prometheus_client import (
    start_http_server, Counter, Histogram, Summary, Gauge
)

# ── generic HTTP metrics ────────────────────────────────────────────────
HTTP_REQS = Counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "path", "status"]
)
HTTP_LAT = Histogram(
    "http_request_latency_seconds", "Request latency",
    ["method", "path"],
    buckets=(.05,.1,.2,.5,1,2,5,10)
)

# ── AI-specific metrics ────────────────────────────────────────────────
AG_REQS = Counter("ai_agent_requests_total", "Agent requests", ["result"])
AG_LAT  = Histogram("ai_agent_latency_seconds", "Agent latency seconds")
TOKENS  = Summary ("ai_agent_tokens", "Tokens used", ["direction"])
ACTIVE_MODEL = Gauge("ai_agent_active_model", "Model active (1/0)", ["model"])


# ── e-mail pipeline ───────────────────────────────────────────────────
EMAIL_SENT   = Counter(
    "emails_sent_total",
    "Quotation e-mails sent",
    ["result"]                         # success / error
)
EMAIL_LAT    = Histogram(
    "email_latency_seconds",
    "Latency of sending e-mail",
    buckets=(0.1, 0.3, 0.5, 1, 2, 5)
)

# ── endpoint metrics ─────────────────────────────────────────
QUERY_REQS = Counter("query_requests", "POST /query", ["result"])
QUERY_LAT  = Histogram("query_latency_seconds", "Latency of /query",
                       buckets=(0.2,0.5,1,2,5,10))

PDF_GEN    = Counter("pdf_generated", "PDF quotations generated", ["result"])
PDF_LAT    = Histogram("pdf_gen_latency_seconds", "Latency of /generate-pdf")

PPT_GEN    = Counter("ppt_generated", "PPT quotations generated", ["result"])
PPT_LAT    = Histogram("ppt_gen_latency_seconds", "Latency of /generate-slides")

# ── RAG helpers ──────────────────────────────────────────────
RETRIEVE_LAT = Histogram("rag_retrieval_latency_seconds",
                         "Vector-store similarity_search latency",
                         buckets=(0.05,0.1,0.25,0.5,1,2,5))
GENERATE_LAT = Histogram("rag_generation_latency_seconds",
                         "LLM quotation generation latency",
                         buckets=(0.5,1,2,5,10,20))

# ── product popularity ────────────────────────────────────────────────
PROD_RECOMMENDED = Counter(
    "product_recommended_total",
    "Products recommended to users",
    ["sku", "name"]
)
PROD_REQUESTED   = Counter(
    "product_requested_total",
    "Products the user explicitly asked for",
    ["sku", "name"]
)


def start_metrics_server(port: int = 9100):
    """Expose /metrics.  Call once at program startup."""
    start_http_server(port)
