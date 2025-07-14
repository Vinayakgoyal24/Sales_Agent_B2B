from prometheus_client import Histogram, Counter

RETRIEVER_LATENCY = Histogram("retriever_latency_seconds", "Time taken by retriever to fetch context")
GENERATOR_LATENCY = Histogram("generator_latency_seconds", "Time taken by LLM to generate response")
PROMPT_LENGTH = Histogram(
    "prompt_length_tokens", 
    "Token length of generated prompt", 
    buckets=(0, 100, 200, 400, 800, 1200, 1600, 2000, 3000, 4000)
)
RECOMMENDED_PRODUCT_COUNTER = Counter(
    "recommended_product_total",
    "Counts how many times each product was recommended",
    ["product_name"]
)

TOKENS_GENERATED = Counter(
    "llm_response_tokens_total",
    "Total number of tokens generated in LLM responses"
)

LLM_RESPONSE_COUNT = Counter(
    "llm_response_count_total",
    "Total number of LLM responses generated"
)