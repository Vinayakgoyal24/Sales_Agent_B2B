from rank_bm25 import BM25Okapi
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
import re

# Load .env variables
load_dotenv()

# Setup Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
)

# Load Vector Store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

#  Custom Tokenizer for BM25
def tokenize_for_bm25(text: str):
    text = re.sub(r"[^\w\s]", " ", text.lower())  # Remove punctuation, lowercase
    return text.split()

# Fetch Sample Docs
docs = vector_store.similarity_search("suggest me a good gaming mouse", k=5)

#Show Tokens
for i, doc in enumerate(docs):
    print(f"\n📄 Original Doc #{i}:\n{doc.page_content}")
    print(f"\n🔍 BM25 Tokens:\n{tokenize_for_bm25(doc.page_content)}")
