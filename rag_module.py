import os
import re
from typing import List
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
from dotenv import load_dotenv
import time
from metrics import RETRIEVER_LATENCY, GENERATOR_LATENCY, PROMPT_LENGTH, TOKENS_GENERATED, LLM_RESPONSE_COUNT


load_dotenv()


# Load environment variables (make sure they are set)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    chunk_size=1024,
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

def load_csv_as_documents(folder_path="data"):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file_name)).head(20)
            for _, row in df.iterrows():
                content = "\n".join([f"{k}: {v}" for k, v in row.items()])
                documents.append(Document(page_content=content, metadata={"source": file_name}))
    return documents


import tiktoken

def count_tokens(text: str) -> int:
    if not isinstance(text, str):
        print("⚠️ Prompt token counting failed: input is not a string")
        return 0

    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # or your model
        return len(encoding.encode(text))
    except Exception as e:
        print(f"⚠️ Token counting failed: {e}")
        return 0


# If vector store is empty, populate it
if vector_store._collection.count() == 0:
    docs = load_csv_as_documents("data")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    vector_store.add_documents(splits)
    vector_store.persist()

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional hardware sales assistant at Otsuka Shokai. Based on the user's request and the context, provide 2–3 detailed hardware configuration quotations. 
Each quotation should include:
- Product Name
- Specs
- Price
- Quantity
- Total Price

Use this structure:
## Quotation 1
Product Name: ...
Specs: ...
Price: ...
Quantity: ...
...
Total Price: ...
## Quotation 2 ...

Then provide a clear comparison of the quotations and recommend the best one based on:
- Price
- Suitability for the user's need
- Performance vs cost.

Use a section titled:
## Recommendation

Mention why the chosen quote is the best and highlight key differences with others.

Keep tone professional and brief. Do not fabricate information if context is insufficient."""),

    ("human", "Question: {question}\n\nContext:\n{context}")
])

# Retrieval Function
def retrieve_relevant_chunks(query: str, feedback: str = "") -> List[str]:
    from metrics import RETRIEVER_LATENCY
    starttime=time.time()
    if feedback:
        query += f"\nAdditional feedback: {feedback}"
    results = vector_store.similarity_search(query)
    RETRIEVER_LATENCY.observe(time.time()-starttime)
    elapsed=  time.time() - starttime
    print(f"Retrieval: {elapsed}")
    return [doc.page_content for doc in results]

# Generation Function
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

def generate_answer(query: str, context: List[str], feedback: str = "", lang: str = "en") -> str:
    from metrics import GENERATOR_LATENCY, PROMPT_LENGTH
    startg = time.time()
    full_context = "\n\n".join(context)

    if feedback:
        query += f"\n\n{feedback}"

    # Build dynamic system prompt
    if lang == "ja":
        system_prompt = (
            "あなたは大塚商会のプロフェッショナルなハードウェア営業アシスタントです。"
            "ユーザーのリクエストとコンテキストに基づいて、2-3件の詳細なハードウェア構成見積もりを提供してください。\n"
            "各見積もりには以下を含めてください:\n"
            "- 製品名\n- スペック\n- 価格\n- 数量\n- 合計価格\n\n"
            "以下の構造を使用してください:\n"
            "## 見積もり 1\n製品名: ...\n...\n合計価格: ...\n\n"
            "最後に、以下の観点に基づいて、最適な見積もりを推薦してください:\n"
            "- 価格\n- ユーザーの要望との適合性\n- 性能とコストのバランス\n\n"
            "## 推薦\n"
            "なぜその見積もりが最適なのかを簡潔に説明してください。\n"
            "情報が不足している場合は、情報が不十分であることを明記し、仮定を避けてください。"
        )
    else:
        system_prompt = (
    "You are a professional hardware sales assistant at Otsuka Shokai.\n\n"
    "Always respond in the same language as the user's query.\n\n"
    "Based on the user's request and the context, provide 2–3 detailed hardware configuration quotations.\n\n"
    "Each quotation should include the following fields:\n"
    "- Product Name\n"
    "- Specs\n"
    "- Price\n"
    "- Quantity\n"
    "- Total Price\n\n"
    "Use this exact structure:\n\n"
    "## Quotation 1 (or ## 見積もり1 in Japanese)\n"
    "Product Name: ... (or 商品名: ...)\n"
    "Specs: ... (or 仕様: ...)\n"
    "Price: ... (or 価格: ...)\n"
    "Quantity: ... (or 数量: ...)\n"
    "Total Price: ... (or 合計金額: ...)\n\n"
    "Then provide a comparison and a clear recommendation.\n\n"
    "Use a section titled:\n\n"
    "## Recommendation (or ## 推奨案 in Japanese)\n\n"
    "In the recommendation, mention:\n"
    "- Which quotation is the best\n"
    "- Why it is the best (cost, specs, suitability)\n"
    "- Key differences with other quotations\n\n"
    "Keep the tone professional and concise.\n\n"
    "Do NOT fabricate specs or prices if context is missing.\n\n"
    "📝 IMPORTANT:\n"
    "If the user's query is in Japanese, your entire response — including all section titles and field labels — must be in Japanese using the exact terms:\n"
    "見積もり, 商品名, 仕様, 価格, 数量, 合計金額, 推奨案.\n\n"
    "Otherwise, respond entirely in English using the English terms and structure."
)
    # Build dynamic prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_count = len(enc.encode(query + "\n\n" + full_context))
        PROMPT_LENGTH.observe(token_count)
    except Exception as e:
        print("⚠️ Prompt token counting failed:", e)

    messages = prompt.invoke({
        "question": query,
        "context": full_context
    })

    response = llm.invoke(messages)
    GENERATOR_LATENCY.observe(time.time() - startg)

    print(response)
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_count_re = len(enc.encode(response.content))
        TOKENS_GENERATED.inc(token_count_re)
        LLM_RESPONSE_COUNT.inc()
    except Exception as e:
        print("⚠️ Prompt token counting failed:", e)
    elapsedg = time.time() - startg
    print(f"Generator time: {elapsedg}")
    return response.content



# (Optional) Extract Recommendation Text
def extract_recommendation_text(response: str) -> str:
    match = re.search(r"Recommendation[:\s]*([\s\S]+)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""
