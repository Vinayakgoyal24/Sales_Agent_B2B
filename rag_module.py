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

# Prometheus AI-metrics
from ai_metrics import AG_REQS, AG_LAT, TOKENS, ACTIVE_MODEL


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
    starttime=time.time()
    if feedback:
        query += f"\nAdditional feedback: {feedback}"
    results = vector_store.similarity_search(query)
    elapsed=  time.time() - starttime
    print(f"Retrieval: {elapsed}")
    return [doc.page_content for doc in results]

# Generation Function
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

def generate_answer(query: str, context: List[str], feedback: str = "", lang: str = "en") -> str:
    startg = time.time()
    ACTIVE_MODEL.labels("gpt-4o").set(1)

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

    messages = prompt.invoke({
        "question": query,
        "context": full_context
    })

    try:
        response = llm.invoke(messages)
        # ── record tokens & latency
        if hasattr(response, "usage"):
            TOKENS.labels("prompt").observe(getattr(response.usage, "prompt_tokens", 0))
            TOKENS.labels("completion").observe(getattr(response.usage, "completion_tokens", 0))
        AG_REQS.labels("success").inc()
        AG_LAT.observe(time.time() - startg)
        print(response)
        elapsedg = time.time() - startg
        print(f"Generator time: {elapsedg}")
        return response.content
    except Exception:
        AG_REQS.labels("error").inc()
        raise
    finally:
        ACTIVE_MODEL.labels("gpt-4o").set(0)



# (Optional) Extract Recommendation Text
def extract_recommendation_text(response: str) -> str:
    match = re.search(r"Recommendation[:\s]*([\s\S]+)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""



def generate_avatar_script(query, recommendation, feedback=""):
    avatar_prompt = f"""
    You are a digital sales avatar. Create a friendly and persuasive 1-minute pitch based on this:

    Recommendation:
    {recommendation}

    User Query & Feedback:
    {query}
    {feedback}

    Speak naturally and professionally.
    """
    messages = [
        SystemMessage(content="You are a professional sales avatar."),
        HumanMessage(content=avatar_prompt)
    ]
    response = llm.invoke(messages)
    return response.content