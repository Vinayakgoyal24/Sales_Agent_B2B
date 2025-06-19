import os
import json
from tqdm import tqdm
from typing import List, Dict
from datasets import Dataset
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity
from ragas import evaluate
from dotenv import load_dotenv
import pandas as pd
import pickle
import pandas as pd
from datetime import datetime
from rerankers import get_reranker, TOP_K
# Load environment variables
load_dotenv()

reranker = get_reranker('colbert')
# Azure LLM & Embedding setup
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
)

# Vectorstore (ChromaDB)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
import os
os.environ["OPENAI_API_KEY"] = "sk-proj-UZIz7Fn3dIy1BHywWVrXT3BlbkFJImUsx697xQbmU9cc0hyM"
# Prompt setup (same as used in your app)
from langchain.prompts import ChatPromptTemplate
prompt_template = [
    ("system",
     "You are a professional hardware sales assistant at Otsuka Shokai. Based on the user's request and the context, "
     "provide 2–3 detailed hardware configuration quotations. Each quotation should include:\n"
     "- Product Name\n- Specs\n- Price\n- Quantity\n- Total Price\n\n"
     "Use this structure:\n"
     "## Quotation 1\nProduct Name: ...\nSpecs: ...\nPrice: ...\nQuantity: ...\n...\nTotal Price: ...\n"
     "## Quotation 2 ...\n\n"
     "Then provide a clear comparison of the quotations and recommend the best one based on:\n"
     "- Price\n- Suitability for the user's need\n- Performance vs cost.\n"
     "Use a section titled:\n## Recommendation\n"
     "Mention why the chosen quote is the best and highlight key differences with others.\n\n"
     "Keep tone professional and brief. Do not fabricate information if context is insufficient."),
    ("human", "Question: {question}\n\nContext:\n{context}")
]
prompt = ChatPromptTemplate.from_messages(prompt_template)

# Load evaluation JSON data
def load_eval_data(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return json.load(f)

# Run your RAG pipeline per sample
def run_rag_pipeline(question: str, log_to_csv: bool = True) -> Dict:
    rough_docs = vector_store.similarity_search(question, k=10)
    try:
        refined_docs = reranker(question, rough_docs)
        # expose metrics to Streamlit
        rerank_metrics = {
            **reranker.metrics(),
            "first_pass_k": len(rough_docs)
        }
    except Exception as e:
        print("[RERANK-ERROR]", e)
        refined_docs = rough_docs[:5]
        rerank_metrics = {"error": str(e), "first_pass_k": len(rough_docs)}

    context = "\n\n".join(doc.page_content for doc in refined_docs)
    messages = prompt.invoke({"question": question, "context": context})
    answer = llm.invoke(messages).content
    rows = []
    if log_to_csv:
        rows = []
        for i, doc in enumerate(rough_docs):
            rows.append({
                "question": question,
                "retrieval_stage": "dense",
                "rank": i + 1,
                "content": doc.page_content
            })
        for i, doc in enumerate(refined_docs):
            rows.append({
                "question": question,
                "retrieval_stage": "reranked",
                "rank": i + 1,
                "content": doc.page_content
            })

        df = pd.DataFrame(rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"retrieval_t5_{timestamp}.csv"
        df.to_csv(file_path, index=False)
        print(f"[LOG] Saved retrieval trace to {file_path}")

    return {
        "question": question,
        "contexts": [doc.page_content for doc in refined_docs],
        "answer": answer
    }

# Prepare RAGAS-format samples
def prepare_ragas_samples(eval_data: List[Dict]) -> List[Dict]:
    ragas_samples = []
    for row in tqdm(eval_data):
        rag_output = run_rag_pipeline(row["question"])
        ragas_samples.append({
            "question": rag_output["question"],
            "contexts": rag_output["contexts"],
            "answer": rag_output["answer"],
            "reference": row["ground_truth"]
        })
    with open("ragas_samples.pkl", "wb") as f:
        pickle.dump(ragas_samples, f)
    return ragas_samples

# Evaluate using RAGAS
def run_ragas_eval(samples: List[Dict]):
    dataset = Dataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity]
    )
    return result

# Main runner
def main():
    file_path = "ragas_eval_data.json"  # Assuming this is in the same folder
    eval_data = load_eval_data(file_path)
    ragas_samples = prepare_ragas_samples(eval_data)
    results = run_ragas_eval(ragas_samples)
    print(type(results))
    
    print("\n--- RAGAS Evaluation Results ---")

    print(results)
main()

