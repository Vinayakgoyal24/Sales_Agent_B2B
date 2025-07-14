import os
import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity
from ragas import evaluate

load_dotenv()

# Constants
TOP_K = 5
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

# Initialize
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

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
os.environ["OPENAI_API_KEY"] = "sk-proj-UZIz7Fn3dIy1BHywWVrXT3BlbkFJImUsx697xQbmU9cc0hyM"
# Prompt
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


# Load eval data
def load_eval_data(path="ragas_eval_data.json") -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


# BM25 setup (load raw docs)
def load_all_docs() -> List[Document]:
    return vector_store.similarity_search("dummy", k=1000)


all_docs = load_all_docs()
bm25_corpus = [doc.page_content.split() for doc in all_docs]
bm25 = BM25Okapi(bm25_corpus)
bm25_doc_map = {i: doc for i, doc in enumerate(all_docs)}


# Hybrid search
def hybrid_retrieve(query: str, alpha: float, top_k=TOP_K) -> List[Document]:
    dense_results = vector_store.similarity_search_with_score(query, k=20)
    dense_dict = {doc.page_content: score for doc, score in dense_results}

    bm25_scores = bm25.get_scores(query.split())
    combined_scores = []

    for i, score in enumerate(bm25_scores):
        doc = bm25_doc_map[i]
        dense_score = dense_dict.get(doc.page_content, 0)
        final_score = alpha * dense_score + (1 - alpha) * score
        combined_scores.append((final_score, doc))

    combined_scores.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in combined_scores[:top_k]]


# RAG pipeline
def run_rag_pipeline(question: str, alpha: float = 0.5) -> Dict:
    docs = hybrid_retrieve(question, alpha)
    print(docs)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = prompt.invoke({"question": question, "context": context})
    answer = llm.invoke(messages).content
    return {
        "question": question,
        "contexts": [doc.page_content for doc in docs],
        "answer": answer
    }


# Prepare RAGAS samples
def prepare_ragas_samples(eval_data: List[Dict], alpha: float) -> List[Dict]:
    samples = []
    for row in tqdm(eval_data):
        rag_output = run_rag_pipeline(row["question"], alpha)
        samples.append({
            "question": rag_output["question"],
            "contexts": rag_output["contexts"],
            "answer": rag_output["answer"],
            "reference": row["ground_truth"]
        })
    with open(f"ragas_samples_hybrid_alpha{alpha}.pkl", "wb") as f:
        pickle.dump(samples, f)
    return samples


# Evaluate RAGAS
def run_ragas_eval(samples: List[Dict]) -> Dict:
    dataset = Dataset.from_list(samples)
    result = evaluate(dataset, metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity
    ])
    return result


# Plotting
def plot_ragas_results(results: List[Dict]):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_similarity"]
    for metric in metrics:
        plt.plot([r["alpha"] for r in results], [r[metric] for r in results], label=metric)

    plt.xlabel("alpha (dense weight)")
    plt.ylabel("Score")
    plt.title("RAGAS Metrics vs Hybrid alpha")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("hybrid_alpha_ragas_plot.png")
    plt.show()


# Main loop
def main():
    eval_data = load_eval_data()
    final_results = []

    for alpha in ALPHA_VALUES:
        print(f"\n🚀 Alpha = {alpha}")
        samples = prepare_ragas_samples(eval_data, alpha=alpha)
        scores = run_ragas_eval(samples)
        print(scores)
        #result = {metric: scores[metric] for metric in scores}
        #result["alpha"] = alpha
        #final_results.append(result)
        #print(f"✅ RAGAS @ alpha={alpha}: {result}")

    # Save + Plot
    #df = pd.DataFrame(final_results)
    #df.to_csv("hybrid_alpha_ragas_summary.csv", index=False)
    #plot_ragas_results(final_results)


if __name__ == "__main__":
    main()
