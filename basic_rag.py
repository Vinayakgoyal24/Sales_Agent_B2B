import os
import pandas as pd
from dotenv import load_dotenv
from typing_extensions import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langgraph.graph import START, StateGraph

load_dotenv()

# Load LLM & Embeddings
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
)

# Function: Load top 1000 rows per CSV
def load_csv_as_documents(folder_path="data", max_rows_per_file=5) -> List[Document]:
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file_name), nrows=max_rows_per_file)
            for _, row in df.iterrows():
                content = "\n".join([f"{k}: {v}" for k, v in row.items()])
                documents.append(Document(page_content=content, metadata={"source": file_name}))
    return documents

# Load and split data
docs = load_csv_as_documents("data")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# Create vector store with persistence
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# Add to vector store if not already persisted
if not os.path.exists("./chroma_langchain_db/index"):
    vector_store.add_documents(splits)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context to answer questions. Say 'I don't know' if unsure."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# State type
class State(dict):
    question: str
    context: List[Document]
    answer: str

# Graph steps
def retrieve(state: State):
    return {"context": vector_store.similarity_search(state["question"])}

def generate(state: State):
    context_text = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": context_text})
    response = llm.invoke(messages, stream=False)
    
    # Print token usage
    if hasattr(response, "usage"):
        print("Tokens used:", response.usage)
    
    return {"answer": response.content}

# Create and run graph
graph = StateGraph(State).add_sequence([retrieve, generate]).add_edge(START, "retrieve").compile()

response = graph.invoke({
    "question": "Suggest a laptop for software development with at least 16GB RAM and SSD."
})

print("\nAnswer:\n", response["answer"])
