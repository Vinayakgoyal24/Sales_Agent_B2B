# chains/generate_quotation.py
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

prompt = ChatPromptTemplate.from_messages([
    ("system", "..."),  # keep your detailed prompt here
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def build_generation_chain(llm):
    def generate_input(state):
        context = "\n\n".join(doc.page_content for doc in state["context"])
        return {"question": state["question"], "context": context}
    
    return prompt | RunnableLambda(generate_input) | llm
