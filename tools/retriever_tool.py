# tools/retriever_tool.py
def build_retriever(vector_store):
    def retrieve(state):
        return {"context": vector_store.similarity_search(state["question"])}
    return retrieve
