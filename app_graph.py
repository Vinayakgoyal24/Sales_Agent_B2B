# app_graph.py or agent.py
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def build_graph(retriever, generator):
    graph = StateGraph(State)
    graph.add_node("retrieve", retriever)
    graph.add_node("generate", generator)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.set_entry_point(START)
    return graph.compile()
