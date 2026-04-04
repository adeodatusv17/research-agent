from langgraph.graph import END, StateGraph

from research_agent.agents.nodes.generate_answer_node import generate_answer_node
from research_agent.agents.nodes.query_analysis_node import query_analysis_node
from research_agent.agents.nodes.rerank_chunks_node import rerank_chunks_node
from research_agent.agents.nodes.retrieve_context_node import retrieve_context_node
from research_agent.agents.nodes.retrieve_sections_node import retrieve_sections_node
from research_agent.agents.nodes.retrieve_subsections_node import retrieve_subsections_node
from research_agent.agents.state.qa_state import QAState


def build_research_qa_graph():
    graph = StateGraph(QAState)
    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("retrieve_sections", retrieve_sections_node)
    graph.add_node("retrieve_subsections", retrieve_subsections_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("rerank_chunks", rerank_chunks_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("query_analysis")
    graph.add_edge("query_analysis", "retrieve_sections")
    graph.add_edge("retrieve_sections", "retrieve_subsections")
    graph.add_edge("retrieve_subsections", "retrieve_context")
    graph.add_edge("retrieve_context", "rerank_chunks")
    graph.add_edge("rerank_chunks", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()
