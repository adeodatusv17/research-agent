from langgraph.graph import END, StateGraph

from research_agent.agents.nodes.adaptive_retry_node import adaptive_retry_node
from research_agent.agents.nodes.critique_answer_node import critique_answer_node
from research_agent.agents.nodes.evaluate_answer_node import evaluate_answer_node
from research_agent.agents.nodes.evidence_diagnostics_node import evidence_diagnostics_node
from research_agent.agents.nodes.generate_answer_node import generate_answer_node
from research_agent.agents.nodes.planner_node import planner_node
from research_agent.agents.nodes.query_analysis_node import query_analysis_node
from research_agent.agents.nodes.rerank_chunks_node import rerank_chunks_node
from research_agent.agents.nodes.revise_answer_node import revise_answer_node
from research_agent.agents.nodes.retrieve_context_node import retrieve_context_node
from research_agent.agents.nodes.retrieve_sections_node import retrieve_sections_node
from research_agent.agents.nodes.retrieve_subsections_node import retrieve_subsections_node
from research_agent.agents.nodes.verify_answer_node import verify_answer_node
from research_agent.agents.state.qa_state import QAState
from research_agent.services.qa_orchestration_service import (
    route_after_critique,
    route_after_diagnostics,
    route_after_generation,
    route_after_verification,
    should_run_planner,
)


def build_research_qa_graph():
    graph = StateGraph(QAState)
    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retrieve_sections", retrieve_sections_node)
    graph.add_node("retrieve_subsections", retrieve_subsections_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("rerank_chunks", rerank_chunks_node)
    graph.add_node("evidence_diagnostics", evidence_diagnostics_node)
    graph.add_node("adaptive_retry", adaptive_retry_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("verify_answer", verify_answer_node)
    graph.add_node("critique_answer", critique_answer_node)
    graph.add_node("revise_answer", revise_answer_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)

    graph.set_entry_point("query_analysis")
    graph.add_conditional_edges(
        "query_analysis",
        should_run_planner,
        {
            "planner": "planner",
            "retrieve_sections": "retrieve_sections",
        },
    )
    graph.add_edge("planner", "retrieve_sections")
    graph.add_edge("retrieve_sections", "retrieve_subsections")
    graph.add_edge("retrieve_subsections", "retrieve_context")
    graph.add_edge("retrieve_context", "rerank_chunks")
    graph.add_edge("rerank_chunks", "evidence_diagnostics")
    graph.add_conditional_edges(
        "evidence_diagnostics",
        route_after_diagnostics,
        {
            "adaptive_retry": "adaptive_retry",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("adaptive_retry", "retrieve_sections")
    graph.add_conditional_edges(
        "generate_answer",
        route_after_generation,
        {
            "verify_answer": "verify_answer",
            "evaluate_answer": "evaluate_answer",
        },
    )
    graph.add_conditional_edges(
        "verify_answer",
        route_after_verification,
        {
            "critique_answer": "critique_answer",
            "evaluate_answer": "evaluate_answer",
        },
    )
    graph.add_conditional_edges(
        "critique_answer",
        route_after_critique,
        {
            "revise_answer": "revise_answer",
            "evaluate_answer": "evaluate_answer",
        },
    )
    graph.add_edge("revise_answer", "verify_answer")
    graph.add_edge("evaluate_answer", END)
    return graph.compile()
