from langgraph.graph import END, StateGraph

from research_agent.agents.state.analysis_state import AnalysisState


def build_paper_analysis_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("parse_paper", lambda state: state)
    graph.set_entry_point("parse_paper")
    graph.add_edge("parse_paper", END)
    return graph.compile()
