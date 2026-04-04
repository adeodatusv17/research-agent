from langgraph.graph import END, StateGraph

from research_agent.agents.state.analysis_state import AnalysisState


def build_reproducibility_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("score_reproducibility", lambda state: state)
    graph.set_entry_point("score_reproducibility")
    graph.add_edge("score_reproducibility", END)
    return graph.compile()
