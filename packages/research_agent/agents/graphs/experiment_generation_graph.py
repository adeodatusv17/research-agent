from langgraph.graph import END, StateGraph

from research_agent.agents.nodes.apply_defaults_node import apply_defaults_node
from research_agent.agents.nodes.generate_code_node import generate_code_node
from research_agent.agents.nodes.generate_config_node import generate_config_node
from research_agent.agents.nodes.infer_missing_fields_node import infer_missing_fields_node
from research_agent.agents.nodes.load_analysis_node import load_analysis_node
from research_agent.agents.nodes.store_metadata_node import store_metadata_node
from research_agent.agents.nodes.validate_artifact_node import validate_artifact_node
from research_agent.agents.nodes.verify_repositories_node import verify_repositories_node
from research_agent.agents.nodes.write_files_node import write_files_node
from research_agent.agents.state.experiment_state import ExperimentState


def build_experiment_generation_graph():
    graph = StateGraph(ExperimentState)
    graph.add_node("load_analysis", load_analysis_node)
    graph.add_node("verify_repositories", verify_repositories_node)
    graph.add_node("apply_defaults", apply_defaults_node)
    graph.add_node("infer_missing_fields", infer_missing_fields_node)
    graph.add_node("generate_config", generate_config_node)
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("write_files", write_files_node)
    graph.add_node("validate_artifact", validate_artifact_node)
    graph.add_node("store_metadata", store_metadata_node)

    graph.set_entry_point("load_analysis")
    graph.add_edge("load_analysis", "verify_repositories")
    graph.add_edge("verify_repositories", "apply_defaults")
    graph.add_edge("apply_defaults", "infer_missing_fields")
    graph.add_edge("infer_missing_fields", "generate_config")
    graph.add_edge("generate_config", "generate_code")
    graph.add_edge("generate_code", "write_files")
    graph.add_edge("write_files", "validate_artifact")
    graph.add_edge("validate_artifact", "store_metadata")
    graph.add_edge("store_metadata", END)
    return graph.compile()
