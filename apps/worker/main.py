from research_agent.agents.graphs.paper_analysis_graph import build_paper_analysis_graph


def run() -> None:
    graph = build_paper_analysis_graph()
    print(f"Worker ready with graph: {graph}")


if __name__ == "__main__":
    run()
