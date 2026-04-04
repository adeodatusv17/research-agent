class IngestArxivPaperUseCase:
    def execute(self, arxiv_url: str) -> dict:
        return {"arxiv_url": arxiv_url, "status": "stub"}
