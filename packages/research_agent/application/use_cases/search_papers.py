class SearchPapersUseCase:
    def execute(self, query: str) -> dict:
        return {"query": query, "results": []}
