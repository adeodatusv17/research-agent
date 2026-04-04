class PgVectorIndex:
    def upsert_chunks(self, chunks: list[dict]) -> None:
        return None

    def similarity_search(self, query_embedding: list[float], limit: int = 5) -> list[dict]:
        return []
