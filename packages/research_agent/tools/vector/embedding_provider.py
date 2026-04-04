class EmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]
