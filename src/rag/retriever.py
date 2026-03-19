from __future__ import annotations

from langchain_core.documents import Document

from rag.config import Settings, get_settings
from rag.vectorstore import VectorStore


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        settings: Settings | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._settings = settings or get_settings()

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k if k is not None else self._settings.retrieval_k
        return self._vector_store.similarity_search(query, k=k)

    def retrieve_with_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        k = k if k is not None else self._settings.retrieval_k
        return self._vector_store.similarity_search_with_score(query, k=k)
