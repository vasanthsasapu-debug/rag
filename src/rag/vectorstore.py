from __future__ import annotations

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from rag.config import Settings, get_settings
from rag.embeddings import EmbeddingModel


class VectorStore:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding_model = embedding_model
        self._store = Chroma(
            collection_name=self._settings.chroma_collection_name,
            embedding_function=embedding_model.langchain_embeddings,
            persist_directory=self._settings.chroma_persist_directory,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self._store.add_documents(documents)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k=k)

    def delete_collection(self) -> None:
        self._store.delete_collection()

    def count(self) -> int:
        return self._store._collection.count()
