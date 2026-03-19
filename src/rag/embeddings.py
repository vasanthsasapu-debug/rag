from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from rag.config import Settings, get_settings


class EmbeddingModel:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = OpenAIEmbeddings(
            model=self._settings.openai_embedding_model,
            openai_api_key=self._settings.openai_api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)

    @property
    def langchain_embeddings(self) -> OpenAIEmbeddings:
        return self._model
