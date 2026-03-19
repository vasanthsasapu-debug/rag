from __future__ import annotations

from rag.config import Settings, get_settings
from rag.embeddings import EmbeddingModel
from rag.generator import Generator
from rag.ingestion import DocumentIngester
from rag.retriever import Retriever
from rag.vectorstore import VectorStore


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._ingester = DocumentIngester(self._settings)
        embedding_model = EmbeddingModel(self._settings)
        self._vector_store = VectorStore(embedding_model, self._settings)
        self._retriever = Retriever(self._vector_store, self._settings)
        self._generator = Generator(self._settings)

    def ingest_text(self, text: str, metadata: dict | None = None) -> int:
        chunks = self._ingester.ingest(text=text, metadata=metadata)
        self._vector_store.add_documents(chunks)
        return len(chunks)

    def ingest_file(self, file_path: str) -> int:
        chunks = self._ingester.ingest(source=file_path)
        self._vector_store.add_documents(chunks)
        return len(chunks)

    def query(self, question: str) -> dict:
        docs = self._retriever.retrieve(question)
        answer = self._generator.generate(question, docs)
        sources = [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ]
        return {"answer": answer, "sources": sources, "num_sources": len(sources)}

    def clear(self) -> None:
        self._vector_store.delete_collection()
