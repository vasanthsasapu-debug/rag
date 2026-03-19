from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.config import Settings
from rag.pipeline import RAGPipeline

DUMMY_SETTINGS = Settings(openai_api_key="test-key")


@pytest.fixture()
def pipeline() -> RAGPipeline:
    with (
        patch("rag.pipeline.EmbeddingModel") as mock_emb,
        patch("rag.pipeline.VectorStore") as mock_vs,
        patch("rag.pipeline.Retriever") as mock_ret,
        patch("rag.pipeline.Generator") as mock_gen,
        patch("rag.pipeline.DocumentIngester") as mock_ing,
    ):
        p = RAGPipeline(settings=DUMMY_SETTINGS)
        # Expose mocks for assertions
        p._mock_ingester = mock_ing.return_value
        p._mock_vector_store = mock_vs.return_value
        p._mock_retriever = mock_ret.return_value
        p._mock_generator = mock_gen.return_value
        yield p


def test_ingest_text_returns_chunk_count(pipeline: RAGPipeline) -> None:
    chunks = [Document(page_content=f"chunk {i}") for i in range(3)]
    pipeline._mock_ingester.ingest.return_value = chunks

    count = pipeline.ingest_text("Some long text to ingest")

    pipeline._mock_ingester.ingest.assert_called_once_with(
        text="Some long text to ingest", metadata=None
    )
    pipeline._mock_vector_store.add_documents.assert_called_once_with(chunks)
    assert count == 3


def test_query_returns_expected_structure(pipeline: RAGPipeline) -> None:
    doc = Document(page_content="relevant info", metadata={"source": "doc.txt"})
    pipeline._mock_retriever.retrieve.return_value = [doc]
    pipeline._mock_generator.generate.return_value = "The answer is 42."

    result = pipeline.query("What is the answer?")

    assert result["answer"] == "The answer is 42."
    assert result["num_sources"] == 1
    assert result["sources"][0]["content"] == "relevant info"
    assert result["sources"][0]["metadata"] == {"source": "doc.txt"}


def test_clear_delegates_to_vector_store(pipeline: RAGPipeline) -> None:
    pipeline.clear()
    pipeline._mock_vector_store.delete_collection.assert_called_once()
