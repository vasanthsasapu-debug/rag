from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag.config import Settings
from rag.retriever import Retriever

DUMMY_SETTINGS = Settings(openai_api_key="test-key")


@pytest.fixture()
def mock_vector_store() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def retriever(mock_vector_store: MagicMock) -> Retriever:
    return Retriever(vector_store=mock_vector_store, settings=DUMMY_SETTINGS)


def test_retrieve_calls_similarity_search(
    retriever: Retriever, mock_vector_store: MagicMock
) -> None:
    expected = [Document(page_content="relevant chunk")]
    mock_vector_store.similarity_search.return_value = expected

    results = retriever.retrieve("What is RAG?")

    mock_vector_store.similarity_search.assert_called_once_with(
        "What is RAG?", k=DUMMY_SETTINGS.retrieval_k
    )
    assert results == expected


def test_retrieve_with_scores(
    retriever: Retriever, mock_vector_store: MagicMock
) -> None:
    doc = Document(page_content="scored chunk")
    expected = [(doc, 0.92)]
    mock_vector_store.similarity_search_with_score.return_value = expected

    results = retriever.retrieve_with_scores("explain embeddings", k=2)

    mock_vector_store.similarity_search_with_score.assert_called_once_with(
        "explain embeddings", k=2
    )
    assert results == expected
