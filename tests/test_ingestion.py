from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rag.config import Settings
from rag.ingestion import DocumentIngester

DUMMY_SETTINGS = Settings(openai_api_key="test-key")


@pytest.fixture()
def ingester() -> DocumentIngester:
    return DocumentIngester(settings=DUMMY_SETTINGS)


def test_load_text_creates_document(ingester: DocumentIngester) -> None:
    docs = ingester.load_text("Hello, world!")
    assert len(docs) == 1
    assert docs[0].page_content == "Hello, world!"
    assert docs[0].metadata == {}


def test_load_text_with_metadata(ingester: DocumentIngester) -> None:
    meta = {"source": "test", "author": "tester"}
    docs = ingester.load_text("Some content.", metadata=meta)
    assert len(docs) == 1
    assert docs[0].metadata == meta


def test_split_documents_creates_chunks(ingester: DocumentIngester) -> None:
    long_text = "word " * 600  # ~3000 chars — should produce multiple chunks
    docs = ingester.load_text(long_text)
    chunks = ingester.split_documents(docs)
    assert len(chunks) > 1


def test_load_file_txt(tmp_path: Path, ingester: DocumentIngester) -> None:
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("This is a sample text file.", encoding="utf-8")
    docs = ingester.load_file(str(txt_file))
    assert len(docs) == 1
    assert "sample text file" in docs[0].page_content


def test_load_file_unsupported_raises(
    tmp_path: Path, ingester: DocumentIngester
) -> None:
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingester.load_file(str(bad_file))
