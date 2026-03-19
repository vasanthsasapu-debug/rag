from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Settings, get_settings


class DocumentIngester:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )

    def load_text(self, text: str, metadata: dict | None = None) -> list[Document]:
        return [Document(page_content=text, metadata=metadata or {})]

    def load_file(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
            return [Document(page_content=text, metadata={"source": str(path)})]

        if suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(str(path))
            return loader.load()

        raise ValueError(f"Unsupported file type: {suffix!r}")

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)

    def ingest(
        self,
        source: str | Path | None = None,
        text: str | None = None,
        metadata: dict | None = None,
    ) -> list[Document]:
        if text is not None:
            docs = self.load_text(text, metadata)
        elif source is not None:
            docs = self.load_file(source)
        else:
            raise ValueError("Either 'source' or 'text' must be provided.")
        return self.split_documents(docs)
